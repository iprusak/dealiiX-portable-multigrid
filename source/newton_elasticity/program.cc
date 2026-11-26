// Newton's method for hyperelasticity (Saint Venant-Kirchhoff or Ogden) with
// Portable::ElasticityOperator: a beam [0, 4] x [0, 1] x [0, 1], clamped at
// x = 0, under the body force (0, 0, -g), in load steps lambda = 1/n, ..., 1.
// Per Newton iteration the residual R(u) = A(u) - lambda F by
// evaluate_residual(), which stores grad u, and the correction from
// A'(u) du = -R(u) by CG on vmult_linearized() with the identity as
// preconditioner. Newton stops at |R| <= 1e-10 |lambda F|; the printed
// |R_k| / |R_{k-1}|^2 shows the quadratic convergence.
//
// With output = 1, u after each load step goes to solution_<step>.pvtu, and
// solution.pvd collects the steps for ParaView ("Warp By Vector" on u).
//
//   ./program [degree (default 2)] [n_refinements (default 2)] [n_load_steps (default 4)]
//             [g (default 3e-3)] [cg_tolerance (default 1e-10, relative)]
//             [output (default 1)] [--mesh=file.msh] [--stretch=d or dx,dy,dz] [--cg-monitor=N]
//             [--material=svk|ogden] [--ogden=mu1,alpha1,mu2,alpha2,kappa]
//
// SVK: E = 1, nu = 0.3. Ogden (decoupled, Portable::OgdenMaterial) by default
// the alveolar wall of Singh et al. (2024), Table 2, as in ExaDG's
// alveolar_hex application: mu1 = 6.24, alpha1 = 16.714, mu2 = 935,
// alpha2 = 4.456 (Pa), kappa = 2.1e4. If an update gives J <= 0 somewhere
// (residual NaN), the Newton step is halved, at most 10 times.
//
// --mesh reads a hex mesh (Gmsh) in place of the beam; boundary ids are set by
// position on the bounding box: 0/1 at x_min/x_max, 2/3 y, 4/5 z, 6 elsewhere,
// with symmetry conditions u_d = 0 on id 2 d; --stretch=dx,dy,dz (one value: the
// same in all directions) adds u_d = d_d on id 2 d + 1 for all d, also where
// d_d = 0: the box [0, L]^3 under the macroscopic deformation gradient
// F = I + diag(d / L), as BoundaryConditions = Symmetry of ExaDG's alveolar_hex.
// In load steps, each added to all of u as the homogeneous deformation; Newton
// then stops at |R| <= 1e-10 |R_0| of the step. --cg-monitor=N prints the CG
// residual every N iterations.
//
// meshes/alveolar_hex_small.msh: 55200 hexes of an idealized alveolar
// (Kelvin foam) geometry in [0, 1]^3, e.g.
//   mpirun -n 8 ./program 1 0 10 0 1e-8 1 --mesh=meshes/alveolar_hex_small.msh --stretch=0.1,0,0

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "operators/portable_elasticity_operator.h"

using namespace dealii;

using Number     = double;
using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;



// the operator linearized at the u of the last evaluate_residual(), for CG;
// with prescribed DoFs (indicator S, D = 1 - S), which the operator does not
// know about, D A D + S
template <typename Operator>
struct LinearizedOperator
{
  const Operator    &op;
  const VectorType  *prescribed = nullptr;
  const VectorType  *free       = nullptr;
  mutable VectorType tmp;

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    if (prescribed == nullptr)
      {
        op.vmult_linearized(dst, src);
        return;
      }
    tmp = src;
    tmp.scale(*free);
    op.vmult_linearized(dst, tmp);
    dst.scale(*free);
    tmp = src;
    tmp.scale(*prescribed);
    dst += tmp;
  }
};



constexpr int dim = 3;

// E = 1, nu = 0.3
constexpr double E = 1., nu = 0.3;
constexpr double lame_lambda = E * nu / ((1. + nu) * (1. - 2. * nu)),
                 lame_mu     = E / (2. * (1. + nu));



template <int degree, typename Material>
int
run(const unsigned int  n_refinements,
    const unsigned int  n_load_steps,
    const double        g,
    const double        cg_tolerance,
    const bool          output,
    const std::string  &mesh_file,
    const unsigned int  cg_monitor,
    const Tensor<1, 3> &stretch,
    const Material     &material,
    const std::string  &material_description)
{
  using Operator = Portable::ElasticityOperator<dim, degree, Number, Material>;

  // a mesh file with --stretch: the faces at the upper bounds are prescribed
  const bool has_stretch = !mesh_file.empty() && stretch.norm() != 0.;

  const MPI_Comm     comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  parallel::distributed::Triangulation<dim> tria(comm);
  BoundingBox<dim>                          box({Point<dim>(), Point<dim>(4., 1., 1.)});
  if (mesh_file.empty())
    GridGenerator::subdivided_hyper_rectangle(
      tria, {4, 1, 1}, Point<dim>(), Point<dim>(4., 1., 1.), /*colorize*/ true);
  else
    {
      Triangulation<dim> serial;
      GridIn<dim>        grid_in(serial);
      std::ifstream      in(mesh_file);
      AssertThrow(in, ExcMessage("cannot open " + mesh_file));
      grid_in.read_msh(in);

      box              = GridTools::compute_bounding_box(serial);
      const double tol = 1e-8 * box.side_length(0);
      for (const auto &cell : serial.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              face->set_boundary_id(6);
              const Point<dim> c = face->center();
              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (std::abs(c[d] - box.lower_bound(d)) < tol)
                    face->set_boundary_id(2 * d);
                  else if (std::abs(c[d] - box.upper_bound(d)) < tol)
                    face->set_boundary_id(2 * d + 1);
                }
            }
      tria.copy_triangulation(serial);
    }
  tria.refine_global(n_refinements);

  const FESystem<dim> fe(FE_Q<dim>{degree}, dim);
  const MappingQ<dim> mapping(degree);
  DoFHandler<dim>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // beam: clamped at x = 0 (boundary id 0 of the colorized box); mesh file:
  // symmetry, u_d = 0 on the face at the lower bound in direction d, and with
  // a stretch u_d prescribed on the face at the upper bound, by the driver:
  // Portable::MatrixFree reads constrained DoFs as zero, so the operator gets
  // the symmetry constraints only
  AffineConstraints<Number> constraints(locally_owned_dofs, locally_relevant_dofs);
  if (mesh_file.empty())
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(dim), constraints);
  else
    for (unsigned int d = 0; d < dim; ++d)
      {
        ComponentMask mask(dim, false);
        mask.set(d, true);
        VectorTools::interpolate_boundary_values(mapping,
                                                 dof_handler,
                                                 2 * d,
                                                 Functions::ZeroFunction<dim, Number>(dim),
                                                 constraints,
                                                 mask);
      }
  constraints.close();

  Operator op(dof_handler, constraints, material, false);

  VectorType u, residual, du, force, free_mask, stretch_direction, stretch_free;
  for (VectorType *v : {&u, &residual, &du, &force, &free_mask, &stretch_direction, &stretch_free})
    op.initialize_dof_vector(*v);

  // stretch_direction: 1 on the DoFs of u_d at the upper bound in direction d,
  // stretch_free = 1 - stretch_direction; free_mask: 0 on these and the
  // constrained DoFs (the residual rows evaluate_residual() sets to u)
  {
    LinearAlgebra::ReadWriteVector<Number> rw_mask(locally_owned_dofs),
      rw_stretch(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      {
        rw_mask(i)    = constraints.is_constrained(i) ? 0. : 1.;
        rw_stretch(i) = 0.;
      }
    if (has_stretch)
      for (unsigned int d = 0; d < dim; ++d)
        {
          ComponentMask mask(dim, false);
          mask.set(d, true);
          std::map<types::global_dof_index, Number> values;
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   2 * d + 1,
                                                   Functions::ConstantFunction<dim, Number>(1.,
                                                                                            dim),
                                                   values,
                                                   mask);
          for (const auto &[i, value] : values)
            if (locally_owned_dofs.is_element(i))
              rw_stretch(i) = value;
        }
    for (const auto i : locally_owned_dofs)
      if (rw_stretch(i) != 0.)
        rw_mask(i) = 0.;
    free_mask.import_elements(rw_mask, VectorOperation::insert);
    stretch_direction.import_elements(rw_stretch, VectorOperation::insert);
    for (const auto i : locally_owned_dofs)
      rw_stretch(i) = 1. - rw_stretch(i);
    stretch_free.import_elements(rw_stretch, VectorOperation::insert);
  }

  // the stretch as a homogeneous deformation, u_d = d_d (x_d - x_min,d) / L_d:
  // 0 on the symmetry faces, d_d on the stretched ones; prescribing it on the
  // faces only puts the whole load increment into the first layer of cells
  VectorType affine_stretch;
  op.initialize_dof_vector(affine_stretch);
  if (has_stretch)
    {
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> host(locally_owned_dofs,
                                                                         locally_relevant_dofs,
                                                                         comm);
      VectorTools::interpolate(mapping,
                               dof_handler,
                               VectorFunctionFromTensorFunctionObject<dim, Number>(
                                 [&](const Point<dim> &x)
                                 {
                                   Tensor<1, dim, Number> value;
                                   for (unsigned int d = 0; d < dim; ++d)
                                     value[d] = stretch[d] * (x[d] - box.lower_bound(d)) /
                                                box.side_length(d);
                                   return value;
                                 }),
                               host);
      constraints.set_zero(host);
      LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
      rw.import_elements(host, VectorOperation::insert);
      affine_stretch.import_elements(rw, VectorOperation::insert);
    }

  // F = (v, (0, 0, -g)), zero on the constrained DoFs
  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> force_host(locally_owned_dofs,
                                                                             locally_relevant_dofs,
                                                                             comm);
    Vector<Number>                                                body_force(dim);
    body_force[dim - 1] = -g;
    VectorTools::create_right_hand_side(mapping,
                                        dof_handler,
                                        QGauss<dim>(degree + 1),
                                        Functions::ConstantFunction<dim, Number>(body_force),
                                        force_host,
                                        constraints);
    force_host.compress(VectorOperation::add);
    constraints.set_zero(force_host);
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    rw.import_elements(force_host, VectorOperation::insert);
    force.import_elements(rw, VectorOperation::insert);
  }
  const double force_norm = force.l2_norm();

  pcout << "Newton: " << material_description
        << " mesh=" << (mesh_file.empty() ? "beam" : mesh_file) << " degree=" << degree
        << " cells=" << tria.n_global_active_cells() << " dofs=" << dof_handler.n_dofs()
        << " ranks=" << Utilities::MPI::n_mpi_processes(comm) << " g=" << g
        << " stretch=" << stretch << " load_steps=" << n_load_steps
        << " cg_tolerance=" << cg_tolerance << std::endl;

  LinearizedOperator<Operator> linearized{op};
  if (has_stretch)
    {
      linearized.prescribed = &stretch_direction;
      linearized.free       = &stretch_free;
      op.initialize_dof_vector(linearized.tmp);
    }
  const PreconditionIdentity identity;

  double       time_residual = 0., time_cg = 0.;
  unsigned int n_newton_total = 0, n_cg_total = 0, n_step_cuts = 0;

  constexpr unsigned int max_newton       = 30;
  constexpr double       newton_tolerance = 1e-10;
  constexpr unsigned int max_step_cuts    = 10;

  // R(u) - load F on the free DoFs; NaN where J <= 0 (Ogden)
  const auto compute_residual = [&](const double load)
  {
    Timer timer;
    op.evaluate_residual(residual, u);
    residual.add(-load, force);
    residual.scale(free_mask);
    const double norm = residual.l2_norm();
    Kokkos::fence();
    time_residual += timer.wall_time();
    return norm;
  };

  std::vector<std::pair<double, std::string>> pvd_records;
  const auto write_solution = [&](const unsigned int step, const double load)
  {
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    rw.import_elements(u, VectorOperation::insert);
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> u_host(locally_owned_dofs,
                                                                         locally_relevant_dofs,
                                                                         comm);
    u_host.import_elements(rw, VectorOperation::insert);
    constraints.distribute(u_host);
    u_host.update_ghost_values();

    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(u_host,
                             std::vector<std::string>(dim, "u"),
                             DataOut<dim>::type_dof_data,
                             std::vector<DataComponentInterpretation::DataComponentInterpretation>(
                               dim, DataComponentInterpretation::component_is_part_of_vector));
    data_out.build_patches(mapping, degree, DataOut<dim>::curved_inner_cells);

    pvd_records.emplace_back(load,
                             data_out.write_vtu_with_pvtu_record("./", "solution", step, comm, 2));
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        std::ofstream pvd("solution.pvd");
        DataOutBase::write_pvd_record(pvd, pvd_records);
      }
  };
  if (output)
    write_solution(0, 0.);

  for (unsigned int step = 1; step <= n_load_steps; ++step)
    {
      const double load = static_cast<double>(step) / n_load_steps;
      pcout << "load step " << step << ", lambda = " << load << std::endl;
      u.add(1. / n_load_steps, affine_stretch);
      double reference_norm = load * force_norm;

      double residual_norm = compute_residual(load);
      AssertThrow(std::isfinite(residual_norm),
                  ExcMessage("J <= 0 at the start of load step " + std::to_string(step) +
                             ": more load steps needed"));
      if (has_stretch)
        reference_norm = std::max(reference_norm, residual_norm);

      double       previous_norm = 0.;
      bool         converged     = false;
      unsigned int it            = 0;
      for (; it <= max_newton; ++it)
        {
          pcout << "  it " << std::setw(2) << it << "  |R| = " << std::scientific
                << std::setprecision(3) << residual_norm;
          if (it > 0)
            pcout << "  |R|/|R_prev|^2 = " << residual_norm / (previous_norm * previous_norm);
          pcout << std::defaultfloat;

          if (residual_norm <= newton_tolerance * reference_norm)
            {
              pcout << std::endl;
              converged = true;
              break;
            }

          residual *= -1.;
          SolverControl        control(100000, cg_tolerance * residual_norm);
          SolverCG<VectorType> cg(control);
          if (cg_monitor > 0)
            cg.connect(
              [&](const unsigned int i, const double r, const VectorType &)
              {
                if (i % cg_monitor == 0)
                  pcout << std::endl
                        << "      CG " << std::setw(6) << i << "  |r| = " << std::scientific
                        << std::setprecision(3) << r << "  |r|/|r_0| = " << r / residual_norm
                        << std::defaultfloat;
                return SolverControl::success;
              });
          du = 0.;
          Timer timer;
          cg.solve(linearized, du, residual, identity);
          Kokkos::fence();
          time_cg += timer.wall_time();
          pcout << "  CG " << control.last_step() << std::endl;
          n_cg_total += control.last_step();
          ++n_newton_total;
          previous_norm = residual_norm;

          // u + s du with s = 1, 1/2, ... until J > 0 everywhere
          u += du;
          double step_length = 1.;
          residual_norm      = compute_residual(load);
          for (unsigned int cut = 0; !std::isfinite(residual_norm); ++cut)
            {
              AssertThrow(cut < max_step_cuts,
                          ExcMessage("J <= 0 after " + std::to_string(max_step_cuts) +
                                     " step cuts"));
              step_length *= 0.5;
              u.add(-step_length, du);
              residual_norm = compute_residual(load);
              ++n_step_cuts;
              pcout << "        J <= 0: step length " << step_length << std::endl;
            }
        }
      AssertThrow(converged, ExcMessage("Newton did not converge"));
      if (output)
        write_solution(step, load);
    }

  pcout << "max |u| = " << u.linfty_norm() << ", Newton iterations " << n_newton_total
        << ", CG iterations " << n_cg_total << ", step cuts " << n_step_cuts << std::endl
        << "time: residual " << time_residual << " s, CG " << time_cg << " s ("
        << (n_cg_total > 0 ? 1e3 * time_cg / n_cg_total : 0.) << " ms per CG iteration)"
        << std::endl;

  return 0;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  std::string         mesh_file, material_name = "svk", ogden_parameters;
  unsigned int        cg_monitor = 0;
  Tensor<1, 3>        stretch;
  std::vector<char *> args;
  for (int i = 0; i < argc; ++i)
    {
      const std::string a = argv[i];
      if (a.rfind("--mesh=", 0) == 0)
        mesh_file = a.substr(7);
      else if (a.rfind("--stretch=", 0) == 0)
        {
          std::vector<double> d;
          std::stringstream   stream(a.substr(10));
          for (std::string item; std::getline(stream, item, ',');)
            d.push_back(std::stod(item));
          AssertThrow(d.size() == 1 || d.size() == 3,
                      ExcMessage("--stretch needs one or three values"));
          for (unsigned int i = 0; i < 3; ++i)
            stretch[i] = d[d.size() == 1 ? 0 : i];
        }
      else if (a.rfind("--cg-monitor=", 0) == 0)
        cg_monitor = std::stoi(a.substr(13));
      else if (a.rfind("--material=", 0) == 0)
        material_name = a.substr(11);
      else if (a.rfind("--ogden=", 0) == 0)
        ogden_parameters = a.substr(8);
      else
        args.push_back(argv[i]);
    }
  argc = args.size();
  argv = args.data();

  const int          degree        = (argc > 1) ? std::atoi(argv[1]) : 2;
  const unsigned int n_refinements = (argc > 2) ? std::atoi(argv[2]) : 2;
  const unsigned int n_load_steps  = (argc > 3) ? std::atoi(argv[3]) : 4;
  const double       g             = (argc > 4) ? std::atof(argv[4]) : 3e-3;
  const double       cg_tolerance  = (argc > 5) ? std::atof(argv[5]) : 1e-10;
  const bool         output        = (argc > 6) ? std::atoi(argv[6]) != 0 : true;

  const auto usage = [&]()
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cerr << "usage: " << argv[0]
                << " [degree in [1, 4] (default 2)] [n_refinements (default 2)]"
                << " [n_load_steps (default 4)] [g (default 3e-3)]"
                << " [cg_tolerance (default 1e-10)] [output (default 1)]"
                << " [--mesh=file.msh] [--stretch=d or dx,dy,dz] [--cg-monitor=N]"
                << " [--material=svk|ogden] [--ogden=mu1,alpha1,mu2,alpha2,kappa]" << std::endl;
    return 1;
  };

  const auto run_degree = [&](const auto &material, const std::string &description)
  {
    using Material      = std::decay_t<decltype(material)>;
    const auto run_with = [&](auto degree_constant)
    {
      return run<decltype(degree_constant)::value, Material>(n_refinements,
                                                             n_load_steps,
                                                             g,
                                                             cg_tolerance,
                                                             output,
                                                             mesh_file,
                                                             cg_monitor,
                                                             stretch,
                                                             material,
                                                             description);
    };
    switch (degree)
      {
        case 1:
          return run_with(std::integral_constant<int, 1>());
        case 2:
          return run_with(std::integral_constant<int, 2>());
        case 3:
          return run_with(std::integral_constant<int, 3>());
        case 4:
          return run_with(std::integral_constant<int, 4>());
        default:
          return usage();
      }
  };

  if (material_name == "svk")
    {
      const Portable::SaintVenantKirchhoffMaterial<dim, Number> material{lame_lambda, lame_mu};
      return run_degree(material,
                        "SVK lambda=" + std::to_string(lame_lambda) +
                          " mu=" + std::to_string(lame_mu));
    }
  else if (material_name == "ogden")
    {
      Portable::OgdenMaterial<dim, Number> material{6.24, 16.714, 935., 4.456, 2.1e4};
      if (!ogden_parameters.empty())
        {
          std::vector<double> p;
          std::stringstream   stream(ogden_parameters);
          for (std::string item; std::getline(stream, item, ',');)
            p.push_back(std::stod(item));
          if (p.size() != 5)
            return usage();
          material = {p[0], p[1], p[2], p[3], p[4]};
        }
      std::ostringstream description;
      description << "Ogden mu1=" << material.mu1 << " alpha1=" << material.alpha1
                  << " mu2=" << material.mu2 << " alpha2=" << material.alpha2
                  << " kappa=" << material.kappa;
      return run_degree(material, description.str());
    }
  else
    return usage();
}
