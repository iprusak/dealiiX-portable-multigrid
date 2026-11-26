// Newton's method for Saint Venant-Kirchhoff elasticity with
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
//             [output (default 1)] [--mesh=file.msh] [--stretch=delta] [--cg-monitor=N]
//
// --mesh reads a hex mesh (Gmsh) in place of the beam; boundary ids are set by
// position on the bounding box: 0/1 at x_min/x_max, 2/3 y, 4/5 z, 6 elsewhere,
// with symmetry conditions u_d = 0 on id 2 d; --stretch=delta adds u_d = delta
// on id 2 d + 1, in load steps, Newton then stops at |R| <= 1e-10 |R_0| of the
// step. --cg-monitor=N prints the CG residual every N
// iterations.

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
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
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



template <int degree>
int
run(const unsigned int n_refinements,
    const unsigned int n_load_steps,
    const double       g,
    const double       cg_tolerance,
    const bool         output,
    const std::string &mesh_file,
    const unsigned int cg_monitor,
    const double       stretch)
{
  constexpr int dim = 3;
  using Material    = Portable::SaintVenantKirchhoffMaterial<dim, Number>;
  using Operator    = Portable::ElasticityOperator<dim, degree, Number, Material>;

  const MPI_Comm     comm = MPI_COMM_WORLD;
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

  // E = 1, nu = 0.3
  const double   E = 1., nu = 0.3;
  const Material material{E * nu / ((1. + nu) * (1. - 2. * nu)), E / (2. * (1. + nu))};

  parallel::distributed::Triangulation<dim> tria(comm);
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

      const auto   box = GridTools::compute_bounding_box(serial);
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
    if (stretch != 0.)
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

  pcout << "SVK Newton: mesh=" << (mesh_file.empty() ? "beam" : mesh_file) << " degree=" << degree
        << " cells=" << tria.n_global_active_cells() << " dofs=" << dof_handler.n_dofs()
        << " ranks=" << Utilities::MPI::n_mpi_processes(comm) << " lambda=" << material.lambda
        << " mu=" << material.mu << " g=" << g << " stretch=" << stretch
        << " load_steps=" << n_load_steps << " cg_tolerance=" << cg_tolerance << std::endl;

  LinearizedOperator<Operator> linearized{op};
  if (stretch != 0.)
    {
      linearized.prescribed = &stretch_direction;
      linearized.free       = &stretch_free;
      op.initialize_dof_vector(linearized.tmp);
    }
  const PreconditionIdentity identity;

  double       time_residual = 0., time_cg = 0.;
  unsigned int n_newton_total = 0, n_cg_total = 0;

  constexpr unsigned int max_newton       = 30;
  constexpr double       newton_tolerance = 1e-10;

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
      u.add(stretch / n_load_steps, stretch_direction);
      double reference_norm = load * force_norm;

      double       previous_norm = 0.;
      bool         converged     = false;
      unsigned int it            = 0;
      for (; it <= max_newton; ++it)
        {
          Timer timer;
          op.evaluate_residual(residual, u);
          residual.add(-load, force);
          residual.scale(free_mask);
          const double residual_norm = residual.l2_norm();
          if (it == 0 && stretch != 0.)
            reference_norm = std::max(reference_norm, residual_norm);
          Kokkos::fence();
          time_residual += timer.wall_time();

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
          timer.restart();
          cg.solve(linearized, du, residual, identity);
          Kokkos::fence();
          time_cg += timer.wall_time();
          u += du;

          pcout << "  CG " << control.last_step() << std::endl;
          n_cg_total += control.last_step();
          ++n_newton_total;
          previous_norm = residual_norm;
        }
      AssertThrow(converged, ExcMessage("Newton did not converge"));
      if (output)
        write_solution(step, load);
    }

  pcout << "max |u| = " << u.linfty_norm() << ", Newton iterations " << n_newton_total
        << ", CG iterations " << n_cg_total << std::endl
        << "time: residual " << time_residual << " s, CG " << time_cg << " s ("
        << (n_cg_total > 0 ? 1e3 * time_cg / n_cg_total : 0.) << " ms per CG iteration)"
        << std::endl;

  return 0;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  std::string         mesh_file;
  unsigned int        cg_monitor = 0;
  double              stretch    = 0.;
  std::vector<char *> args;
  for (int i = 0; i < argc; ++i)
    {
      const std::string a = argv[i];
      if (a.rfind("--mesh=", 0) == 0)
        mesh_file = a.substr(7);
      else if (a.rfind("--stretch=", 0) == 0)
        stretch = std::stod(a.substr(10));
      else if (a.rfind("--cg-monitor=", 0) == 0)
        cg_monitor = std::stoi(a.substr(13));
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

  switch (degree)
    {
      case 1:
        return run<1>(
          n_refinements, n_load_steps, g, cg_tolerance, output, mesh_file, cg_monitor, stretch);
      case 2:
        return run<2>(
          n_refinements, n_load_steps, g, cg_tolerance, output, mesh_file, cg_monitor, stretch);
      case 3:
        return run<3>(
          n_refinements, n_load_steps, g, cg_tolerance, output, mesh_file, cg_monitor, stretch);
      case 4:
        return run<4>(
          n_refinements, n_load_steps, g, cg_tolerance, output, mesh_file, cg_monitor, stretch);
      default:
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cerr << "usage: " << argv[0]
                    << " [degree in [1, 4] (default 2)] [n_refinements (default 2)]"
                    << " [n_load_steps (default 4)] [g (default 3e-3)]"
                    << " [cg_tolerance (default 1e-10)] [output (default 1)]"
                    << " [--mesh=file.msh] [--stretch=delta] [--cg-monitor=N]" << std::endl;
        return 1;
    }
}
