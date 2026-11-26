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
//             [output (default 1)] [--mesh=file.msh] [--stretch=delta] [--no-predictor]
//             [--cg-monitor=N] [--pmg] [--outer=cg|fcg] [--coarse-tol=tol]
//             [--smoothing-degree=k]
//
// --mesh reads a hex mesh (Gmsh) in place of the beam; boundary ids are set by
// position on the bounding box: 0/1 at x_min/x_max, 2/3 y, 4/5 z, 6 elsewhere,
// with symmetry conditions u_d = 0 on id 2 d; --stretch=delta adds u_d = delta
// on id 2 d + 1, in load steps; each step adds delta / n_load_steps x on the
// free DoFs as predictor (off with --no-predictor), and Newton stops at
// |R| <= 1e-10 |R_jump|, R_jump the residual after adding the boundary values
// alone. --cg-monitor=N prints the CG residual every N
// iterations.
//
// --pmg preconditions CG by polynomial multigrid (tangent_pmg.h): degrees
// p, p / 2, ..., 1, Chebyshev of degree --smoothing-degree (default 5), CG to
// the relative --coarse-tol (default 1e-3) on degree 1, flexible CG outside
// (--outer=fcg, default) or plain CG (--outer=cg).
//
// meshes/alveolar_hex_small.msh: 55200 hexes of an idealized alveolar
// (Kelvin foam) geometry in [0, 1]^3, e.g.
//   mpirun -n 8 ./program 1 0 10 0 1e-8 1 --mesh=meshes/alveolar_hex_small.msh --stretch=0.5

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
#include "operators/portable_elasticity_tangent_level_operator.h"
#include "tangent_pmg.h"

using namespace dealii;

using Number     = double;
using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;



// the position x as a vector field
template <int dim>
struct PositionFunction : public Function<dim, Number>
{
  PositionFunction()
    : Function<dim, Number>(dim)
  {}

  Number
  value(const Point<dim> &p, const unsigned int component) const override
  {
    return p[component];
  }
};



struct Options
{
  unsigned int n_refinements = 2;
  unsigned int n_load_steps  = 4;
  double       g             = 3e-3;
  double       cg_tolerance  = 1e-10;
  bool         output        = true;
  std::string  mesh_file;
  unsigned int cg_monitor       = 0;
  double       stretch          = 0.;
  bool         predictor        = true;
  bool         pmg              = false;
  bool         flexible_cg      = true;
  double       coarse_tolerance = 1e-3;
  unsigned int smoothing_degree = 5;
};



template <int degree>
int
run(const Options &options)
{
  constexpr int dim = 3;
  using Material    = Portable::SaintVenantKirchhoffMaterial<dim, Number>;
  using Operator    = Portable::ElasticityOperator<dim, degree, Number, Material>;
  using FineLevel   = Portable::ElasticityTangentLevelOperator<dim, degree, Number, Material>;

  const unsigned int n_refinements = options.n_refinements;
  const unsigned int n_load_steps  = options.n_load_steps;
  const double       g             = options.g;
  const double       cg_tolerance  = options.cg_tolerance;
  const bool         output        = options.output;
  const std::string &mesh_file     = options.mesh_file;
  const unsigned int cg_monitor    = options.cg_monitor;
  const double       stretch       = options.stretch;
  const bool         predictor     = options.predictor;

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
  // the symmetry constraints only. The same for every multigrid level.
  const auto level_setup = [&](const DoFHandler<dim>     &level_dof_handler,
                               AffineConstraints<Number> &level_constraints,
                               IndexSet                  &prescribed)
  {
    const MappingQ<dim> level_mapping(1);
    level_constraints.reinit(level_dof_handler.locally_owned_dofs(),
                             DoFTools::extract_locally_relevant_dofs(level_dof_handler));
    prescribed = IndexSet(level_dof_handler.n_dofs());
    if (mesh_file.empty())
      VectorTools::interpolate_boundary_values(level_mapping,
                                               level_dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim, Number>(dim),
                                               level_constraints);
    else
      for (unsigned int d = 0; d < dim; ++d)
        {
          ComponentMask mask(dim, false);
          mask.set(d, true);
          VectorTools::interpolate_boundary_values(level_mapping,
                                                   level_dof_handler,
                                                   2 * d,
                                                   Functions::ZeroFunction<dim, Number>(dim),
                                                   level_constraints,
                                                   mask);
          if (stretch != 0.)
            {
              std::map<types::global_dof_index, Number> values;
              VectorTools::interpolate_boundary_values(
                level_mapping,
                level_dof_handler,
                2 * d + 1,
                Functions::ConstantFunction<dim, Number>(1., dim),
                values,
                mask);
              for (const auto &[i, value] : values)
                prescribed.add_index(i);
            }
        }
    level_constraints.close();
  };

  AffineConstraints<Number> constraints;
  IndexSet                  prescribed;
  level_setup(dof_handler, constraints, prescribed);

  FineLevel fine_level(dof_handler, constraints, material, prescribed);
  Operator &op = fine_level.get_operator();

  std::unique_ptr<TangentPMG<dim, Number, Material>> pmg;
  if (options.pmg)
    {
      pmg = std::make_unique<TangentPMG<dim, Number, Material>>(fine_level,
                                                                constraints,
                                                                prescribed,
                                                                material,
                                                                level_setup,
                                                                options.smoothing_degree,
                                                                options.coarse_tolerance);
      pcout << "p-multigrid: degrees";
      for (unsigned int l = pmg->n_levels(); l-- > 0;)
        pcout << " " << pmg->degree(l);
      pcout << ", Chebyshev degree " << options.smoothing_degree << ", coarse CG to "
            << options.coarse_tolerance << ", outer " << (options.flexible_cg ? "FCG" : "CG")
            << std::endl;
    }


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
    for (const auto i : locally_owned_dofs)
      if (prescribed.is_element(i))
        rw_stretch(i) = 1.;
    for (const auto i : locally_owned_dofs)
      if (rw_stretch(i) != 0.)
        rw_mask(i) = 0.;
    free_mask.import_elements(rw_mask, VectorOperation::insert);
    stretch_direction.import_elements(rw_stretch, VectorOperation::insert);
    for (const auto i : locally_owned_dofs)
      rw_stretch(i) = 1. - rw_stretch(i);
    stretch_free.import_elements(rw_stretch, VectorOperation::insert);
  }

  // affine predictor: x on the free DoFs, the homogeneous stretch u = delta x
  // is added per load step, so the step does not start from a jump of the
  // boundary values
  VectorType affine;
  op.initialize_dof_vector(affine);
  if (stretch != 0. && predictor)
    {
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> x_host(locally_owned_dofs,
                                                                           locally_relevant_dofs,
                                                                           comm);
      VectorTools::interpolate(mapping, dof_handler, PositionFunction<dim>(), x_host);
      LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
      rw.import_elements(x_host, VectorOperation::insert);
      affine.import_elements(rw, VectorOperation::insert);
      affine.scale(free_mask);
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

  const PreconditionIdentity identity;

  double       time_residual = 0., time_cg = 0.;
  unsigned int n_newton_total = 0, n_cg_total = 0, n_coarse_total = 0;

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
      // the residual of the boundary jump alone as reference, as without the
      // predictor
      if (stretch != 0.)
        {
          op.evaluate_residual(residual, u);
          residual.add(-load, force);
          residual.scale(free_mask);
          reference_norm = std::max(reference_norm, residual.l2_norm());
          u.add(stretch / n_load_steps, affine);
        }

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
          SolverControl                control(100000, cg_tolerance * residual_norm);
          SolverCG<VectorType>         cg(control);
          SolverFlexibleCG<VectorType> fcg(control);
          SolverBase<VectorType>      &solver =
            (pmg && options.flexible_cg) ? static_cast<SolverBase<VectorType> &>(fcg) : cg;
          if (cg_monitor > 0)
            solver.connect(
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
          if (pmg)
            {
              pmg->update(u);
              pmg->n_coarse_iterations = 0;
              if (options.flexible_cg)
                fcg.solve(fine_level, du, residual, *pmg);
              else
                cg.solve(fine_level, du, residual, *pmg);
            }
          else
            cg.solve(fine_level, du, residual, identity);
          Kokkos::fence();
          time_cg += timer.wall_time();
          u += du;

          pcout << "  CG " << control.last_step();
          if (pmg)
            {
              pcout << " (coarse CG " << pmg->n_coarse_iterations << ")";
              n_coarse_total += pmg->n_coarse_iterations;
            }
          pcout << std::endl;
          n_cg_total += control.last_step();
          ++n_newton_total;
          previous_norm = residual_norm;
        }
      AssertThrow(converged, ExcMessage("Newton did not converge"));
      if (output)
        write_solution(step, load);
    }

  pcout << "max |u| = " << u.linfty_norm() << ", Newton iterations " << n_newton_total
        << ", CG iterations " << n_cg_total;
  if (pmg)
    pcout << ", coarse CG iterations " << n_coarse_total;
  pcout << std::endl
        << "time: residual " << time_residual << " s, CG " << time_cg << " s ("
        << (n_cg_total > 0 ? 1e3 * time_cg / n_cg_total : 0.) << " ms per CG iteration)"
        << std::endl;

  return 0;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Options             options;
  std::vector<char *> args;
  for (int i = 0; i < argc; ++i)
    {
      const std::string a = argv[i];
      if (a.rfind("--mesh=", 0) == 0)
        options.mesh_file = a.substr(7);
      else if (a == "--no-predictor")
        options.predictor = false;
      else if (a.rfind("--stretch=", 0) == 0)
        options.stretch = std::stod(a.substr(10));
      else if (a.rfind("--cg-monitor=", 0) == 0)
        options.cg_monitor = std::stoi(a.substr(13));
      else if (a == "--pmg")
        options.pmg = true;
      else if (a == "--outer=cg")
        options.flexible_cg = false;
      else if (a == "--outer=fcg")
        options.flexible_cg = true;
      else if (a.rfind("--coarse-tol=", 0) == 0)
        options.coarse_tolerance = std::stod(a.substr(13));
      else if (a.rfind("--smoothing-degree=", 0) == 0)
        options.smoothing_degree = std::stoi(a.substr(19));
      else
        args.push_back(argv[i]);
    }
  argc = args.size();
  argv = args.data();

  const int degree = (argc > 1) ? std::atoi(argv[1]) : 2;
  if (argc > 2)
    options.n_refinements = std::atoi(argv[2]);
  if (argc > 3)
    options.n_load_steps = std::atoi(argv[3]);
  if (argc > 4)
    options.g = std::atof(argv[4]);
  if (argc > 5)
    options.cg_tolerance = std::atof(argv[5]);
  if (argc > 6)
    options.output = std::atoi(argv[6]) != 0;

  switch (degree)
    {
      case 1:
        return run<1>(options);
      case 2:
        return run<2>(options);
      case 3:
        return run<3>(options);
      case 4:
        return run<4>(options);
      default:
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cerr << "usage: " << argv[0]
                    << " [degree in [1, 4] (default 2)] [n_refinements (default 2)]"
                    << " [n_load_steps (default 4)] [g (default 3e-3)]"
                    << " [cg_tolerance (default 1e-10)] [output (default 1)]"
                    << " [--mesh=file.msh] [--stretch=delta] [--no-predictor] [--cg-monitor=N]"
                    << " [--pmg] [--outer=cg|fcg] [--coarse-tol=tol] [--smoothing-degree=k]"
                    << std::endl;
        return 1;
    }
}
