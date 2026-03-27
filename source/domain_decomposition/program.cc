#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "base/portable_mg_transfer_base.h"
#include "base/portable_subdomain_laplace_operator_base.h"
#include "domain_decomposition/portable_bnn_preconditioner.h"
#include "domain_decomposition/portable_schur_interface_operator.h"
#include "domain_decomposition/portable_solver_projected_cg.h"
#include "domain_decomposition/subdomain_dof_handler.h"
#include "domain_decomposition/subdomain_triangulation.h"
#include "multigrid/portable_geometric_transfer.h"
#include "multigrid/portable_polynomial_tranfer.h"
#include "multigrid/portable_subdomain_v_cycle_multigrid.h"
#include "operators/portable_subdomain_laplace_operator.h"



using namespace dealii;


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem(const unsigned int n_pre_smooth,
                 const unsigned int n_post_smooth);

  void
  run();

private:
  void
  create_subdomain_triangulations(unsigned int n_refinement_cycles);

  void
  setup_dofs();

  void
  compute_interface_weights();

  void
  setup_matrix_free();

  void
  setup_mg_transfers();

  void
  setup_smoothers();

  void
  setup_mg_preconditioners();

  void
  setup_interface_system();

  void
  setup_bnn_preconditioner();

  void
  assemble_rhs();

  void
  solve_interface();

  void
  postprocess_subdomain_solution();

  void
  output_results(const unsigned int cycle) const;

  void
  test_coarse_problem();

  void
  test_triangulation(int n_refinements);

  MPI_Comm mpi_communicator;

  // parallel::distributed::Triangulation<dim> triangulation;
  parallel::fullydistributed::Triangulation<dim> triangulation;

  // SubdomainTriangulation<dim>                    subdomain_triangulation;



  DoFHandler<dim>          dof_handler;
  SubdomainDoFHandler<dim> subdomain_dof_handler;

  FE_Q<dim> fe;

  // AffineConstraints<double> subdomain_constraints;
  // AffineConstraints<double> subdomain_constraints_physical;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  std::vector<std::shared_ptr<SubdomainTriangulation<dim>>>
    level_subdomain_triangulations;

  std::vector<std::shared_ptr<parallel::fullydistributed::Triangulation<dim>>>
    level_triangulations;

  MGLevelObject<SubdomainDoFHandler<dim>> level_subdomain_dof_handlers;
  MGLevelObject<DoFHandler<dim>>          level_dof_handlers;

  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints;
  MGLevelObject<AffineConstraints<double>> level_subdomain_constraints_physical;

  MGLevelObject<
    std::unique_ptr<Portable::SubdomainLaplaceOperatorBase<dim, double>>>
    level_subdomain_matrices;

  MGLevelObject<std::unique_ptr<Portable::MGTransferBase<dim, double>>>
    subdomain_mg_transfers;


  using VectorTypeMG =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  using SmootherType =
    PreconditionChebyshev<Portable::SubdomainLaplaceOperatorBase<dim, double>,
                          VectorTypeMG>;

  MGLevelObject<SmootherType> subdomain_mg_smoothers;

  std::unique_ptr<
    Portable::SubdomainVCycleMultigrid<dim,
                                       double,
                                       Portable::MGTransferBase<dim, double>>>
    subdomain_dirichlet_mg_preconditioner;


  std::unique_ptr<Portable::SchurInterfaceOperator<dim, fe_degree, double>>
    interface_operator;

  std::unique_ptr<Portable::BNNPreconditioner<dim, fe_degree, double>>
    bnn_preconditioner;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    global_solution_host, subdomain_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    subdomain_rhs_device, schur_rhs;

  std::unique_ptr<Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>
    subdomain_matrix;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    rhs_schur_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_interface_device;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    global_interface_weights;

  const unsigned int n_pre_smooth;
  const unsigned int n_post_smooth;

  double             setup_time;
  ConditionalOStream pcout;
  ConditionalOStream time_details;
};
template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem(const unsigned int n_pre_smooth,
                                               const unsigned int n_post_smooth)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , fe(fe_degree)
  , n_pre_smooth(n_pre_smooth)
  , n_post_smooth(n_post_smooth)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout,
                 true &&
                   Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{
  Assert(n_pre_smooth == n_post_smooth,
         ExcNotImplemented("Change of pre- and post-smoother degree "
                           "currently not possible with deal.II"));
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::create_subdomain_triangulations(
  unsigned int n_refinement_cycles)
{
  Timer time;

  const unsigned int n_subdomains =
    Utilities::MPI::n_mpi_processes(mpi_communicator);

  std::vector<unsigned int> subdomains_per_axis(dim);

  int remaining = n_subdomains;
  for (int d = dim; d > 0; --d)
    {
      int n_this_axis = std::pow(remaining, 1.0 / d);

      int best_divisor = 1;
      for (int j = n_this_axis; j >= 1; --j)
        if (remaining % j == 0)
          {
            best_divisor = j;
            break;
          }
      subdomains_per_axis[d - 1] = best_divisor;

      remaining /= best_divisor;
    }


  Triangulation<dim> coarse_triangulation;

  Point<dim> p1, p2;
  for (int d = 0; d < dim; ++d)
    p2[d] = 1.;

  GridGenerator::subdivided_hyper_rectangle(coarse_triangulation,
                                            subdomains_per_axis,
                                            p1,
                                            p2);

  unsigned int cell_counter = 0;
  for (auto cell : coarse_triangulation.active_cell_iterators())
    cell->set_subdomain_id(cell_counter++);

  this->level_subdomain_triangulations.clear();
  this->level_triangulations.clear();

  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      if (cycle > 0)
        coarse_triangulation.refine_global(1);

      const TriangulationDescription::Description<dim> description =
        TriangulationDescription::Utilities::
          create_description_from_triangulation(coarse_triangulation,
                                                mpi_communicator);

      this->triangulation.clear();
      this->triangulation.create_triangulation(description);

      this->level_triangulations.push_back(
        std::make_shared<parallel::fullydistributed::Triangulation<dim>>(
          mpi_communicator));
      this->level_triangulations.back()->create_triangulation(description);

      this->level_subdomain_triangulations.push_back(
        std::make_shared<SubdomainTriangulation<dim>>());

      if (cycle == 0)
        this->level_subdomain_triangulations.back()
          ->create_subdomain_triangulation(*level_triangulations.back());
      else
        {
          this->level_subdomain_triangulations.back()
            ->copy_subdomain_triangulation(
              *level_subdomain_triangulations[cycle - 1]);
          this->level_subdomain_triangulations.back()->refine_global(1);
        }
    }
  setup_time += time.wall_time();
  time_details << "           Subdomain triangulations extracted    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  const double subdomain_diameter = Utilities::MPI::max(
    GridTools::diameter(
      level_subdomain_triangulations.back()->get_triangulation()),
    mpi_communicator);

  const double subdomain_mesh_size = Utilities::MPI::max(
    GridTools::maximal_cell_diameter(
      level_subdomain_triangulations.back()->get_triangulation()),
    mpi_communicator);


  pcout << "H/h = " << subdomain_diameter / subdomain_mesh_size << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  Timer time;

  dof_handler.clear();
  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);


  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  subdomain_dof_handler.reinit(level_subdomain_triangulations.back(),
                               dof_handler);
  subdomain_dof_handler.distribute_subdomain_dofs();


  pcout << "           Total number of DoFs: " << dof_handler.n_dofs()
        << std::endl;

  level_dof_handlers.resize(0, level_subdomain_triangulations.size() - 1);
  level_subdomain_dof_handlers.resize(0,
                                      level_subdomain_triangulations.size() -
                                        1);


  level_subdomain_constraints.resize(0, level_dof_handlers.max_level());
  level_subdomain_constraints_physical.resize(0,
                                              level_dof_handlers.max_level());


  Functions::ZeroFunction<dim> homogeneous_dirichlet_bc;
  std::map<types::boundary_id, const Function<dim> *>
    dirichlet_boundary_functions = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc},
      {subdomain_dof_handler.get_interface_id(), &homogeneous_dirichlet_bc}};
  std::map<types::boundary_id, const Function<dim> *>
    dirichlet_boundary_functions_physical = {
      {types::boundary_id(0), &homogeneous_dirichlet_bc}};

  for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
    {
      DoFHandler<dim> &dof_h = level_dof_handlers[level];
      dof_h.reinit(*level_triangulations[level]);
      dof_h.distribute_dofs(fe);

      SubdomainDoFHandler<dim> &subdomain_dof_h =
        level_subdomain_dof_handlers[level];

      subdomain_dof_h.reinit(level_subdomain_triangulations[level], dof_h);
      subdomain_dof_h.distribute_subdomain_dofs();

      {
        AffineConstraints<double> &constraints =
          level_subdomain_constraints[level];

        constraints.clear();

        DoFTools::make_hanging_node_constraints(
          subdomain_dof_h.get_dof_handler(), constraints);
        VectorTools::interpolate_boundary_values(
          subdomain_dof_h.get_dof_handler(),
          dirichlet_boundary_functions,
          constraints);
        constraints.close();
      }

      {
        AffineConstraints<double> &constraints_physical =
          level_subdomain_constraints_physical[level];

        constraints_physical.clear();

        DoFTools::make_hanging_node_constraints(
          subdomain_dof_h.get_dof_handler(), constraints_physical);
        VectorTools::interpolate_boundary_values(
          subdomain_dof_h.get_dof_handler(),
          dirichlet_boundary_functions_physical,
          constraints_physical);
        constraints_physical.close();
      }
    }

  global_solution_host.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator);

  time_details << "           Distributed DoFs setup              (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
  time.restart();

  subdomain_solution_host.reinit(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  setup_time += time.wall_time();

  time_details << "           Subdomain DoFs setup                (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_interface_weights()
{
  subdomain_dof_handler.initialize_interface_dof_vector(
    global_interface_weights);

  const unsigned int n_locally_relevant_interface_indices =
    this->subdomain_dof_handler.n_locally_relevant_interface_indices();
  for (unsigned int i = 0; i < n_locally_relevant_interface_indices; ++i)
    global_interface_weights[this->subdomain_dof_handler
                               .local_to_global_interface_partitioner(i)] +=
      1.0;

  global_interface_weights.compress(VectorOperation::add);

  for (unsigned int i = 0; i < global_interface_weights.locally_owned_size();
       ++i)
    global_interface_weights.local_element(i) =
      1. / global_interface_weights.local_element(i);

  global_interface_weights.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  Timer time;

  level_subdomain_matrices.resize(0, level_dof_handlers.max_level());

  for (unsigned int level = 0; level <= level_dof_handlers.max_level(); ++level)
    {
      level_subdomain_matrices[level] = std::make_unique<
        Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>(
        level_subdomain_dof_handlers[level],
        level_subdomain_constraints[level]);
    }

  subdomain_matrix = std::make_unique<
    Portable::SubdomainLaplaceOperator<dim, fe_degree, double>>(
    subdomain_dof_handler, level_subdomain_constraints.back());

  subdomain_matrix->initialize_dof_vector(subdomain_solution_device);
  subdomain_matrix->initialize_dof_vector(subdomain_rhs_device);

  setup_time += time.wall_time();
  time_details << "           Matrix-free operators setup          (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_transfers()
{
  subdomain_mg_transfers.resize(level_subdomain_matrices.min_level(),
                                level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level() + 1;
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      subdomain_mg_transfers[level] =
        std::make_unique<Portable::GeometricTransfer<dim, fe_degree, double>>();
      subdomain_mg_transfers[level]->reinit(
        level_subdomain_matrices[level - 1]->get_matrix_free(),
        level_subdomain_matrices[level]->get_matrix_free(),
        level_subdomain_constraints[level - 1],
        level_subdomain_constraints[level]);
    }
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_smoothers()
{
  subdomain_mg_smoothers.resize(level_subdomain_matrices.min_level(),
                                level_subdomain_matrices.max_level());

  for (unsigned int level = level_subdomain_matrices.min_level();
       level <= level_subdomain_matrices.max_level();
       ++level)
    {
      typename SmootherType::AdditionalData smoother_data;
      if (level > 0)
        {
          smoother_data.smoothing_range     = 15.;
          smoother_data.degree              = n_pre_smooth;
          smoother_data.eig_cg_n_iterations = 10;
        }
      else
        {
          smoother_data.smoothing_range     = 1e-3;
          smoother_data.degree              = numbers::invalid_unsigned_int;
          smoother_data.eig_cg_n_iterations = level_subdomain_matrices[0]->m();
        }

      level_subdomain_matrices[level]->compute_diagonal();
      smoother_data.preconditioner =
        level_subdomain_matrices[level]->get_matrix_diagonal_inverse();

      subdomain_mg_smoothers[level].initialize(*level_subdomain_matrices[level],
                                               smoother_data);
    }
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_mg_preconditioners()
{
  subdomain_dirichlet_mg_preconditioner = std::make_unique<
    Portable::SubdomainVCycleMultigrid<dim,
                                       double,
                                       Portable::MGTransferBase<dim, double>>>(
    level_subdomain_matrices, subdomain_mg_transfers, subdomain_mg_smoothers);
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_interface_system()
{
  Kokkos::fence();
  Timer time;


  interface_operator =
    std::make_unique<Portable::SchurInterfaceOperator<dim, fe_degree, double>>(
      *subdomain_matrix, *subdomain_dirichlet_mg_preconditioner);

  rhs_schur_device.reinit(
    this->subdomain_dof_handler.get_interface_vector_partitioner());

  solution_interface_device.reinit(
    this->subdomain_dof_handler.get_interface_vector_partitioner());

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Interface system setup              (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_bnn_preconditioner()
{
  Timer time;

  Kokkos::fence();
  this->bnn_preconditioner =
    std::make_unique<Portable::BNNPreconditioner<dim, fe_degree, double>>(
      *interface_operator, *subdomain_matrix);
  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           BNN preconditioner setup            (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  Kokkos::fence();
  time.restart();
  this->bnn_preconditioner->setup_coarse_matrix();

  setup_time += time.wall_time();
  time_details << "           Coarse matrix for BNN computed      (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  Kokkos::fence();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::assemble_rhs()
{
  Timer time;
  Kokkos::fence();

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell :
       subdomain_dof_handler.get_dof_handler().active_cell_iterators())
    {
      cell_rhs = 0;

      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          cell_rhs(i) +=
            (fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index));

      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        system_rhs_host[local_dof_indices[i]] += cell_rhs[i];
    }

  for (const auto &index :
       subdomain_dof_handler.get_dof_info().subdomain_physical_boundary_dofs)
    system_rhs_host[index] = 0.;


  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.get_dof_handler().n_dofs());

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  subdomain_rhs_device.import_elements(rw_vector, VectorOperation::insert);

  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           RHS assembled                       (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;

  Kokkos::fence();
  this->interface_operator->assemble_rhs_schur(rhs_schur_device,
                                               subdomain_rhs_device);
  Kokkos::fence();
  setup_time += time.wall_time();
  time_details << "           Schur RHS assembled                 (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve_interface()
{
  Timer time;
  Kokkos::fence();
  ReductionControl solver_control(1000, 1e-12, 1e-7);

  Portable::SolverProjectedCG<
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
    cg(solver_control);

  solution_interface_device = 0.;
  cg.solve(*interface_operator,
           solution_interface_device,
           rhs_schur_device,
           *bnn_preconditioner);

  solution_interface_device.update_ghost_values();

  Kokkos::fence();

  pcout << "           Interface solver converged in "
        << solver_control.last_step() << " iterations.    (CPU/wall) "
        << time.cpu_time() << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::postprocess_subdomain_solution()
{
  Timer time;
  Kokkos::fence();
  this->interface_operator->reconstruct_subdomain_solution_from_interface(
    subdomain_solution_device, solution_interface_device, subdomain_rhs_device);

  LinearAlgebra::ReadWriteVector<double> rw_vector(
    subdomain_dof_handler.get_dof_handler().n_dofs());
  rw_vector.import_elements(subdomain_solution_device, VectorOperation::insert);
  subdomain_solution_host.import_elements(rw_vector, VectorOperation::insert);

  subdomain_solution_host.update_ghost_values();

  level_subdomain_constraints_physical.back().distribute(
    subdomain_solution_host);

  const auto &subdomain_to_global_dof_map =
    subdomain_dof_handler.get_dof_info().subdomain_to_global_dof_map;

  for (unsigned int i = 0; i < subdomain_to_global_dof_map.size(); ++i)
    {
      const auto global_index = subdomain_to_global_dof_map[i];

      global_solution_host[global_index] = subdomain_solution_host[i];
    }

  global_solution_host.compress(VectorOperation::add);


  for (unsigned int i = 0;
       i < subdomain_dof_handler.n_locally_relevant_interface_indices();
       ++i)
    {
      const auto subdomain_index =
        subdomain_dof_handler.local_interface_to_subdomain(i);
      const auto global_index = subdomain_to_global_dof_map[subdomain_index];
      global_solution_host[global_index] *=
        global_interface_weights[subdomain_dof_handler
                                   .local_to_global_interface_partitioner(i)];
    }

  global_solution_host.update_ghost_values();

  Kokkos::fence();
  time_details << "           Subdomain solution post-processed    (CPU/wall) "
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
{
  Kokkos::fence();
  Timer time;
  (void)cycle;

  // DataOut<dim> data_out;

  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(global_solution_host, "solution");
  // data_out.build_patches();

  // DataOutBase::VtkFlags flags;
  // flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  // data_out.set_flags(flags);
  // data_out.write_vtu_with_pvtu_record(
  //   "./", "solution", cycle, mpi_communicator, 2);

  Vector<float> cellwise_norm(level_triangulations.back()->n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    global_solution_host,
                                    Functions::ZeroFunction<dim>(),
                                    cellwise_norm,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);
  const double global_norm =
    VectorTools::compute_global_error(*level_triangulations.back(),
                                      cellwise_norm,
                                      VectorTools::L2_norm);

  pcout << "        solution norm: " << global_norm << std::endl;

  Kokkos::fence();
  time_details << "           Output results    (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;

  // Vector<float> cellwise_norm_subdomain(
  //   subdomain_triangulation.get_triangulation().n_active_cells());
  // VectorTools::integrate_difference(subdomain_dof_handler.get_dof_handler(),
  //                                   subdomain_solution_host,
  //                                   Functions::ZeroFunction<dim>(),
  //                                   cellwise_norm_subdomain,
  //                                   QGauss<dim>(fe.degree + 2),
  //                                   VectorTools::L2_norm);
  // const double subdomain_norm = VectorTools::compute_global_error(
  //   subdomain_triangulation.get_triangulation(),
  //   cellwise_norm_subdomain,
  //   VectorTools::L2_norm);


  // std::cout << " solution norm on subdomain "
  //           << subdomain_dof_handler.get_subdomain_id() << ": "
  //           << subdomain_norm << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::test_coarse_problem()
{
  // LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
  // test_src(
  //   this->subdomain_dof_handler.get_interface_vector_partitioner()),
  //   test_dst(this->subdomain_dof_handler.get_interface_vector_partitioner());

  // test_src = 1.0;
  // test_src.update_ghost_values();

  // bnn_preconditioner->vmult(test_dst, test_src);

  // bnn_preconditioner->project(test_dst, test_src);

  // test_dst.update_ghost_values();

  // test_dst.print(std::cout);

  // rhs_schur_device.print(std::cout);

  // ReductionControl solver_control(rhs_schur_device.size(),
  //                                 1e-9 * rhs_schur_device.l2_norm());

  // ReductionControl solver_control(rhs_schur_device.size(),1e-16, 1e-9);



  // Portable::SolverProjectedCG<
  //   LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  //   cg(solver_control);

  // solution_interface_device = 0.;
  // cg.solve(*interface_operator,
  //          solution_interface_device,
  //          rhs_schur_device,
  //          *bnn_preconditioner);


  // SubdomainHyperCubeTriangulation<dim> tria(this->mpi_communicator);
  // tria.generate_subdomain_triangulations(2);

  // // tria.refine_global(1);

  // tria.save_triangulations();


  // DoFHandler<dim> dh_fully_distributed;
  // dh_fully_distributed.reinit(tria.get_distributed_triangulation());
  // dh_fully_distributed.distribute_dofs(fe);

  // locally_owned_dofs = dh_fully_distributed.locally_owned_dofs();
  // locally_relevant_dofs =
  //   DoFTools::extract_locally_relevant_dofs(dh_fully_distributed);

  // SubdomainDoFHandler<dim> dh_subdomain;

  // dh_subdomain.reinit(tria.get_subdomain_triangulation(),
  // dh_fully_distributed); dh_subdomain.distribute_subdomain_dofs();

  // pcout << "           Total number of DoFs: " <<
  // dh_fully_distributed.n_dofs()
  //       << std::endl;


  // std::cout << "           DoFs on subdomain "
  //           << Utilities::MPI::this_mpi_process(mpi_communicator) << ": "
  //           << dof_handler.n_dofs() << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::test_triangulation(int n_refinements)
{
  const unsigned int n_subdomains =
    Utilities::MPI::n_mpi_processes(mpi_communicator);

  const unsigned int this_subdomain =
    Utilities::MPI::this_mpi_process(mpi_communicator);

  std::vector<unsigned int> subdomains_per_axis(dim);
  int                       remaining = n_subdomains;

  for (int d = dim; d > 0; --d)
    {
      int n_this_axis = std::pow(remaining, 1.0 / d);

      int best_divisor = 1;
      for (int j = n_this_axis; j >= 1; --j)
        if (remaining % j == 0)
          {
            best_divisor = j;
            break;
          }
      subdomains_per_axis[d - 1] = best_divisor;

      remaining /= best_divisor;
    }


  Triangulation<dim> coarse_triangulation;

  if (dim == 2)
    GridGenerator::subdivided_hyper_rectangle(coarse_triangulation,
                                              subdomains_per_axis,
                                              Point<dim>(0., 0.),
                                              Point<dim>(1., 1.));
  else
    GridGenerator::subdivided_hyper_rectangle(coarse_triangulation,
                                              subdomains_per_axis,
                                              Point<dim>(0., 0., 0.),
                                              Point<dim>(1., 1., 1.));



  unsigned int cell_counter = 0;
  for (auto cell : coarse_triangulation.active_cell_iterators())
    cell->set_subdomain_id(cell_counter++);

  coarse_triangulation.refine_global(n_refinements);


  // // Create building blocks:
  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      coarse_triangulation, mpi_communicator);

  // parallel::fullydistributed::Triangulation<dim>
  //   fully_distributed_triangulation(mpi_communicator);

  this->triangulation.clear();
  this->triangulation.create_triangulation(description);

  // const unsigned int my_rank =
  //   Utilities::MPI::this_mpi_process(mpi_communicator);

  // // Synchronize output so the lines don't get jumbled in the console
  // for (unsigned int r = 0; r < n_subdomains; ++r)
  //   {
  //     if (my_rank == r)
  //       {
  //         std::cout << "Rank " << r
  //                   << " owns the following active cell vertices:" <<
  //                   std::endl;

  //         for (const auto &cell :
  //              fully_distributed_triangulation.active_cell_iterators())
  //           {
  //             if (cell->is_locally_owned())
  //               {
  //                 std::cout << "  Cell " << cell->active_cell_index() << ":
  //                 "; for (unsigned int v = 0;
  //                      v < GeometryInfo<dim>::vertices_per_cell;
  //                      ++v)
  //                   {
  //                     std::cout << "(" << cell->vertex(v) << ") ";
  //                   }
  //                 std::cout << std::endl;
  //               }
  //           }
  //         std::cout << "------------------------------------------"
  //                   << std::endl;
  //       }
  //     // Barrier to ensure Rank 0 finishes printing before Rank 1 starts
  //     MPI_Barrier(mpi_communicator);
  //   }



  // std::cout << "On sundomain " << this_subdomain
  //           << ": n_levels = " << description.cell_infos.size() <<
  //           std::endl;

  // const int fine_level = description.cell_infos.size() - 1;

  // for (unsigned int level = 0; level < description.cell_infos.size();
  // ++level)
  //   {
  //     std::cout << "On sundomain " << this_subdomain << ", on level: " <<
  //     level
  //               << ": n_cells = " << description.cell_infos[level].size()
  //               << std::endl;
  //   }

  // std::cout << "On sundomain " << this_subdomain << " cell data: ";
  // for (unsigned int i = 0; i < description.cell_infos[fine_level].size();
  // ++i)
  //   {
  //     const auto &cell = description.cell_infos[fine_level][i];
  //     std::cout << "[" << cell.id << ", " << cell.subdomain_id << ", "
  //               << cell.level_subdomain_id << "]" << " | ";
  //   }
  // std::cout << std::endl << std::endl << std::endl;


  // fully_distributed_triangulation.refine_global(2);
}



template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  for (unsigned int cycle = 0; cycle < 8; ++cycle)
    {
      pcout << "Cycle " << cycle << std::endl;

      // test_triangulation(cycle + 1);

      // triangulation.refine_global(1);

      create_subdomain_triangulations(cycle + 2);

      pcout << "N_cells = "
            << level_triangulations.back()->n_global_active_cells()
            << std::endl;


      setup_dofs();

      compute_interface_weights();

      setup_matrix_free();

      setup_mg_transfers();

      setup_smoothers();

      setup_mg_preconditioners();

      setup_interface_system();

      setup_bnn_preconditioner();

      assemble_rhs();


      pcout << "           setup time: " << setup_time << "s" << std::endl;

      solve_interface();


      // test_coarse_problem();

      // postprocess_subdomain_solution();

      // output_results(cycle);

      // test_triangulation();
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      constexpr int dim       = 2;
      constexpr int fe_degree = 1;

      const unsigned int n_pre_smooth  = 5;
      const unsigned int n_post_smooth = 5;

      LaplaceProblem<dim, fe_degree> laplace_problem(n_pre_smooth,
                                                     n_post_smooth);
      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}