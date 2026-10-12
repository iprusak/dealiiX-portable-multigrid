#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#include "hybrid_mpi_layout.h"
#include "hybrid_triangulations.h"
#include "operators/cpu_laplace_operator.h"
#include "portable_multigrid_solver.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * CPU levels of the hybrid multigrid on comm_world (all ranks): Q1 on the
   * meshes refined 0..switch_refine times, with the operators, transfers and
   * smoothers of poisson_cube_cpu.
   *
   * GPU ranks call setup()/coarse_solve() directly, CPU-only ranks through
   * run_service_loop().
   */
  template <int dim>
  class CpuSide
  {
  public:
    using vcycle_number = float; // as in poisson_cube_bk3
    using full_number   = double;

    using VectorType   = LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host>;
    using VectorTypeMG = LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::Host>;

    using LevelMatrixType = Portable::LaplaceOperatorBase<dim, VectorTypeMG>;
    using TopMatrixType   = Portable::LaplaceOperatorBase<dim, VectorType>;
    using TransferType    = MGTwoLevelTransfer<dim, VectorTypeMG>;
    using SmootherType    = PreconditionChebyshev<LevelMatrixType, VectorTypeMG>;

    // the top CPU level is Q1, hence fe_degree = 1
    using MGSolverType = multigrid::MultigridSolver<dim,
                                                    1,
                                                    vcycle_number,
                                                    full_number,
                                                    SmootherType,
                                                    VectorTypeMG,
                                                    VectorType,
                                                    TransferType>;

    CpuSide(const MPILayout &layout, const unsigned int n_subdivisions, const unsigned int n_smooth)
      : layout(layout)
      , n_subdivisions(n_subdivisions)
      , n_smooth(n_smooth)
      , fe(1)
    {}

    // GPU ranks: the GPU side's copy of the switch level, used by setup() to
    // check who owns which switch-level cells on the two sides.
    void
    set_gpu_switch_level(const Triangulation<dim> *tria)
    {
      gpu_switch_level = tria;
    }

    // Collective on comm_world: build the CPU levels 0..switch_refine.
    void
    setup(const unsigned int switch_refine);

    // Collective on comm_world. For now a test: CG on the switch level with
    // right-hand side f = 1, preconditioned by the CPU V-cycle.
    void
    coarse_solve();

    std::pair<unsigned int, double> last_cg = {0, 0.}; // iterations, reduction per iteration

  private:
    // Everything one setup() builds. Members are destroyed in reverse order:
    // solver first, mesh last.
    struct Hierarchy
    {
      Hierarchy(const MPI_Comm     comm,
                const unsigned int n_subdivisions,
                const unsigned int switch_refine)
        : mesh(comm, n_subdivisions, switch_refine)
      {}

      MeshLevels<dim>                                 mesh;
      MGLevelObject<DoFHandler<dim>>                  dof_handlers;
      MGLevelObject<AffineConstraints<vcycle_number>> constraints;
      AffineConstraints<full_number>                  top_constraints; // switch level, double
      MGLevelObject<std::unique_ptr<LevelMatrixType>> matrices;
      std::unique_ptr<TopMatrixType>                  top_matrix; // switch level, double
      MGLevelObject<std::unique_ptr<TransferType>>    transfers;
      MGLevelObject<SmootherType>                     smoothers;
      VectorType                                      rhs;
      std::unique_ptr<MGSolverType>                   solver;
    };

    void
    setup_dofs(Hierarchy &h) const;

    void
    setup_operators(Hierarchy &h) const;

    void
    compute_rhs(Hierarchy &h) const;

    const MPILayout   &layout;
    const unsigned int n_subdivisions;
    const unsigned int n_smooth;
    const FE_Q<dim>    fe; // Q1 on all CPU levels

    const Triangulation<dim> *gpu_switch_level = nullptr;

    std::unique_ptr<Hierarchy> hierarchy;
  };



  template <int dim>
  void
  CpuSide<dim>::setup(const unsigned int switch_refine)
  {
    Timer time;

    // tear down the previous setup (solver first, mesh last), then rebuild
    hierarchy.reset();
    hierarchy = std::make_unique<Hierarchy>(layout.comm_world, n_subdivisions, switch_refine);
    Hierarchy &h = *hierarchy;

    print_level_partitions(h.mesh.levels, layout.comm_world, "CPU side, comm_world", std::cout);

    // only if the GPU ranks provided their switch level
    if (Utilities::MPI::max(gpu_switch_level != nullptr ? 1u : 0u, layout.comm_world) == 1u)
      check_switch_level_ownership<dim>(layout,
                                        gpu_switch_level,
                                        *h.mesh.levels.back(),
                                        std::cout);

    setup_dofs(h);
    setup_operators(h);
    compute_rhs(h);

    h.solver = std::make_unique<MGSolverType>(h.top_matrix,
                                              h.dof_handlers,
                                              h.constraints,
                                              h.transfers,
                                              h.matrices,
                                              h.smoothers,
                                              h.rhs,
                                              n_smooth,
                                              n_smooth);

    const double setup_time = Utilities::MPI::max(time.wall_time(), layout.comm_world);
    if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
      {
        std::cout << "CPU levels 0.." << h.dof_handlers.max_level() << ", DoFs:";
        for (unsigned int l = 0; l <= h.dof_handlers.max_level(); ++l)
          std::cout << ' ' << h.dof_handlers[l].n_dofs();
        std::cout << std::endl << "CPU side setup: " << setup_time << " s" << std::endl;
      }
  }



  template <int dim>
  void
  CpuSide<dim>::setup_dofs(Hierarchy &h) const
  {
    const unsigned int max_level = h.mesh.levels.size() - 1;

    h.dof_handlers.resize(0, max_level);
    h.constraints.resize(0, max_level);

    for (unsigned int l = 0; l <= max_level; ++l)
      {
        DoFHandler<dim> &dof_handler = h.dof_handlers[l];
        dof_handler.reinit(*h.mesh.levels[l]);
        dof_handler.distribute_dofs(fe);

        const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        // built in double, as in poisson_cube_cpu; kept for the top level
        AffineConstraints<full_number>  level_constraints_double;
        AffineConstraints<full_number> &constraints =
          l == max_level ? h.top_constraints : level_constraints_double;

        constraints.reinit(dof_handler.locally_owned_dofs(), relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 types::boundary_id(0),
                                                 Functions::ZeroFunction<dim, full_number>(),
                                                 constraints);
        constraints.close();

        h.constraints[l].reinit(dof_handler.locally_owned_dofs(), relevant_dofs);
        h.constraints[l].copy_from(constraints);
        h.constraints[l].close();
      }
  }



  template <int dim>
  void
  CpuSide<dim>::setup_operators(Hierarchy &h) const
  {
    const unsigned int max_level = h.dof_handlers.max_level();

    // level operators in vcycle_number, the top level also in double for CG
    h.matrices.resize(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      h.matrices[l] = std::make_unique<CPU::LaplaceOperator<dim, 1, vcycle_number>>(
        h.dof_handlers[l], h.constraints[l], false);

    h.top_matrix = std::make_unique<CPU::LaplaceOperator<dim, 1, full_number>>(
      h.dof_handlers[max_level], h.top_constraints, false);

    // geometric transfers between consecutive CPU levels
    h.transfers.resize(0, max_level);
    for (unsigned int l = 1; l <= max_level; ++l)
      {
        h.transfers[l] = std::make_unique<TransferType>();
        h.transfers[l]->reinit_geometric_transfer(h.dof_handlers[l],
                                                  h.dof_handlers[l - 1],
                                                  h.constraints[l],
                                                  h.constraints[l - 1]);
        h.transfers[l]->enable_inplace_operations_if_possible(
          h.matrices[l - 1]->get_vector_partitioner(), h.matrices[l]->get_vector_partitioner());
      }

    // Chebyshev smoothers; on level 0 Chebyshev serves as coarse solver
    h.smoothers.resize(0, max_level);
    for (unsigned int l = 0; l <= max_level; ++l)
      {
        typename SmootherType::AdditionalData smoother_data;
        if (l > 0)
          {
            smoother_data.smoothing_range     = 15.;
            smoother_data.degree              = n_smooth;
            smoother_data.eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data.smoothing_range     = 1e-3;
            smoother_data.degree              = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = h.matrices[0]->m();
          }

        h.matrices[l]->compute_diagonal();
        smoother_data.preconditioner = h.matrices[l]->get_matrix_diagonal_inverse();

        h.smoothers[l].initialize(*h.matrices[l], smoother_data);
      }
  }



  // f = 1 on the switch level, as poisson_cube_cpu's compute_rhs().
  template <int dim>
  void
  CpuSide<dim>::compute_rhs(Hierarchy &h) const
  {
    const DoFHandler<dim> &dof_handler = h.dof_handlers[h.dof_handlers.max_level()];

    VectorType rhs_ghosted(dof_handler.locally_owned_dofs(),
                           DoFTools::extract_locally_relevant_dofs(dof_handler),
                           layout.comm_world);

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim>     fe_values(fe, quadrature, update_values | update_JxW_values);

    Vector<full_number>                  cell_rhs(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_rhs = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              cell_rhs(i) += fe_values.shape_value(i, q) * fe_values.JxW(q);

          cell->get_dof_indices(dof_indices);
          h.top_constraints.distribute_local_to_global(cell_rhs, dof_indices, rhs_ghosted);
        }
    rhs_ghosted.compress(VectorOperation::add);

    h.top_matrix->initialize_dof_vector(h.rhs);
    h.rhs.copy_locally_owned_data_from(rhs_ghosted);
  }



  template <int dim>
  void
  CpuSide<dim>::coarse_solve()
  {
    Assert(hierarchy, ExcMessage("coarse_solve() before setup()."));

    Timer time;
    last_cg = hierarchy->solver->solve_cg();

    const double solve_time = Utilities::MPI::max(time.wall_time(), layout.comm_world);
    if (Utilities::MPI::this_mpi_process(layout.comm_world) == 0)
      std::cout << "CG on the switch level: " << last_cg.first << " iterations, reduction "
                << std::setprecision(4) << last_cg.second << " per iteration, " << solve_time
                << " s" << std::endl;
  }
} // namespace hybrid
