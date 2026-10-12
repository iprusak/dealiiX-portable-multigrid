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
#include "hybrid_switch_permutation.h"
#include "hybrid_triangulations.h"
#include "operators/cpu_laplace_operator.h"
#include "portable_multigrid_solver.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * CPU levels of the hybrid multigrid on comm_world (all ranks): Q1 on the
   * meshes refined 0..switch_refine times, with the operators, transfers and
   * smoothers of poisson_cube_cpu. The finest CPU level is the switch level
   * s-1, which the GPU side also holds; P moves values between the two.
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

    // GPU ranks: the GPU side's DoFHandler of the switch level. setup() builds
    // P (switch-level values GPU <-> CPU) from it and checks P.
    void
    set_gpu_switch_dofs(const DoFHandler<dim> *dofs)
    {
      gpu_switch_dofs = dofs;
    }

    // Collective on comm_world: build the CPU levels 0..switch_refine.
    void
    setup(const unsigned int switch_refine);

    // Collective on comm_world: one V-cycle (zero initial guess) on the CPU
    // levels. Input and output are the switch-level values in GPU numbering
    // in gpu_values(): right-hand side in, solution out.
    void
    coarse_solve();

    // Switch-level values in GPU numbering: the locally owned entries of the
    // GPU side's level vector, in host memory. Empty on CPU-only ranks.
    std::vector<vcycle_number> &
    gpu_values()
    {
      return gpu_switch_values;
    }

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
      SwitchLevelPermutation<dim>                     permutation; // P
      MGLevelObject<AffineConstraints<vcycle_number>> constraints;
      AffineConstraints<full_number>                  top_constraints; // switch level, double
      MGLevelObject<std::unique_ptr<LevelMatrixType>> matrices;
      std::unique_ptr<TopMatrixType>                  top_matrix; // switch level, double
      MGLevelObject<std::unique_ptr<TransferType>>    transfers;
      MGLevelObject<SmootherType>                     smoothers;
      VectorType                                      rhs; // unused, MultigridSolver needs one
      std::unique_ptr<MGSolverType>                   solver;

      // switch-level values in CPU numbering, for coarse_solve()
      std::vector<vcycle_number> cpu_values;
      VectorType                 cpu_rhs;
      VectorType                 cpu_solution;
    };

    void
    setup_dofs(Hierarchy &h) const;

    void
    setup_operators(Hierarchy &h) const;

    const MPILayout   &layout;
    const unsigned int n_subdivisions;
    const unsigned int n_smooth;
    const FE_Q<dim>    fe; // Q1 on all CPU levels

    const DoFHandler<dim> *gpu_switch_dofs = nullptr;

    std::vector<vcycle_number> gpu_switch_values;

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

    setup_dofs(h);

    // only if the GPU ranks provided their switch level: who owns which
    // switch-level cells on the two sides, then P and its check
    if (Utilities::MPI::max(gpu_switch_dofs != nullptr ? 1u : 0u, layout.comm_world) == 1u)
      {
        const DoFHandler<dim> &cpu_switch_dofs = h.dof_handlers[h.dof_handlers.max_level()];

        check_switch_level_ownership<dim>(layout,
                                          gpu_switch_dofs != nullptr ?
                                            &gpu_switch_dofs->get_triangulation() :
                                            nullptr,
                                          cpu_switch_dofs.get_triangulation(),
                                          std::cout);

        h.permutation.reinit(layout, gpu_switch_dofs, cpu_switch_dofs);

        // the same on all ranks (global max)
        const double p_error = check_switch_level_permutation(
          layout, h.permutation, gpu_switch_dofs, cpu_switch_dofs, std::cout);
        AssertThrow(p_error < 1e-12, ExcMessage("P moves switch-level values to wrong DoFs."));
      }

    setup_operators(h);

    // buffers for coarse_solve(), in the MatrixFree layout of the switch level
    h.top_matrix->initialize_dof_vector(h.rhs);
    h.top_matrix->initialize_dof_vector(h.cpu_rhs);
    h.top_matrix->initialize_dof_vector(h.cpu_solution);
    h.cpu_values.resize(h.cpu_rhs.locally_owned_size());
    gpu_switch_values.resize(
      gpu_switch_dofs != nullptr ? gpu_switch_dofs->locally_owned_dofs().n_elements() : 0);

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
  CpuSide<dim>::coarse_solve()
  {
    Assert(hierarchy, ExcMessage("coarse_solve() before setup()."));
    Hierarchy &h = *hierarchy;

    // GPU numbering -> CPU numbering (P), float -> double
    h.permutation.gpu_to_cpu(ArrayView<const vcycle_number>(gpu_switch_values),
                             ArrayView<vcycle_number>(h.cpu_values));
    for (unsigned int i = 0; i < h.cpu_values.size(); ++i)
      h.cpu_rhs.local_element(i) = h.cpu_values[i];

    // one V-cycle on the CPU levels, zero initial guess
    h.solver->vmult(h.cpu_solution, h.cpu_rhs);

    // double -> float, CPU numbering -> GPU numbering (P back)
    for (unsigned int i = 0; i < h.cpu_values.size(); ++i)
      h.cpu_values[i] = h.cpu_solution.local_element(i);
    h.permutation.cpu_to_gpu(ArrayView<vcycle_number>(h.cpu_values),
                             ArrayView<vcycle_number>(gpu_switch_values));
  }
} // namespace hybrid
