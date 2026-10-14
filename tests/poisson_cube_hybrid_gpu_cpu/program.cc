// Hybrid CPU-GPU multigrid: the ph-multigrid of poisson_cube_bk3 with the
// fine levels switch_level.. on the GPU ranks (comm_gpu) and the coarse
// levels 0..switch_level on all ranks (comm_world). The GPU V-cycle's coarse
// solve on the switch level is one V-cycle on the CPU levels; the CPU-only
// ranks join it through the command loop.
//
//   mpirun -n <ranks> ./program ranks_per_gpu degree [n_refine] [switch_level]
//                               [n_subdivisions] [n_smooth] [stretch]
//
// ranks_per_gpu = k: one GPU rank + k-1 CPU-only ranks per GPU.
// degree:        polynomial degree of the fine level, 1..4
// n_refine:      refinements of the coarse grid for the fine mesh (default 2)
// switch_level:  finest CPU level, index in the ph-hierarchy: 0..n_refine are
//                the Q1 h-levels, then the p-levels of the fine mesh; at most
//                one below the fine level (default n_refine - 1)
// n_subdivisions: coarse grid n x n x n cells (default 3); the fine mesh is
//                poisson_cube_bk3's non-doubling mesh of size n * 2^n_refine
// n_smooth:      Chebyshev degree of the pre- and post-smoother (default 3)
// stretch:       -1 (default): the n_subdivisions cube above. 0, 1, 2: the
//                doubling mesh of poisson_cube_*, cycle = 3 * n_refine + stretch
//                (n_subdivisions is ignored)
// Without a GPU, HYBRID_SIMULATED_RANKS_PER_NODE=<n> mimics nodes of n ranks.

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "hybrid_coarse_solver.h"
#include "hybrid_cpu_side.h"
#include "hybrid_gpu_side.h"
#include "hybrid_mpi_layout.h"
#include "hybrid_service.h"
#include "hybrid_triangulations.h"


namespace hybrid
{
  // Verifies the communicators built by MPILayout. Collective on comm_world.
  void
  check_layout(const MPILayout &layout)
  {
    const unsigned int world_rank = Utilities::MPI::this_mpi_process(layout.comm_world);

    // every block has k ranks
    AssertThrow(Utilities::MPI::n_mpi_processes(layout.comm_grace) == layout.ranks_per_gpu,
                ExcMessage("Block does not have ranks_per_gpu ranks."));

    // count the GPU ranks in my block: sum of 1 (GPU rank) and 0 (CPU-only)
    AssertThrow(Utilities::MPI::sum(layout.is_gpu_rank ? 1u : 0u, layout.comm_grace) == 1u,
                ExcMessage("Block does not have exactly one GPU rank."));

    // comm_gpu has one rank per block; CPU-only ranks are not in it
    if (layout.is_gpu_rank)
      AssertThrow(Utilities::MPI::n_mpi_processes(layout.comm_gpu) == layout.n_graces,
                  ExcMessage("comm_gpu does not contain one rank per block."));
    else
      AssertThrow(layout.comm_gpu == MPI_COMM_NULL,
                  ExcMessage("CPU-only rank is a member of comm_gpu."));

    // the GPU rank (block rank 0) sends its world rank to the block: it has to
    // be the lowest one
    const unsigned int gpu_world_rank = Utilities::MPI::broadcast(layout.comm_grace, world_rank, 0);
    AssertThrow(gpu_world_rank <= world_rank, ExcMessage("GPU rank is not first in its block."));

    // every block index 0..n_graces-1 is used by exactly k ranks: each rank
    // collects all ranks' grace_index and counts
    const std::vector<unsigned int> indices =
      Utilities::MPI::all_gather(layout.comm_world, layout.grace_index);
    for (unsigned int g = 0; g < layout.n_graces; ++g)
      AssertThrow(std::count(indices.begin(), indices.end(), g) == layout.ranks_per_gpu,
                  ExcMessage("Block " + std::to_string(g) + " has the wrong number of ranks."));
  }



  // Driver on the GPU ranks: set up the GPU levels switch_level.. and the
  // CPU levels 0..switch_level (with P from the GPU side's switch level),
  // then solve with CG on the fine level.
  template <int dim, int fe_degree, typename CpuSideType>
  void
  run_driver(const MPILayout          &layout,
             GpuSide<dim, fe_degree> &gpu_side,
             CpuSideType             &cpu_side,
             const unsigned int       n_refine,
             const unsigned int       switch_level)
  {
    // the GPU V-cycle stops at the switch level and hands it to the CPU levels
    HybridCoarseSolver<CpuSideType> coarse_solver(layout, cpu_side);
    gpu_side.setup(n_refine, switch_level, &coarse_solver);

    // P is built from the GPU side's switch level
    cpu_side.set_gpu_switch_dofs(&gpu_side.level_dofs(switch_level));
    send_command(layout, Command::setup, switch_level);
    cpu_side.setup(switch_level);

    // every V-cycle of the CG solve runs one coarse solve on all ranks;
    // timers of all parts are reset per solve, printed for the last one
    for (unsigned int i = 0; i < 10; ++i)
      {
        coarse_solver.reset_times();
        cpu_side.reset_times();
        gpu_side.solve();
      }
    gpu_side.print_level_times();
    cpu_side.print_times();
    coarse_solver.print_times(std::cout);

    send_command(layout, Command::stop);
  }



  // cpu_degree: degree of the switch level, the top CPU level
  template <int dim, int fe_degree, int cpu_degree>
  void
  run_hybrid(const MPILayout   &layout,
             const unsigned int n_refine,
             const unsigned int switch_level,
             const CoarseGrid  &coarse_grid,
             const unsigned int n_smooth)
  {
    // GPU side only on the GPU ranks; declared first so that it outlives
    // cpu_side, which points to its switch-level DoFs
    std::unique_ptr<GpuSide<dim, fe_degree>> gpu_side;
    if (layout.is_gpu_rank)
      gpu_side = std::make_unique<GpuSide<dim, fe_degree>>(layout, coarse_grid, n_smooth);

    // CPU side on all ranks. GPU ranks drive, CPU-only ranks follow their
    // commands.
    CpuSide<dim, cpu_degree> cpu_side(layout, coarse_grid, n_smooth, n_refine, fe_degree);
    if (layout.is_gpu_rank)
      run_driver(layout, *gpu_side, cpu_side, n_refine, switch_level);
    else
      run_service_loop(layout, cpu_side);
  }



  // dispatches the degree of the switch level (known at run time)
  template <int dim, int fe_degree>
  void
  run(const MPILayout   &layout,
      const unsigned int n_refine,
      const unsigned int switch_level,
      const CoarseGrid  &coarse_grid,
      const unsigned int n_smooth)
  {
    const std::vector<unsigned int> degrees = ph_level_degrees(n_refine + 1, fe_degree);
    AssertThrow(switch_level + 1 < degrees.size(),
                ExcMessage("The switch level has to be below the fine level."));

    // below Q4 the p-levels are 4 -> 2 -> 1, 3 -> 1, 2 -> 1
    if (degrees[switch_level] == 1)
      run_hybrid<dim, fe_degree, 1>(layout, n_refine, switch_level, coarse_grid, n_smooth);
    else if constexpr (fe_degree > 2)
      {
        AssertThrow(degrees[switch_level] == 2, ExcNotImplemented());
        run_hybrid<dim, fe_degree, 2>(layout, n_refine, switch_level, coarse_grid, n_smooth);
      }
  }
} // namespace hybrid



int
main(int argc, char **argv)
{
  using namespace dealii;

  // starts MPI and Kokkos, 1 thread per rank; shuts both down at the end of
  // main, after `layout` below has freed its communicators
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  try
    {
      const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

      // all ranks return 1; only rank 0 prints
      if (argc < 3)
        {
          if (is_root)
            std::cout << "Usage: mpirun -n <ranks> ./program ranks_per_gpu degree [n_refine] "
                         "[switch_level] [n_subdivisions] [n_smooth] [stretch]"
                      << std::endl;
          return 1;
        }

      const unsigned int ranks_per_gpu = std::atoi(argv[1]);
      const unsigned int degree        = std::atoi(argv[2]);
      const unsigned int n_refine      = argc > 3 ? std::atoi(argv[3]) : 2;
      const unsigned int switch_level   = argc > 4 ? std::atoi(argv[4]) : n_refine - 1;
      const unsigned int n_subdivisions = argc > 5 ? std::atoi(argv[5]) : 3;
      const unsigned int n_smooth       = argc > 6 ? std::atoi(argv[6]) : 3;
      const int          stretch        = argc > 7 ? std::atoi(argv[7]) : -1;
      const hybrid::CoarseGrid coarse_grid{n_subdivisions, stretch};
      AssertThrow(degree >= 1 && degree <= 4, ExcMessage("degree has to be 1..4."));

      // build the communicators, print the table, verify
      hybrid::MPILayout layout(MPI_COMM_WORLD, ranks_per_gpu);
      layout.print(std::cout);
      hybrid::check_layout(layout);

      // only warnings: on pc02 they never fire; on JUPITER they catch a wrong
      // srun rank placement or --ntasks-per-node that does not match k
      const bool share_socket = layout.blocks_share_socket();
      const bool match_gpus   = layout.blocks_match_slurm_gpus();
      if (is_root)
        {
          if (!share_socket)
            std::cout << "WARNING: the ranks of some block are not pinned to one CPU socket."
                      << std::endl
                      << "         With srun use --distribution=block:block --cpu-bind=cores."
                      << std::endl;
          if (!match_gpus)
            std::cout << "WARNING: ranks per node / ranks_per_gpu differs from SLURM_GPUS_ON_NODE."
                      << std::endl;
          std::cout << "MPI layout checks passed." << std::endl << std::endl;
        }

      // degree known at run time, templates need it at compile time
      constexpr int dim = 3;
      switch (degree)
        {
          case 1:
            hybrid::run<dim, 1>(layout, n_refine, switch_level, coarse_grid, n_smooth);
            break;
          case 2:
            hybrid::run<dim, 2>(layout, n_refine, switch_level, coarse_grid, n_smooth);
            break;
          case 3:
            hybrid::run<dim, 3>(layout, n_refine, switch_level, coarse_grid, n_smooth);
            break;
          case 4:
            hybrid::run<dim, 4>(layout, n_refine, switch_level, coarse_grid, n_smooth);
            break;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
