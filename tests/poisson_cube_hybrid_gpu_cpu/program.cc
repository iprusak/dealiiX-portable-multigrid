// Hybrid CPU-GPU multigrid: the ph-multigrid of poisson_cube_bk3 with the
// fine levels switch_refine.. on the GPU ranks (comm_gpu) and the coarse
// levels 0..switch_refine on all ranks (comm_world). The GPU V-cycle's coarse
// solve on the switch level is one V-cycle on the CPU levels; the CPU-only
// ranks join it through the command loop.
//
//   mpirun -n <ranks> ./program ranks_per_gpu degree [n_refine] [switch_refine]
//
// ranks_per_gpu = k: one GPU rank + k-1 CPU-only ranks per GPU.
// degree:        polynomial degree of the fine level, 1..4
// n_refine:      refinements of the 3x3x3 coarse grid for the fine mesh
//                (default 2); same mesh as poisson_cube_bk3 with the
//                non-doubling mesh of size 3 * 2^n_refine
// switch_refine: refinements of the switch level = finest CPU level
//                (default n_refine - 1)
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



  // Driver on the GPU ranks: set up the GPU levels switch_refine.. and the
  // CPU levels 0..switch_refine (with P from the GPU side's switch level),
  // then solve with CG on the fine level.
  template <int dim, int fe_degree>
  void
  run_driver(const MPILayout          &layout,
             GpuSide<dim, fe_degree> &gpu_side,
             CpuSide<dim>            &cpu_side,
             const unsigned int       n_refine,
             const unsigned int       switch_refine)
  {
    // the GPU V-cycle stops at the switch level and hands it to the CPU levels
    HybridCoarseSolver<dim> coarse_solver(layout, cpu_side);
    gpu_side.setup(n_refine, switch_refine, &coarse_solver);

    // GPU level switch_refine is Q1 on the switch-level mesh: P is built from it
    cpu_side.set_gpu_switch_dofs(&gpu_side.h_level_dofs(switch_refine));
    send_command(layout, Command::setup, switch_refine);
    cpu_side.setup(switch_refine);

    // every V-cycle of the CG solve runs one coarse solve on all ranks
    for (unsigned int i = 0; i < 2; ++i)
      {
        coarse_solver.reset_times();
        gpu_side.solve();
        coarse_solver.print_times(std::cout);
      }

    send_command(layout, Command::stop);
  }



  template <int dim, int fe_degree>
  void
  run(const MPILayout   &layout,
      const unsigned int n_refine,
      const unsigned int switch_refine)
  {
    const unsigned int n_subdivisions = 3; // coarse grid: 3x3x3 cells
    const unsigned int n_smooth       = 3; // Chebyshev degree, as poisson_cube_*

    // GPU side only on the GPU ranks; declared first so that it outlives
    // cpu_side, which points to its switch-level DoFs
    std::unique_ptr<GpuSide<dim, fe_degree>> gpu_side;
    if (layout.is_gpu_rank)
      gpu_side = std::make_unique<GpuSide<dim, fe_degree>>(layout, n_subdivisions, n_smooth);

    // CPU side on all ranks. GPU ranks drive, CPU-only ranks follow their
    // commands.
    CpuSide<dim> cpu_side(layout, n_subdivisions, n_smooth);
    if (layout.is_gpu_rank)
      run_driver(layout, *gpu_side, cpu_side, n_refine, switch_refine);
    else
      run_service_loop(layout, cpu_side);
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
                         "[switch_refine]"
                      << std::endl;
          return 1;
        }

      const unsigned int ranks_per_gpu = std::atoi(argv[1]);
      const unsigned int degree        = std::atoi(argv[2]);
      const unsigned int n_refine      = argc > 3 ? std::atoi(argv[3]) : 2;
      const unsigned int switch_refine = argc > 4 ? std::atoi(argv[4]) : n_refine - 1;
      AssertThrow(degree >= 1 && degree <= 4, ExcMessage("degree has to be 1..4."));
      AssertThrow(switch_refine <= n_refine,
                  ExcMessage("switch_refine has to be at most n_refine."));

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
            hybrid::run<dim, 1>(layout, n_refine, switch_refine);
            break;
          case 2:
            hybrid::run<dim, 2>(layout, n_refine, switch_refine);
            break;
          case 3:
            hybrid::run<dim, 3>(layout, n_refine, switch_refine);
            break;
          case 4:
            hybrid::run<dim, 4>(layout, n_refine, switch_refine);
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
