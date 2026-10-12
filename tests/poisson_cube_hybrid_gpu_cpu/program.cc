// Hybrid CPU-GPU multigrid (work in progress), step 1: rank roles and
// communicators.
//
//   mpirun -n <ranks> ./program ranks_per_gpu
//
// ranks_per_gpu = k: one GPU rank + k-1 CPU-only ranks per GPU. Without a
// GPU, HYBRID_SIMULATED_RANKS_PER_NODE=<n> mimics nodes of n ranks each.

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "hybrid_mpi_layout.h"


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
      if (argc < 2)
        {
          if (is_root)
            std::cout << "Usage: mpirun -n <ranks> ./program ranks_per_gpu" << std::endl;
          return 1;
        }

      const unsigned int ranks_per_gpu = std::atoi(argv[1]);

      // build the communicators, print the table, verify
      hybrid::MPILayout layout(MPI_COMM_WORLD, ranks_per_gpu);
      layout.print(std::cout);
      hybrid::check_layout(layout);

      // only a warning: on pc02 (one socket) it never fires; on JUPITER it
      // catches a wrong srun rank placement
      const bool share_socket = layout.blocks_share_socket();
      if (is_root)
        {
          if (!share_socket)
            std::cout << "WARNING: the ranks of some block are not pinned to one CPU socket."
                      << std::endl
                      << "         With srun use --distribution=block:block --cpu-bind=cores."
                      << std::endl;
          std::cout << "MPI layout checks passed." << std::endl;
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
