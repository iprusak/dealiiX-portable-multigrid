// Hybrid CPU-GPU multigrid (work in progress), step 2: rank roles,
// communicators, and the command loop of the CPU-only ranks (dummy work).
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
#include "hybrid_service.h"


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



  // Stand-in for the CPU levels. setup() and coarse_solve() only run
  // collectives over comm_world that check all ranks are at the same call.
  class DummyCpuSide
  {
  public:
    explicit DummyCpuSide(const MPILayout &layout)
      : layout(layout)
    {}

    void
    setup(const unsigned int switch_level)
    {
      ++n_setups;
      AssertThrow(Utilities::MPI::min(n_setups, layout.comm_world) ==
                    Utilities::MPI::max(n_setups, layout.comm_world),
                  ExcMessage("Ranks are at different setups."));
      AssertThrow(Utilities::MPI::min(switch_level, layout.comm_world) ==
                    Utilities::MPI::max(switch_level, layout.comm_world),
                  ExcMessage("Ranks got different switch levels."));
      this->switch_level = switch_level;
    }

    void
    coarse_solve()
    {
      ++n_solves;
      AssertThrow(Utilities::MPI::min(n_solves, layout.comm_world) ==
                    Utilities::MPI::max(n_solves, layout.comm_world),
                  ExcMessage("Ranks are at different coarse solves."));
    }

    unsigned int n_setups     = 0;
    unsigned int n_solves     = 0;
    unsigned int switch_level = 0;

  private:
    const MPILayout &layout;
  };



  // Stand-in for the driver on the GPU ranks: refinement cycles, each with a
  // CPU-side setup and a few V-cycles that reach the CPU levels.
  void
  run_dummy_driver(const MPILayout   &layout,
                   DummyCpuSide      &cpu_side,
                   const unsigned int n_cycles,
                   const unsigned int n_vcycles)
  {
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        const unsigned int switch_level = 2 + cycle; // placeholder

        send_command(layout, Command::setup, switch_level);
        cpu_side.setup(switch_level);

        for (unsigned int v = 0; v < n_vcycles; ++v)
          {
            // GPU levels: collectives on comm_gpu only, CPU-only ranks wait
            const unsigned int n_gpu_ranks = Utilities::MPI::sum(1u, layout.comm_gpu);
            AssertThrow(n_gpu_ranks == layout.n_graces, ExcInternalError());

            send_command(layout, Command::solve);
            cpu_side.coarse_solve();
          }
      }

    send_command(layout, Command::stop);
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
          std::cout << "MPI layout checks passed." << std::endl;
        }

      // GPU ranks drive, CPU-only ranks follow their commands
      const unsigned int n_cycles  = 3;
      const unsigned int n_vcycles = 5;

      hybrid::DummyCpuSide cpu_side(layout);
      if (layout.is_gpu_rank)
        hybrid::run_dummy_driver(layout, cpu_side, n_cycles, n_vcycles);
      else
        hybrid::run_service_loop(layout, cpu_side);

      // every rank took part in every setup and coarse solve
      AssertThrow(Utilities::MPI::min(cpu_side.n_setups, MPI_COMM_WORLD) == n_cycles &&
                    Utilities::MPI::max(cpu_side.n_setups, MPI_COMM_WORLD) == n_cycles,
                  ExcMessage("Wrong number of setups."));
      AssertThrow(Utilities::MPI::min(cpu_side.n_solves, MPI_COMM_WORLD) == n_cycles * n_vcycles &&
                    Utilities::MPI::max(cpu_side.n_solves, MPI_COMM_WORLD) ==
                      n_cycles * n_vcycles,
                  ExcMessage("Wrong number of coarse solves."));

      if (is_root)
        std::cout << "Service loop passed: " << n_cycles << " setups and "
                  << n_cycles * n_vcycles << " coarse solves on all "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << " ranks." << std::endl;
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
