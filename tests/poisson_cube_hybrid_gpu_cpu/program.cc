// Hybrid CPU-GPU multigrid (work in progress): rank roles and communicators,
// mesh levels of the GPU side (comm_gpu), and the CPU levels on comm_world,
// driven by the GPU ranks through the command loop. For now each coarse
// solve is a CG solve on the switch level (f = 1) with the CPU V-cycle.
//
//   mpirun -n <ranks> ./program ranks_per_gpu [n_refine] [switch_refine]
//
// ranks_per_gpu = k: one GPU rank + k-1 CPU-only ranks per GPU.
// n_refine:      refinements of the fine mesh (default 3)
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

#include "hybrid_cpu_side.h"
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



  // Driver on the GPU ranks, for now without GPU levels: one setup of the CPU
  // side, then a few coarse solves.
  template <int dim>
  void
  run_driver(const MPILayout   &layout,
             CpuSide<dim>      &cpu_side,
             const unsigned int switch_refine,
             const unsigned int n_solves)
  {
    send_command(layout, Command::setup, switch_refine);
    cpu_side.setup(switch_refine);

    for (unsigned int i = 0; i < n_solves; ++i)
      {
        send_command(layout, Command::solve);
        cpu_side.coarse_solve();
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
            std::cout << "Usage: mpirun -n <ranks> ./program ranks_per_gpu [n_refine] "
                         "[switch_refine]"
                      << std::endl;
          return 1;
        }

      const unsigned int ranks_per_gpu = std::atoi(argv[1]);
      const unsigned int n_refine      = argc > 2 ? std::atoi(argv[2]) : 3;
      const unsigned int switch_refine = argc > 3 ? std::atoi(argv[3]) : n_refine - 1;
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

      constexpr int      dim            = 3;
      const unsigned int n_subdivisions = 3; // coarse grid: 3x3x3 cells
      const unsigned int n_smooth       = 3; // Chebyshev degree, as poisson_cube_*
      const unsigned int n_solves       = 2;

      // GPU side, for now only its mesh levels and the switch-level DoFs (Q1)
      // on the GPU ranks
      std::unique_ptr<hybrid::MeshLevels<dim>> gpu_mesh;
      std::unique_ptr<DoFHandler<dim>>         gpu_switch_dofs;
      if (layout.is_gpu_rank)
        {
          gpu_mesh =
            std::make_unique<hybrid::MeshLevels<dim>>(layout.comm_gpu, n_subdivisions, n_refine);
          hybrid::print_level_partitions(gpu_mesh->levels,
                                         layout.comm_gpu,
                                         "GPU side, comm_gpu",
                                         std::cout);

          gpu_switch_dofs = std::make_unique<DoFHandler<dim>>(*gpu_mesh->levels[switch_refine]);
          gpu_switch_dofs->distribute_dofs(FE_Q<dim>(1));
        }

      // CPU side: built on all ranks by the SETUP command. GPU ranks drive,
      // CPU-only ranks follow their commands.
      hybrid::CpuSide<dim> cpu_side(layout, n_subdivisions, n_smooth);
      if (layout.is_gpu_rank)
        {
          cpu_side.set_gpu_switch_dofs(gpu_switch_dofs.get());
          hybrid::run_driver(layout, cpu_side, switch_refine, n_solves);
        }
      else
        hybrid::run_service_loop(layout, cpu_side);
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
