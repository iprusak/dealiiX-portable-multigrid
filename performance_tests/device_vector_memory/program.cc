// Does the memory of a device vector return to the GPU after the vector is
// destroyed? Only deal.II, Kokkos and MPI are used.
//
//   mpirun -n <ranks> ./program [n_owned] [n_ghost] [n_repetitions]
//
// Every rank owns n_owned entries (default 2^26 = 512 MiB in double) and has
// n_ghost ghost entries on each of its two neighbours in a ring of ranks.
// Each case creates one array or vector, uses it as described and destroys
// it, n_repetitions times; after every repetition the memory in use on the
// GPUs is printed (smallest and largest value over the ranks, all processes
// on a GPU counted) with its change since the line before:
//   1. Kokkos view, no MPI
//   2. deal.II vector, no ghost exchange
//   3. Kokkos view, MPI_Sendrecv of n_ghost entries from and into the view
//   4. deal.II vector, update_ghost_values()
//   5. deal.II vector, compress(VectorOperation::add)
// Memory that stays in use shows as a positive change that does not go away.

#include <deal.II/base/index_set.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#  include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace dealii;

using VectorType = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

// MiB in use on this rank's GPU, 0 without CUDA
double
device_memory_in_use()
{
#ifdef KOKKOS_ENABLE_CUDA
  Kokkos::fence();
  std::size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return (total_bytes - free_bytes) / 1048576.;
#else
  return 0.;
#endif
}

void
report(const std::string &label)
{
  static double last = -1.;

  const Utilities::MPI::MinMaxAvg memory =
    Utilities::MPI::min_max_avg(device_memory_in_use(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std::left << std::setw(46) << label << std::right << std::fixed
                << std::setprecision(0) << std::setw(8) << memory.min << " .. " << std::setw(8)
                << memory.max << " MiB";
      if (last >= 0.)
        std::cout << "   change " << std::showpos << std::setw(8) << memory.max - last
                  << std::noshowpos;
      std::cout << std::endl;
    }
  last = memory.max;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm     comm    = MPI_COMM_WORLD;
  const unsigned int rank    = Utilities::MPI::this_mpi_process(comm);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);

  const types::global_dof_index n_owned = argc > 1 ? std::atoll(argv[1]) : (1ull << 26);
  const types::global_dof_index n_ghost = argc > 2 ? std::atoll(argv[2]) : (1ull << 16);
  const unsigned int            n_rep   = argc > 3 ? std::atoi(argv[3]) : 3;

  const unsigned int next = (rank + 1) % n_ranks;
  const unsigned int prev = (rank + n_ranks - 1) % n_ranks;

  // owned: a contiguous block; ghosts: the first entries of the next rank
  // and the last entries of the previous one
  IndexSet owned(n_owned * n_ranks), ghost(n_owned * n_ranks);
  owned.add_range(rank * n_owned, (rank + 1) * n_owned);
  if (n_ranks > 1)
    {
      ghost.add_range(next * n_owned, next * n_owned + n_ghost);
      ghost.add_range((prev + 1) * n_owned - n_ghost, (prev + 1) * n_owned);
    }
  const auto partitioner = std::make_shared<const Utilities::MPI::Partitioner>(owned, ghost, comm);

  if (rank == 0)
    std::cout << "ranks: " << n_ranks << ", owned entries per rank: " << n_owned << " ("
              << n_owned * sizeof(double) / 1048576. << " MiB), ghost entries per neighbour: "
              << n_ghost << ", repetitions: " << n_rep << std::endl
#ifdef DEAL_II_MPI_WITH_DEVICE_SUPPORT
              << "deal.II: device-aware MPI" << std::endl
#else
              << "deal.II: MPI through host buffers" << std::endl
#endif
              << "GPU memory in use, smallest .. largest over the ranks:" << std::endl;

  report("start");

  for (unsigned int i = 0; i < n_rep; ++i)
    {
      {
        Kokkos::View<double *, MemorySpace::Default::kokkos_space> view("view", n_owned);
        Kokkos::deep_copy(view, 1.);
      }
      report("1. Kokkos view, no MPI");
    }

  for (unsigned int i = 0; i < n_rep; ++i)
    {
      {
        VectorType vector(partitioner);
        vector = 1.;
      }
      report("2. deal.II vector, no exchange");
    }

#if !defined(KOKKOS_ENABLE_CUDA) || defined(DEAL_II_MPI_WITH_DEVICE_SUPPORT)
  for (unsigned int i = 0; i < n_rep; ++i)
    {
      {
        Kokkos::View<double *, MemorySpace::Default::kokkos_space> view("view", n_owned + n_ghost);
        Kokkos::deep_copy(view, 1.);
        Kokkos::fence();
        MPI_Sendrecv(view.data(),
                     n_ghost,
                     MPI_DOUBLE,
                     next,
                     0,
                     view.data() + n_owned,
                     n_ghost,
                     MPI_DOUBLE,
                     prev,
                     0,
                     comm,
                     MPI_STATUS_IGNORE);
      }
      report("3. Kokkos view, MPI_Sendrecv from the view");
    }
#else
  if (rank == 0)
    std::cout << "3. skipped: MPI cannot take device pointers in this build" << std::endl;
#endif

  for (unsigned int i = 0; i < n_rep; ++i)
    {
      {
        VectorType vector(partitioner);
        vector = 1.;
        vector.update_ghost_values();
      }
      report("4. deal.II vector, update_ghost_values()");
    }

  for (unsigned int i = 0; i < n_rep; ++i)
    {
      {
        VectorType vector(partitioner);
        vector = 1.;
        vector.compress(VectorOperation::add);
      }
      report("5. deal.II vector, compress(add)");
    }

  // does a later allocation get the memory back?
  {
    Kokkos::View<double *, MemorySpace::Default::kokkos_space> view("view", n_owned);
    Kokkos::deep_copy(view, 1.);
    report("6. one more Kokkos view, alive");
  }
  report("   and destroyed");

  return 0;
}
