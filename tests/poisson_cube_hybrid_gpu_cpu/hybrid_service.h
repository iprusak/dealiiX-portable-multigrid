#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <utility>

#include "hybrid_mpi_layout.h"

namespace hybrid
{
  using namespace dealii;

  // What the GPU rank of a block tells its CPU-only ranks to do next.
  enum class Command : int
  {
    setup = 0, // build the CPU levels; payload = switch level s
    solve = 1, // one coarse solve on the CPU levels
    stop  = 2  // leave the service loop
  };



  // GPU rank: send the next command to its block. Matches the MPI_Bcast in
  // receive_command() on the CPU-only ranks of the block.
  inline void
  send_command(const MPILayout &layout, const Command command, const unsigned int payload = 0)
  {
    Assert(layout.is_gpu_rank, ExcMessage("Only GPU ranks send commands."));

    int message[2] = {static_cast<int>(command), static_cast<int>(payload)};

    const int ierr = MPI_Bcast(message, 2, MPI_INT, 0, layout.comm_grace);
    AssertThrowMPI(ierr);
  }



  // CPU-only rank: wait until the GPU rank of the block sends a command.
  inline std::pair<Command, unsigned int>
  receive_command(const MPILayout &layout)
  {
    Assert(!layout.is_gpu_rank, ExcMessage("GPU ranks send commands, they do not receive them."));

    int message[2] = {0, 0};

    const int ierr = MPI_Bcast(message, 2, MPI_INT, 0, layout.comm_grace);
    AssertThrowMPI(ierr);

    return {static_cast<Command>(message[0]), static_cast<unsigned int>(message[1])};
  }



  // CPU-only ranks: execute commands until stop. The GPU rank calls the same
  // setup()/coarse_solve() of its own cpu_side right after sending the
  // command, so all ranks of comm_world enter them together.
  template <typename CpuSide>
  void
  run_service_loop(const MPILayout &layout, CpuSide &cpu_side)
  {
    while (true)
      {
        const auto [command, payload] = receive_command(layout);

        if (command == Command::setup)
          cpu_side.setup(payload);
        else if (command == Command::solve)
          cpu_side.coarse_solve();
        else
          break;
      }
  }
} // namespace hybrid
