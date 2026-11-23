#ifndef device_memory_h
#define device_memory_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/vector_memory.templates.h>

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#  include <cuda_runtime.h>
#endif

#include <ostream>
#include <string>

namespace multigrid
{
  // Memory in use on this rank's GPU in MiB, all processes on it counted;
  // 0 without CUDA.
  inline double
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



  // Collective on comm: smallest and largest value over its ranks.
  inline void
  print_device_memory(const std::string &label, const MPI_Comm comm, std::ostream &out)
  {
    const dealii::Utilities::MPI::MinMaxAvg memory =
      dealii::Utilities::MPI::min_max_avg(device_memory_in_use(), comm);
    if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
      out << "GPU memory in use [MiB], " << label << ": " << memory.min << " .. " << memory.max
          << std::endl;
  }



  // Frees the device vectors that deal.II's solvers have returned to their
  // pool (GrowingVectorMemory keeps them for reuse).
  inline void
  release_device_vector_pools()
  {
    using namespace dealii;
    GrowingVectorMemory<LinearAlgebra::distributed::Vector<float, MemorySpace::Default>>::
      release_unused_memory();
    GrowingVectorMemory<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>::
      release_unused_memory();
  }
} // namespace multigrid

#endif
