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

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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



  namespace internal
  {
    // The Kokkos allocations that are alive, filled by the callbacks below.
    struct LiveAllocations
    {
      struct Entry
      {
        std::string   space, label;
        std::uint64_t size;
      };

      static std::map<const void *, Entry> &
      entries()
      {
        static std::map<const void *, Entry> map;
        return map;
      }

      static std::mutex &
      mutex()
      {
        static std::mutex m;
        return m;
      }

      static void
      allocate(const Kokkos::Tools::SpaceHandle space,
               const char                      *label,
               const void                      *ptr,
               const std::uint64_t              size)
      {
        const std::lock_guard<std::mutex> lock(mutex());
        entries()[ptr] = Entry{space.name, label, size};
      }

      static void
      deallocate(const Kokkos::Tools::SpaceHandle,
                 const char *,
                 const void *ptr,
                 const std::uint64_t)
      {
        const std::lock_guard<std::mutex> lock(mutex());
        entries().erase(ptr);
      }
    };
  } // namespace internal



  // Call once after Kokkos is initialized: from then on every Kokkos
  // allocation and deallocation of this rank is recorded.
  inline void
  track_kokkos_allocations()
  {
    Kokkos::Tools::Experimental::set_allocate_data_callback(&internal::LiveAllocations::allocate);
    Kokkos::Tools::Experimental::set_deallocate_data_callback(
      &internal::LiveAllocations::deallocate);
  }



  // This rank's live Kokkos allocations: the total per memory space and the
  // n_groups largest groups of allocations with the same space, label and
  // size. Not collective; prints nothing without track_kokkos_allocations().
  inline void
  print_live_kokkos_allocations(const std::string &label,
                                std::ostream      &out,
                                const unsigned int n_groups = 12)
  {
    using Key = std::tuple<std::string, std::string, std::uint64_t>;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> spaces; // bytes, count
    std::map<Key, std::uint64_t>                                   groups; // count
    {
      const std::lock_guard<std::mutex> lock(internal::LiveAllocations::mutex());
      for (const auto &[ptr, entry] : internal::LiveAllocations::entries())
        {
          spaces[entry.space].first += entry.size;
          spaces[entry.space].second += 1;
          groups[Key(entry.space, entry.label, entry.size)] += 1;
        }
    }

    out << "Live Kokkos allocations of this rank, " << label << ":" << std::endl;
    for (const auto &[space, total] : spaces)
      out << "  " << space << ": " << total.first / 1048576. << " MiB in " << total.second
          << " allocations" << std::endl;

    std::vector<std::pair<std::uint64_t, Key>> sorted;
    for (const auto &[key, count] : groups)
      sorted.emplace_back(count * std::get<2>(key), key);
    std::sort(sorted.rbegin(), sorted.rend());
    for (unsigned int i = 0; i < std::min<std::size_t>(n_groups, sorted.size()); ++i)
      out << "    " << sorted[i].first / 1048576. << " MiB = " << groups[sorted[i].second] << " x "
          << std::get<2>(sorted[i].second) / 1048576. << " MiB  [" << std::get<0>(sorted[i].second)
          << "] " << std::get<1>(sorted[i].second) << std::endl;
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
