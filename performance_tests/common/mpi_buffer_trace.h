#ifndef mpi_buffer_trace_h
#define mpi_buffer_trace_h

// Diagnostic: which MPI_Isend / MPI_Irecv calls get a buffer inside a large
// Kokkos allocation on the GPU? Defines MPI_Isend and MPI_Irecv (they forward
// to PMPI_*), so include it in one source file of a program only, after
// track_kokkos_allocations() is available (device_memory.h).

#include "device_memory.h"

#include <execinfo.h>
#include <mpi.h>

#include <cstdlib>

namespace multigrid
{
  namespace internal
  {
    struct MPIBufferTrace
    {
      // send or receive, label and bytes of the allocation, bytes of the message
      using Key = std::tuple<std::string, std::string, std::uint64_t, std::uint64_t>;

      static std::map<Key, std::uint64_t> &
      calls()
      {
        static std::map<Key, std::uint64_t> map;
        return map;
      }

      static std::map<Key, std::vector<std::string>> &
      stacks()
      {
        static std::map<Key, std::vector<std::string>> map;
        return map;
      }

      // GPU allocations of 16 MiB and more; for a test without a GPU:
      // MPI_BUFFER_TRACE_SPACE=Host MPI_BUFFER_TRACE_MIN_MIB=0
      static const std::string &
      space()
      {
        static const std::string name =
          std::getenv("MPI_BUFFER_TRACE_SPACE") ? std::getenv("MPI_BUFFER_TRACE_SPACE") : "Cuda";
        return name;
      }

      static std::uint64_t
      min_allocation()
      {
        static const std::uint64_t size =
          1048576 * (std::getenv("MPI_BUFFER_TRACE_MIN_MIB") ?
                       std::atoll(std::getenv("MPI_BUFFER_TRACE_MIN_MIB")) :
                       16);
        return size;
      }

      static void
      record(const char *kind, const void *buffer, const int count, const MPI_Datatype type)
      {
        int type_size = 0;
        PMPI_Type_size(type, &type_size);
        const std::uint64_t n_bytes = static_cast<std::uint64_t>(count) * type_size;

        const std::lock_guard<std::mutex> lock(LiveAllocations::mutex());
        auto                             &entries = LiveAllocations::entries();
        auto                              it      = entries.upper_bound(buffer);
        if (it == entries.begin())
          return;
        --it;
        const char *begin = static_cast<const char *>(it->first);
        if (static_cast<const char *>(buffer) >= begin + it->second.size ||
            it->second.space != space() || it->second.size < min_allocation())
          return;

        const Key key(kind, it->second.label, it->second.size, n_bytes);
        if (calls()[key]++ == 0 && it->second.size >= 4 * n_bytes && stacks().size() < 8)
          {
            void       *frames[24];
            const int   n       = backtrace(frames, 24);
            char      **symbols = backtrace_symbols(frames, n);
            auto       &stack   = stacks()[key];
            for (int i = 2; i < n && symbols != nullptr; ++i)
              stack.emplace_back(symbols[i]);
            std::free(symbols);
          }
      }
    };
  } // namespace internal



  // This rank's MPI_Isend / MPI_Irecv calls since the last call whose buffer
  // lies in a GPU allocation of 16 MiB or more, and the call stack of the
  // first call of a kind where the allocation is at least 4 x the message.
  inline void
  print_mpi_buffers(const std::string &label, std::ostream &out)
  {
    const std::lock_guard<std::mutex> lock(internal::LiveAllocations::mutex());
    out << "MPI buffers inside GPU allocations of this rank, " << label << ":" << std::endl;
    for (const auto &[key, count] : internal::MPIBufferTrace::calls())
      out << "    " << std::get<0>(key) << " of " << std::get<3>(key) / 1048576. << " MiB in a "
          << std::get<2>(key) / 1048576. << " MiB allocation (" << std::get<1>(key) << "): "
          << count << " calls" << std::endl;
    for (const auto &[key, stack] : internal::MPIBufferTrace::stacks())
      {
        out << "  call stack, " << std::get<0>(key) << " of " << std::get<3>(key) / 1048576.
            << " MiB in a " << std::get<2>(key) / 1048576. << " MiB allocation:" << std::endl;
        for (const std::string &frame : stack)
          out << "      " << frame << std::endl;
      }
    internal::MPIBufferTrace::calls().clear();
    internal::MPIBufferTrace::stacks().clear();
  }
} // namespace multigrid



// the names in parentheses: an MPI header may define them as macros too
extern "C" int
(MPI_Isend)(const void  *buffer,
          int          count,
          MPI_Datatype type,
          int          dest,
          int          tag,
          MPI_Comm     comm,
          MPI_Request *request)
{
  multigrid::internal::MPIBufferTrace::record("send", buffer, count, type);
  return PMPI_Isend(buffer, count, type, dest, tag, comm, request);
}

// the names in parentheses: an MPI header may define them as macros too
extern "C" int
(MPI_Irecv)(void        *buffer,
          int          count,
          MPI_Datatype type,
          int          source,
          int          tag,
          MPI_Comm     comm,
          MPI_Request *request)
{
  multigrid::internal::MPIBufferTrace::record("receive", buffer, count, type);
  return PMPI_Irecv(buffer, count, type, source, tag, comm, request);
}

#endif
