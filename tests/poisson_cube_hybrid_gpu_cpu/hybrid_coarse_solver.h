#pragma once

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>

#include <Kokkos_Core.hpp>

#include <iomanip>
#include <ostream>

#include "hybrid_cpu_side.h"
#include "hybrid_mpi_layout.h"
#include "hybrid_service.h"

namespace hybrid
{
  using namespace dealii;

  /**
   * Coarse solver of the GPU side, called by its V-cycle on the switch level
   * s-1 (GPU ranks only): copy the right-hand side to the host, let the
   * block's CPU-only ranks join, run one V-cycle on the CPU levels (P in and
   * out inside CpuSide::coarse_solve()), copy the correction back.
   */
  template <int dim>
  class HybridCoarseSolver
    : public MGCoarseGridBase<
        LinearAlgebra::distributed::Vector<typename CpuSide<dim>::vcycle_number,
                                           MemorySpace::Default>>
  {
  public:
    using Number     = typename CpuSide<dim>::vcycle_number;
    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

    HybridCoarseSolver(const MPILayout &layout, CpuSide<dim> &cpu_side)
      : layout(layout)
      , cpu_side(cpu_side)
    {}

    void
    operator()(const unsigned int, VectorType &dst, const VectorType &src) const override
    {
      std::vector<Number> &host_values = cpu_side.gpu_values();
      AssertDimension(host_values.size(), src.locally_owned_size());

      Timer time;

      // wake up the CPU-only ranks of the block
      send_command(layout, Command::solve);

      // device -> host, owned entries; same layout on both ends
      Kokkos::deep_copy(host_view(host_values), device_view(src));
      times[0] += time.wall_time();

      // P down, V-cycle on the CPU levels (all ranks), P up
      time.restart();
      cpu_side.coarse_solve();
      times[1] += time.wall_time();

      // host -> device
      time.restart();
      Kokkos::deep_copy(device_view(dst), host_view(host_values));
      times[2] += time.wall_time();

      ++n_calls;
    }

    // Collective on comm_gpu: accumulated times (max over GPU ranks).
    void
    print_times(std::ostream &out) const
    {
      double max_times[3];
      for (unsigned int i = 0; i < 3; ++i)
        max_times[i] = Utilities::MPI::max(times[i], layout.comm_gpu);

      if (Utilities::MPI::this_mpi_process(layout.comm_gpu) == 0)
        out << "Hybrid coarse solver: " << n_calls << " calls, device->host "
            << std::setprecision(3) << max_times[0] << " s, CPU levels " << max_times[1]
            << " s, host->device " << max_times[2] << " s" << std::endl;
    }

    void
    reset_times()
    {
      times[0] = times[1] = times[2] = 0.;
      n_calls                        = 0;
    }

  private:
    using HostView =
      Kokkos::View<Number *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using DeviceView = Kokkos::View<Number *,
                                    MemorySpace::Default::kokkos_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static HostView
    host_view(std::vector<Number> &values)
    {
      return HostView(values.data(), values.size());
    }

    // locally owned entries of a level vector
    static DeviceView
    device_view(const VectorType &v)
    {
      return DeviceView(const_cast<Number *>(v.get_values()), v.locally_owned_size());
    }

    const MPILayout &layout;
    CpuSide<dim>    &cpu_side;

    mutable double       times[3] = {0., 0., 0.}; // device->host, CPU levels, host->device
    mutable unsigned int n_calls  = 0;
  };
} // namespace hybrid
