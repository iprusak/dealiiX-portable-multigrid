#ifndef bk1_kokkos_kernels_h
#define bk1_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK1
{
  namespace Parallel
  {

    template <typename Number>
    using DeviceView =
      Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView =
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <typename Number>
    using WeightsView =
      Kokkos::View<Number **, MemorySpace::Default::kokkos_space>;

    template <int                dim,
              const unsigned int nm_coarse,
              const unsigned int nm_fine,
              typename Number>
    void
    KokkosProlongationKernel(
      const DeviceView<Number>  d_shape_values,
      const DeviceView<Number>  d_in,
      DeviceView<Number>        d_out,
      const DoFIndicesView      dof_indices_coarse,
      const DoFIndicesView      dof_indices_fine,
      const WeightsView<Number> weights,
      const unsigned int        n_cells,
      unsigned int              numThreads      = numbers::invalid_unsigned_int,
      unsigned int              threadsPerBlock = numbers::invalid_unsigned_int)

    {
      AssertDimension(dim, 3);

      constexpr unsigned nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr unsigned nm_fine_total   = Utilities::pow(nm_fine, dim);

      if (numThreads == numbers::invalid_unsigned_int)
        numThreads = n_cells * nm_fine_total / 2;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = nm_fine_total;

      unsigned int numBlocks =
        numThreads / (std::min(nm_fine_total, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;

      {
        const unsigned int ssize = nm_coarse * nm_fine + // kernel matrix
                                   2 * nm_fine_total;    // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));


        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch =
              (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nm_fine_total;

            unsigned int threadIdx = team_member.team_rank();
            unsigned int blockSize = team_member.team_size();

            for (unsigned int tid = threadIdx; tid < nm_coarse * nm_fine;
                 tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            unsigned int cell_index = team_member.league_rank();

            while (cell_index < n_cells)
              {
                // step-1 : Copy from in to the scratch values
                for (unsigned int tid = threadIdx; tid < nm_coarse_total;
                     tid += blockSize)
                  {
                    // Fetch the global DoF index
                    const unsigned int dof_index =
                      dof_indices_coarse(tid, cell_index);

                    if (dof_index == numbers::invalid_unsigned_int)
                      s_wsp0[tid] = 0;
                    else
                      s_wsp0[tid] = d_in[dof_index];
                  }
                team_member.team_barrier();

                // step-2 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < nm_fine * nm_coarse * nm_coarse;
                     tid += blockSize)
                  {
                    unsigned int p = tid / (nm_coarse * nm_coarse);
                    unsigned int j =
                      (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                    unsigned int k = tid % nm_coarse;

                    Number tmp = 0.;

                    for (unsigned int i = 0; i < nm_coarse; ++i)
                      {
                        tmp +=
                          s_shape_values[i * nm_fine + p] *
                          s_wsp0[k * nm_coarse * nm_coarse + j * nm_coarse + i];
                      }

                    s_wsp1[k * nm_coarse * nm_fine + j * nm_fine + p] = tmp;
                  }
                team_member.team_barrier();

                // step-3 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < nm_fine * nm_fine * nm_coarse;
                     tid += blockSize)
                  {
                    unsigned int i = tid / (nm_fine * nm_coarse);
                    unsigned int q = (tid % (nm_fine * nm_coarse)) / nm_coarse;
                    unsigned int k = tid % nm_coarse;

                    Number tmp = 0.;

                    for (unsigned int j = 0; j < nm_coarse; ++j)
                      {
                        tmp +=
                          s_shape_values[j * nm_fine + q] *
                          s_wsp1[k * nm_fine * nm_coarse + j * nm_fine + i];
                      }

                    s_wsp0[k * nm_fine * nm_fine + q * nm_fine + i] = tmp;
                  }
                team_member.team_barrier();

                // step-4 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < nm_fine * nm_fine * nm_fine;
                     tid += blockSize)
                  {
                    unsigned int i = tid / (nm_fine * nm_fine);
                    unsigned int j = (tid % (nm_fine * nm_fine)) / nm_fine;
                    unsigned int r = tid % nm_fine;

                    Number tmp = 0.;

                    for (unsigned int k = 0; k < nm_coarse; ++k)
                      {
                        tmp += s_shape_values[k * nm_fine + r] *
                               s_wsp0[k * nm_fine * nm_fine + j * nm_fine + i];
                      }

                    s_wsp1[r * nm_fine * nm_fine + j * nm_fine + i] = tmp;
                  }
                team_member.team_barrier();

                // step-9 : Copy s_wsp1 to out
                for (unsigned int tid = threadIdx;
                     tid < nm_fine * nm_fine * nm_fine;
                     tid += blockSize)
                  {
                    // Find where this node lives in the global 'd_out'
                    // vector
                    const unsigned int dof_index =
                      dof_indices_fine(tid, cell_index);

                    // apply weights
                    Number value_out = weights(tid, cell_index) * s_wsp1[tid];

                    // CRITICAL: Use atomic_add because elements share
                    // nodes!
                    Kokkos::atomic_add(&d_out[dof_index], value_out);
                  }

                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

    template <int                dim,
              const unsigned int nm_coarse,
              const unsigned int nm_fine,
              typename Number>
    void
    KokkosRestrictionKernel(
      const DeviceView<Number>  d_shape_values,
      const DeviceView<Number>  d_in,
      DeviceView<Number>        d_out,
      const DoFIndicesView      dof_indices_coarse,
      const DoFIndicesView      dof_indices_fine,
      const WeightsView<Number> weights,
      const unsigned int        n_cells,
      unsigned int              numThreads      = numbers::invalid_unsigned_int,
      unsigned int              threadsPerBlock = numbers::invalid_unsigned_int)

    {
      AssertDimension(dim, 3);

      constexpr unsigned nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr unsigned nm_fine_total   = Utilities::pow(nm_fine, dim);

      if (numThreads == numbers::invalid_unsigned_int)
        numThreads = n_cells * nm_fine_total / 2;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = nm_fine_total;

      unsigned int numBlocks =
        numThreads / (std::min(nm_fine_total, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;

      {
        const unsigned int ssize = nm_coarse * nm_fine + // kernel matrix
                                   2 * nm_fine_total;    // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch =
              (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nm_fine_total;

            unsigned int threadIdx = team_member.team_rank();
            unsigned int blockSize = team_member.team_size();

            for (unsigned int tid = threadIdx; tid < nm_coarse * nm_fine;
                 tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            unsigned int cell_index = team_member.league_rank();

            while (cell_index < n_cells)
              {
                // step-1 : Copy from in to the scratch values
                for (unsigned int tid = threadIdx; tid < nm_fine_total;
                     tid += blockSize)
                  {
                    // Fetch the global DoF index
                    const unsigned int dof_index =
                      dof_indices_fine(tid, cell_index);

                    // read dof value and apply weights
                    Number value_in =
                      weights(tid, cell_index) * d_in[dof_index];

                    s_wsp0[tid] = value_in;
                  }
                team_member.team_barrier();

                // step-2 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < nm_fine * nm_fine * nm_coarse;
                     tid += blockSize)
                  {
                    unsigned int i = tid / (nm_fine * nm_coarse);
                    unsigned int j = (tid % (nm_fine * nm_coarse)) / nm_coarse;
                    unsigned int r = tid % nm_coarse;

                    Number tmp = 0.;
                    for (unsigned int k = 0; k < nm_fine; ++k)
                      {
                        tmp += s_shape_values[r * nm_fine + k] *
                               s_wsp0[k * nm_fine * nm_fine + j * nm_fine + i];
                      }

                    s_wsp1[r * nm_fine * nm_fine + j * nm_fine + i] = tmp;
                  }
                team_member.team_barrier();

                // step-3 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < nm_fine * nm_coarse * nm_coarse;
                     tid += blockSize)
                  {
                    unsigned int i = tid / (nm_coarse * nm_coarse);
                    unsigned int q =
                      (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                    unsigned int k = tid % nm_coarse;

                    Number tmp = 0.;

                    for (unsigned int j = 0; j < nm_fine; ++j)
                      {
                        tmp += s_shape_values[q * nm_fine + j] *
                               s_wsp1[k * nm_fine * nm_fine + j * nm_fine + i];
                      }
                    s_wsp0[k * nm_fine * nm_coarse + q * nm_fine + i] = tmp;
                  }
                team_member.team_barrier();

                // step-4 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < nm_coarse * nm_coarse * nm_coarse;
                     tid += blockSize)
                  {
                    unsigned int p = tid / (nm_coarse * nm_coarse);
                    unsigned int j =
                      (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                    unsigned int k = tid % nm_coarse;

                    Number tmp = 0.;

                    for (unsigned int i = 0; i < nm_fine; ++i)
                      {
                        tmp +=
                          s_shape_values[p * nm_fine + i] *
                          s_wsp0[k * nm_fine * nm_coarse + j * nm_fine + i];
                      }

                    s_wsp1[k * nm_coarse * nm_coarse + j * nm_coarse + p] = tmp;
                  }
                team_member.team_barrier();

                // step-9 : Copy s_wsp1 to out
                for (unsigned int tid = threadIdx; tid < nm_coarse_total;
                     tid += blockSize)
                  {
                    // Find where this node lives in the global 'd_out'
                    // vector
                    const unsigned int dof_index =
                      dof_indices_coarse(tid, cell_index);

                    // CRITICAL: Use atomic_add because elements share
                    // nodes!
                    if (dof_index != numbers::invalid_unsigned_int)
                      Kokkos::atomic_add(&d_out[dof_index], s_wsp1[tid]);
                  }

                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

    template <int                dim,
              const unsigned int nm_coarse,
              const unsigned int nm_fine,
              typename Number>
    void
    KokkosProlongationBatchedKernel(
      const DeviceView<Number>  d_shape_values,
      const DeviceView<Number>  d_in,
      DeviceView<Number>        d_out,
      const DoFIndicesView      dof_indices_coarse,
      const DoFIndicesView      dof_indices_fine,
      const WeightsView<Number> weights,
      const unsigned int        n_cells,
      unsigned int              numBlocks       = numbers::invalid_unsigned_int,
      unsigned int              threadsPerBlock = numbers::invalid_unsigned_int)

    {
      AssertDimension(dim, 3);

      constexpr unsigned nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr unsigned nm_fine_total   = Utilities::pow(nm_fine, dim);

      constexpr int shmemPerBlock = 10800;

      unsigned int nelmtPerBatch =
        shmemPerBlock / (2 * nm_fine_total) / sizeof(Number);

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;

      if (numBlocks == numbers::invalid_unsigned_int)
        numBlocks = (n_cells + nelmtPerBatch - 1) / nelmtPerBatch / 2;

      if (numBlocks == 0)
        numBlocks = 1;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = nm_fine * nm_fine * nelmtPerBatch;

      {
        const unsigned int ssize =
          nm_coarse * nm_fine +              // kernel matrix
          2 * nelmtPerBatch * nm_fine_total; // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));


        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch =
              (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nelmtPerBatch * nm_fine_total;

            Number reg[nm_fine];

            unsigned int threadIdx = team_member.team_rank();
            unsigned int blockSize = team_member.team_size();

            for (unsigned int tid = threadIdx; tid < nm_coarse * nm_fine;
                 tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            unsigned int cell_batch_index = team_member.league_rank();

            while (cell_batch_index <
                   (n_cells + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const unsigned int c_nelmtPerBatch =
                  (cell_batch_index * nelmtPerBatch + nelmtPerBatch > n_cells) ?
                    (n_cells - cell_batch_index * nelmtPerBatch) :
                    nelmtPerBatch;

                // step-1 : Copy from in to the scratch values
                for (unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm_coarse_total; tid += blockSize)
                {
                    const int e = tid / nm_coarse_total;
                    const int local_idx = tid % nm_coarse_total;
                
                    const unsigned int global_cell_index = cell_batch_index * nelmtPerBatch + e;
                
                    const unsigned int dof_index = dof_indices_coarse(local_idx, global_cell_index);
                
                    if (dof_index == numbers::invalid_unsigned_int)
                    {
                        s_wsp0[tid] = 0.0;
                    }
                    else
                    {
                        s_wsp0[tid] = d_in[dof_index];
                    }
                }
                team_member.team_barrier();

                // step-2 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_coarse * nm_coarse;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_coarse * nm_coarse);

                    const unsigned int j =
                      (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                    const unsigned int k = tid % nm_coarse;

                    for (unsigned i = 0; i < nm_coarse; ++i)
                      {
                        reg[i] =
                          s_wsp0[batch_id * nm_coarse * nm_coarse * nm_coarse +
                                 k * nm_coarse * nm_coarse + j * nm_coarse + i];
                      }

                    for (unsigned int p = 0; p < nm_fine; ++p)
                      {
                        Number tmp = 0.;
                        for (unsigned int i = 0; i < nm_coarse; ++i)
                          {
                            tmp += s_shape_values[i * nm_fine + p] * reg[i];
                          }

                        s_wsp1[batch_id * nm_fine * nm_coarse * nm_coarse +
                               k * nm_fine * nm_coarse + j * nm_fine + p] = tmp;
                      }
                  }
                team_member.team_barrier();


                // step-3 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_fine * nm_coarse;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_fine * nm_coarse);

                    const unsigned int i =
                      (tid % (nm_fine * nm_coarse)) / nm_coarse;
                    const unsigned int k = tid % nm_coarse;

                    for (unsigned j = 0; j < nm_coarse; ++j)
                      {
                        reg[j] =
                          s_wsp1[batch_id * nm_fine * nm_coarse * nm_coarse +
                                 k * nm_fine * nm_coarse + j * nm_fine + i];
                      }

                    for (unsigned int q = 0; q < nm_fine; ++q)
                      {
                        Number tmp = 0.;
                        for (unsigned int j = 0; j < nm_coarse; ++j)
                          {
                            tmp += s_shape_values[j * nm_fine + q] * reg[j];
                          }

                        s_wsp0[batch_id * nm_fine * nm_fine * nm_coarse +
                               k * nm_fine * nm_fine + q * nm_fine + i] = tmp;
                      }
                  }
                team_member.team_barrier();

                // step-4 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_fine * nm_fine;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_fine * nm_fine);

                    const unsigned int i =
                      (tid % (nm_fine * nm_fine)) / nm_fine;
                    const unsigned int j = tid % nm_fine;

                    for (unsigned k = 0; k < nm_coarse; ++k)
                      {
                        reg[k] =
                          s_wsp0[batch_id * nm_fine * nm_fine * nm_coarse +
                                 k * nm_fine * nm_fine + j * nm_fine + i];
                      }

                    for (unsigned int r = 0; r < nm_fine; ++r)
                      {
                        Number tmp = 0.;
                        for (unsigned int k = 0; k < nm_coarse; ++k)
                          {
                            tmp += s_shape_values[k * nm_fine + r] * reg[k];
                          }

                        s_wsp1[batch_id * nm_fine * nm_fine * nm_fine +
                               r * nm_fine * nm_fine + j * nm_fine + i] = tmp;
                      }
                  }
                team_member.team_barrier();

                // step-9 : Copy s_wsp1 to out (Scatter/Add)
                for (unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm_fine_total; tid += blockSize)
                {
                    const int e = tid / nm_fine_total;
                    const int local_idx = tid % nm_fine_total;
                
                    const unsigned int global_cell_index = cell_batch_index * nelmtPerBatch + e;
                
                    const unsigned int dof_index = dof_indices_fine(local_idx, global_cell_index);
                
                    if (dof_index != numbers::invalid_unsigned_int)
                    {
                        Number value_out = weights(local_idx, global_cell_index) * s_wsp1[tid];
                    
                        // CRITICAL: Use atomic_add because elements share nodes!
                        Kokkos::atomic_add(&d_out[dof_index], value_out);
                    }
                }
                team_member.team_barrier();

                cell_batch_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }

    template <int                dim,
              const unsigned int nm_coarse,
              const unsigned int nm_fine,
              typename Number>
    void
    KokkosRestrictionBatchedKernel(
      const DeviceView<Number>  d_shape_values,
      const DeviceView<Number>  d_in,
      DeviceView<Number>        d_out,
      const DoFIndicesView      dof_indices_coarse,
      const DoFIndicesView      dof_indices_fine,
      const WeightsView<Number> weights,
      const unsigned int        n_cells,
      unsigned int              numBlocks       = numbers::invalid_unsigned_int,
      unsigned int              threadsPerBlock = numbers::invalid_unsigned_int)

    {
      AssertDimension(dim, 3);

      constexpr unsigned nm_coarse_total = Utilities::pow(nm_coarse, dim);
      constexpr unsigned nm_fine_total   = Utilities::pow(nm_fine, dim);

      constexpr int shmemPerBlock = 10800;

      unsigned int nelmtPerBatch =
        shmemPerBlock / (2 * nm_fine_total) / sizeof(Number);

      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;

      if (numBlocks == numbers::invalid_unsigned_int)
        numBlocks = (n_cells + nelmtPerBatch - 1) / nelmtPerBatch / 2;

      if (numBlocks == 0)
        numBlocks = 1;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = nm_fine * nm_fine * nelmtPerBatch;

      {
        const unsigned int ssize =
          nm_coarse * nm_fine +              // kernel matrix
          2 * nelmtPerBatch * nm_fine_total; // temp vectors

        const unsigned int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));


        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number *scratch =
              (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values = scratch;

            Number *s_wsp0 = s_shape_values + nm_coarse * nm_fine;
            Number *s_wsp1 = s_wsp0 + nelmtPerBatch * nm_fine_total;

            Number reg[nm_fine];

            unsigned int threadIdx = team_member.team_rank();
            unsigned int blockSize = team_member.team_size();

            for (unsigned int tid = threadIdx; tid < nm_coarse * nm_fine;
                 tid += blockSize)
              s_shape_values[tid] = d_shape_values(tid);

            team_member.team_barrier();

            // element index
            unsigned int cell_batch_index = team_member.league_rank();

            while (cell_batch_index <
                   (n_cells + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const unsigned int c_nelmtPerBatch =
                  (cell_batch_index * nelmtPerBatch + nelmtPerBatch > n_cells) ?
                    (n_cells - cell_batch_index * nelmtPerBatch) :
                    nelmtPerBatch;

                // step-1 : Copy from in to the scratch values
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_fine * nm_fine;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_fine * nm_fine);

                    const int j = (tid % (nm_fine * nm_fine)) / nm_fine;
                    const int k = tid % nm_fine;

                    const unsigned int global_cell_index =
                      cell_batch_index * nelmtPerBatch + batch_id;

                    for (unsigned int i = 0; i < nm_fine; ++i)
                      {
                        // Calculate the flat local index within the 3D
                        // element
                        const int local_idx =
                          k * nm_fine * nm_fine + j * nm_fine + i;

                        // Fetch the global DoF index
                        const unsigned int dof_index =
                          dof_indices_fine(local_idx, global_cell_index);

                        // The index in the batched shared memory array
                        const int shared_idx =
                          batch_id * nm_fine_total + local_idx;

                        Number value_in =
                          weights(local_idx, global_cell_index) *
                          d_in[dof_index];

                        s_wsp0[shared_idx] = value_in;
                      }
                  }
                team_member.team_barrier();

                // step-2 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_fine * nm_fine;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_fine * nm_fine);

                    const unsigned int i =
                      (tid % (nm_fine * nm_fine)) / nm_fine;
                    const unsigned int j = tid % nm_fine;

                    for (unsigned k = 0; k < nm_fine; ++k)
                      {
                        reg[k] =
                          s_wsp0[batch_id * nm_fine * nm_fine * nm_fine +
                                 k * nm_fine * nm_fine + j * nm_fine + i];
                      }

                    for (unsigned int r = 0; r < nm_coarse; ++r)
                      {
                        Number tmp = 0.;
                        for (unsigned int k = 0; k < nm_fine; ++k)
                          {
                            tmp += s_shape_values[r * nm_fine + k] * reg[k];
                          }

                        s_wsp1[batch_id * nm_fine * nm_fine * nm_coarse +
                               r * nm_fine * nm_fine + j * nm_fine + i] = tmp;
                      }
                  }
                team_member.team_barrier();

                // step-3 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_fine * nm_coarse;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_fine * nm_coarse);

                    const unsigned int i =
                      (tid % (nm_fine * nm_coarse)) / nm_coarse;
                    const unsigned int k = tid % nm_coarse;

                    for (unsigned j = 0; j < nm_fine; ++j)
                      {
                        reg[j] =
                          s_wsp1[batch_id * nm_fine * nm_fine * nm_coarse +
                                 k * nm_fine * nm_fine + j * nm_fine + i];
                      }

                    for (unsigned int q = 0; q < nm_coarse; ++q)
                      {
                        Number tmp = 0.;
                        for (unsigned int j = 0; j < nm_fine; ++j)
                          {
                            tmp += s_shape_values[q * nm_fine + j] * reg[j];
                          }

                        s_wsp0[batch_id * nm_fine * nm_coarse * nm_coarse +
                               k * nm_fine * nm_coarse + q * nm_fine + i] = tmp;
                      }
                  }
                team_member.team_barrier();

                // step-4 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_coarse * nm_coarse;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_coarse * nm_coarse);

                    const unsigned int j =
                      (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                    const unsigned int k = tid % nm_coarse;

                    for (unsigned i = 0; i < nm_fine; ++i)
                      {
                        reg[i] =
                          s_wsp0[batch_id * nm_fine * nm_coarse * nm_coarse +
                                 k * nm_fine * nm_coarse + j * nm_fine + i];
                      }

                    for (unsigned int p = 0; p < nm_coarse; ++p)
                      {
                        Number tmp = 0.;
                        for (unsigned int i = 0; i < nm_fine; ++i)
                          {
                            tmp += s_shape_values[p * nm_fine + i] * reg[i];
                          }

                        s_wsp1[batch_id * nm_coarse * nm_coarse * nm_coarse +
                               k * nm_coarse * nm_coarse + j * nm_coarse + p] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                // step-9 : Copy s_wsp1 to out
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm_coarse * nm_coarse;
                     tid += blockSize)
                  {
                    const unsigned int batch_id = tid / (nm_coarse * nm_coarse);

                    const int j = (tid % (nm_coarse * nm_coarse)) / nm_coarse;
                    const int k = tid % nm_coarse;

                    const unsigned int global_cell_index =
                      cell_batch_index * nelmtPerBatch + batch_id;

                    for (unsigned int i = 0; i < nm_coarse; ++i)
                      {
                        // Calculate the flat local index within the 3D
                        // element
                        const int local_idx =
                          k * nm_coarse * nm_coarse + j * nm_coarse + i;

                        // Fetch the global DoF index
                        const unsigned int dof_index =
                          dof_indices_coarse(local_idx, global_cell_index);

                        // The index in the batched shared memory array
                        const int shared_idx =
                          batch_id * nm_coarse_total + local_idx;

                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        if (dof_index != numbers::invalid_unsigned_int)
                          Kokkos::atomic_add(&d_out[dof_index],
                                             s_wsp1[shared_idx]);
                      }
                  }

                team_member.team_barrier();

                cell_batch_index += team_member.league_size();
              }
          });

        Kokkos::fence();
      }
    }
  } // namespace Parallel
} // namespace BK1

DEAL_II_NAMESPACE_CLOSE

#endif