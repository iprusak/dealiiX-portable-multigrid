#ifndef bk3_kokkos_kernel_h
#define bk3_kokkos_kernel_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
{
  namespace Parallel
  {

    template <typename number>
    using DeviceView =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView =
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename number>
    void
    KokkosKernel(const DeviceView<number> d_shape_values,
                 const DeviceView<number> d_co_shape_gradients,
                 const DeviceView<number> d_G,
                 const DeviceView<number> d_in,
                 DeviceView<number>       d_out,
                 const DoFIndicesView     dof_indices,
                 const unsigned int       n_cells,
                 unsigned int numBlocks       = numbers::invalid_unsigned_int,
                 unsigned int threadsPerBlock = numbers::invalid_unsigned_int)
    {
      constexpr unsigned nq_total = Utilities::pow(nq, dim);
      constexpr unsigned nm_total = Utilities::pow(nm, dim);


      // finding the batch size
      int shmemPerBlock = 10800; // total shared memory used per block (KB)

      unsigned int nelmtPerBatch =
        shmemPerBlock / (4 * nq_total) / sizeof(number);
      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;

      if (nelmtPerBatch > n_cells)
        nelmtPerBatch = n_cells;

      if (numBlocks == numbers::invalid_unsigned_int)
        numBlocks = (n_cells + nelmtPerBatch - 1) / nelmtPerBatch / 2;

      if (numBlocks == 0)
        numBlocks = 1;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = nq * nq * std::max(1u, nelmtPerBatch);

      {
        unsigned int ssize =
          nm * nq + // shape values
          nq * nq + // co-shape gradients
          4 * nelmtPerBatch *
            nq_total; // working scratch arrays: s_wsp0, s_wsp1, rqr, rqs, rqt


        const unsigned int shmem_size = ssize * sizeof(number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            number r_p[nq];
            number r_q[nq];
            number r_r[nq];

            number *scratch =
              (number *)team_member.team_shmem().get_shmem(shmem_size);

            number *s_shape_values       = scratch;
            number *s_co_shape_gradients = s_shape_values + nq * nm;

            number *s_wsp0 = s_co_shape_gradients + nq * nq;
            number *s_wsp1 = s_wsp0 + nelmtPerBatch * nq * nq * nq;

            number *s_rqr = s_wsp1 + nelmtPerBatch * nq * nq * nq;
            number *s_rqs = s_rqr + nelmtPerBatch * nq * nq * nq;
            number *s_rqt = s_wsp0;

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();

            // copy to shared memory
            for (unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = d_shape_values[tid];
              }

            for (unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
              }
            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            // element batch iteration
            unsigned int eb = team_member.league_rank();
            while (eb < (n_cells + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                unsigned int c_nelmtPerBatch =
                  (eb * nelmtPerBatch + nelmtPerBatch > n_cells) ?
                    (n_cells - eb * nelmtPerBatch) :
                    nelmtPerBatch;


                // step-1 : Copy from in to the scratch values
                for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                {
                    const int e = tid / nm_total;
                    const int local_idx = tid % nm_total; 
                
                    const unsigned int global_cell_index = eb * nelmtPerBatch + e;
                
                    const unsigned int dof_index = dof_indices(local_idx, global_cell_index);
                
                    if (dof_index == numbers::invalid_unsigned_int)
                        s_wsp0[tid] = 0;
                    else
                        s_wsp0[tid] = d_in[dof_index];
                }
                team_member.team_barrier();

                // step-2 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm * nm;
                     tid += blockSize)
                  {
                    int e = tid / (nm * nm);
                    int j = (tid % (nm * nm)) / nm;
                    int k = tid % nm;

                    for (int i = 0; i < nm; ++i)
                      {
                        r_p[i] =
                          s_wsp0[e * nm * nm * nm + k * nm * nm + j * nm + i];
                      }

                    for (int p = 0; p < nq; ++p)
                      {
                        number tmp = 0.0;

                        for (int i = 0; i < nm; ++i)
                          {
                            tmp += s_shape_values[i * nq + p] * r_p[i];
                          }

                        s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + p] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                // step-3 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nq * nm;
                     tid += blockSize)
                  {
                    int e = tid / (nq * nm);

                    int i = (tid % (nq * nm)) / nm;
                    int k = tid % nm;

                    for (int j = 0; j < nm; ++j)
                      {
                        r_q[j] =
                          s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + i];
                      }

                    for (int q = 0; q < nq; ++q)
                      {
                        number tmp = 0.0;

                        for (int j = 0; j < nm; ++j)
                          {
                            tmp += s_shape_values[j * nq + q] * r_q[j];
                          }

                        s_wsp0[e * nq * nq * nm + k * nq * nq + q * nq + i] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                // step-4 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nq * nq;
                     tid += blockSize)
                  {
                    int e = tid / (nq * nq);

                    int i = (tid % (nq * nq)) / nq;
                    int j = tid % nq;

                    for (int k = 0; k < nm; ++k)
                      {
                        r_r[k] =
                          s_wsp0[e * nq * nq * nm + k * nq * nq + j * nq + i];
                      }
                    for (int r = 0; r < nq; ++r)
                      {
                        number tmp = 0.0;

                        for (int k = 0; k < nm; ++k)
                          {
                            tmp += s_shape_values[k * nq + r] * r_r[k];
                          }

                        s_wsp1[e * nq * nq * nq + r * nq * nq + j * nq + i] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nq * nq;
                     tid += blockSize)
                  {
                    int e = tid / (nq * nq);
                    int q = tid % (nq * nq) / nq;
                    int r = tid % nq;

                    // copy to register
                    for (unsigned int n = 0; n < nq; n++)
                      {
                        r_p[n] =
                          s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + n];
                        r_q[n] = s_co_shape_gradients[n * nq + q];
                        r_r[n] = s_co_shape_gradients[n * nq + r];
                      }

                    number Grr, Grs, Grt, Gss, Gst, Gtt;
                    number qr, qs, qt;

                    for (unsigned int p = 0; p < nq; ++p)
                      {
                        qr = 0;
                        qs = 0;
                        qt = 0;

                        // Load Geometric Factors, coalesced access
                        Grr = d_G[eb * nelmtPerBatch * 6 * nq_total +
                                  e * 6 * nq_total + 0 * nq_total +
                                  r * nq * nq + q * nq + p];

                        Grs = d_G[eb * nelmtPerBatch * 6 * nq_total +
                                  e * 6 * nq_total + 1 * nq_total +
                                  r * nq * nq + q * nq + p];

                        Grt = d_G[eb * nelmtPerBatch * 6 * nq_total +
                                  e * 6 * nq_total + 2 * nq_total +
                                  r * nq * nq + q * nq + p];

                        Gss = d_G[eb * nelmtPerBatch * 6 * nq_total +
                                  e * 6 * nq_total + 3 * nq_total +
                                  r * nq * nq + q * nq + p];

                        Gst = d_G[eb * nelmtPerBatch * 6 * nq_total +
                                  e * 6 * nq_total + 4 * nq_total +
                                  r * nq * nq + q * nq + p];

                        Gtt = d_G[eb * nelmtPerBatch * 6 * nq_total +
                                  e * 6 * nq_total + 5 * nq_total +
                                  r * nq * nq + q * nq + p];

                        // Multiply by D
                        for (unsigned int n = 0; n < nq; n++)
                          {
                            qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                            qs += r_q[n] * s_wsp1[e * nq * nq * nq +
                                                  r * nq * nq + n * nq + p];
                            qt += r_r[n] * s_wsp1[e * nq * nq * nq +
                                                  n * nq * nq + q * nq + p];
                          }

                        // Apply chain rule
                        s_rqr[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                          Grr * qr + Grs * qs + Grt * qt;

                        s_rqs[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                          Grs * qr + Gss * qs + Gst * qt;

                        s_rqt[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                          Grt * qr + Gst * qs + Gtt * qt;
                      }
                  }
                team_member.team_barrier();

                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nq * nq;
                     tid += blockSize)
                  {
                    int e = tid / (nq * nq);
                    int q = tid % (nq * nq) / nq;
                    int r = tid % nq;

                    // copy to register
                    for (unsigned int n = 0; n < nq; n++)
                      {
                        r_p[n] =
                          s_rqr[e * nq * nq * nq + r * nq * nq + q * nq + n];
                        r_q[n] = s_co_shape_gradients[q * nq + n];
                        r_r[n] = s_co_shape_gradients[r * nq + n];
                      }

                    for (unsigned int p = 0; p < nq; ++p)
                      {
                        number tmp0 = 0;
                        for (unsigned int n = 0; n < nq; ++n)
                          tmp0 += r_p[n] * s_co_shape_gradients[p * nq + n];

                        for (unsigned int n = 0; n < nq; ++n)
                          tmp0 +=
                            s_rqs[e * nq * nq * nq + r * nq * nq + n * nq + p] *
                            r_q[n];

                        for (unsigned int n = 0; n < nq; ++n)
                          tmp0 +=
                            s_rqt[e * nq * nq * nq + n * nq * nq + q * nq + p] *
                            r_r[n];

                        s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                          tmp0;
                      }
                  }
                team_member.team_barrier();

                /*
                Interpolate to GLL nodes
                */

                // step-9 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nq * nq;
                     tid += blockSize)
                  {
                    int e = tid / (nq * nq);

                    int i = (tid % (nq * nq)) / nq;
                    int j = tid % nq;

                    for (int k = 0; k < nq; ++k)
                      {
                        r_r[k] =
                          s_wsp1[e * nq * nq * nq + k * nq * nq + j * nq + i];
                      }

                    for (int r = 0; r < nm; ++r)
                      {
                        number tmp = 0.0;

                        for (int k = 0; k < nq; ++k)
                          {
                            tmp += s_shape_values[r * nq + k] * r_r[k];
                          }

                        s_wsp0[e * nq * nq * nm + r * nq * nq + j * nq + i] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                // step-10 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm * nq;
                     tid += blockSize)
                  {
                    int e = tid / (nq * nm);

                    int i = (tid % (nq * nm)) / nm;
                    int k = tid % nm;

                    for (int j = 0; j < nq; ++j)
                      {
                        r_q[j] =
                          s_wsp0[e * nq * nq * nm + k * nq * nq + j * nq + i];
                      }

                    for (int q = 0; q < nm; ++q)
                      {
                        number tmp = 0.0;

                        for (int j = 0; j < nq; ++j)
                          {
                            tmp += s_shape_values[q * nq + j] * r_q[j];
                          }
                        s_wsp1[e * nq * nm * nm + k * nq * nm + q * nq + i] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                for (unsigned int tid = threadIdx;
                     tid < c_nelmtPerBatch * nm * nm;
                     tid += blockSize)
                  {
                    int e = tid / (nm * nm);

                    int j = (tid % (nm * nm)) / nm;
                    int k = tid % nm;

                    for (int i = 0; i < nq; ++i)
                      {
                        r_p[i] =
                          s_wsp1[e * nq * nm * nm + k * nq * nm + j * nq + i];
                      }

                    for (int p = 0; p < nm; ++p)
                      {
                        number tmp = 0.0;
                        for (int i = 0; i < nq; ++i)
                          {
                            tmp += s_shape_values[p * nq + i] * r_p[i];
                          }
                        s_wsp0[e * nm * nm * nm + k * nm * nm + j * nm + p] =
                          tmp;
                      }
                  }
                team_member.team_barrier();

                // step-12 : Copy wsp0 (result) back to global out vector
                for (unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                {
                    const int e = tid / nm_total;
                    const int local_idx = tid % nm_total;
                
                    const unsigned int global_cell_index = eb * nelmtPerBatch + e;
                
                    const unsigned int dof_index = dof_indices(local_idx, global_cell_index);
                
                    if (dof_index != numbers::invalid_unsigned_int)
                    {
                        // CRITICAL: Use atomic_add because elements share nodes!
                        Kokkos::atomic_add(&d_out[dof_index], s_wsp0[tid]);
                    }
                }

                team_member.team_barrier();

                eb += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

    template <int dim, int n_local_dofs_1d, int n_q_points_1d, typename number>
    void
    KokkosKernel_1D_Block(
      const DeviceView<number> shape_values_device,
      const DeviceView<number> co_shape_gradients_device,
      const DeviceView<number> G_device,
      const DeviceView<number> in_device,
      DeviceView<number>       out_device,
      const DoFIndicesView     dof_indices,
      const unsigned int       n_cells,
      unsigned int             numThreads      = numbers::invalid_unsigned_int,
      unsigned int             threadsPerBlock = numbers::invalid_unsigned_int)
    {
      constexpr unsigned n_q_points_total = Utilities::pow(n_q_points_1d, dim);
      constexpr unsigned n_local_dofs_total =
        Utilities::pow(n_local_dofs_1d, dim);

      if (numThreads == numbers::invalid_unsigned_int)
        numThreads = n_cells * n_q_points_total / 2;


      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = n_q_points_total;

      unsigned int numBlocks =
        numThreads / (std::min(n_q_points_total, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;


      {
        const unsigned int scratch_pad_size =
          5 * n_q_points_total; // working scratch arrays: s_wsp0, s_wsp1,
                                // rqr,rqq, rqt

        unsigned int ssize =
          n_local_dofs_1d * n_q_points_1d + // shape values
          n_q_points_1d * n_q_points_1d +   // co-shape gradients
          scratch_pad_size;                 // at most 5 tmp arrays

        const unsigned int shmem_size = ssize * sizeof(number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            number *scratch =
              (number *)team_member.team_shmem().get_shmem(shmem_size);

            number *shape_values_scratch = scratch;
            number *co_shape_gradients_scratch =
              shape_values_scratch + n_q_points_1d * n_local_dofs_1d;

            number *s_wsp0 =
              co_shape_gradients_scratch + n_q_points_1d * n_q_points_1d;
            number *s_wsp1 = s_wsp0 + n_q_points_total;

            number *rqr = s_wsp1 + n_q_points_total;
            number *rqs = rqr + n_q_points_total;
            number *rqt = rqs + n_q_points_total;

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();


            // copy to shared memory
            {
              for (unsigned int tid = threadIdx;
                   tid < n_local_dofs_1d * n_q_points_1d;
                   tid += blockSize)
                {
                  shape_values_scratch[tid] = shape_values_device[tid];
                }

              for (unsigned int tid = threadIdx;
                   tid < n_q_points_1d * n_q_points_1d;
                   tid += blockSize)
                {
                  co_shape_gradients_scratch[tid] =
                    co_shape_gradients_device[tid];
                }
            }

            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            unsigned int cell_index = team_member.league_rank();

            while (cell_index < n_cells)
              {
                team_member.team_barrier();
                {
                  // step-1 : Copy from in to the scratch values
                  for (unsigned int tid = threadIdx; tid < n_local_dofs_total;
                       tid += blockSize)
                    {
                      const unsigned int dof_index =
                        dof_indices(tid, cell_index);

                      if (dof_index == numbers::invalid_unsigned_int)
                        s_wsp0[tid] = 0;
                      else
                        s_wsp0[tid] = in_device[dof_index];
                    }
                }
                team_member.team_barrier();

                if constexpr (dim == 3)
                  {
                    // step-2 : direction 0
                    for (unsigned int tid = threadIdx;
                         tid <
                         n_q_points_1d * n_local_dofs_1d * n_local_dofs_1d;
                         tid += blockSize)
                      {
                        const int p = tid / (n_local_dofs_1d * n_local_dofs_1d);
                        const int j =
                          (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                          n_local_dofs_1d;
                        const int k = tid % n_local_dofs_1d;

                        number sum = 0.0;
                        for (unsigned int i = 0; i < n_local_dofs_1d; ++i)
                          {
                            sum +=
                              s_wsp0[k * n_local_dofs_1d * n_local_dofs_1d +
                                     j * n_local_dofs_1d + i] *
                              shape_values_scratch[i * n_q_points_1d + p];
                          }
                        s_wsp1[k * n_q_points_1d * n_local_dofs_1d +
                               j * n_q_points_1d + p] = sum;
                      }
                    team_member.team_barrier();

                    // step-3 : direction 1
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                         tid += blockSize)
                      {
                        const int i = tid / (n_q_points_1d * n_local_dofs_1d);
                        const int q =
                          (tid % (n_q_points_1d * n_local_dofs_1d)) /
                          n_local_dofs_1d;
                        const int k = tid % n_local_dofs_1d;

                        number sum = 0.0;
                        for (unsigned int j = 0; j < n_local_dofs_1d; j++)
                          {
                            sum += s_wsp1[k * n_q_points_1d * n_local_dofs_1d +
                                          j * n_q_points_1d + i] *
                                   shape_values_scratch[j * n_q_points_1d + q];
                          }

                        s_wsp0[k * n_q_points_1d * n_q_points_1d +
                               q * n_q_points_1d + i] = sum;
                      }
                    team_member.team_barrier();

                    // step-4 : direction 2
                    for (unsigned int tid = threadIdx;
                         tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                         tid += blockSize)
                      {
                        const int i = tid / (n_q_points_1d * n_q_points_1d);
                        const int j = (tid % (n_q_points_1d * n_q_points_1d)) /
                                      n_q_points_1d;
                        const int r = tid % n_q_points_1d;

                        number sum = 0.0;
                        for (unsigned int k = 0; k < n_local_dofs_1d; ++k)
                          {
                            sum += s_wsp0[k * n_q_points_1d * n_q_points_1d +
                                          j * n_q_points_1d + i] *
                                   shape_values_scratch[k * n_q_points_1d + r];
                          }
                        s_wsp1[r * n_q_points_1d * n_q_points_1d +
                               j * n_q_points_1d + i] = sum;
                      }
                    team_member.team_barrier();
                  }

                // Geometric vals
                number Grr, Grs, Grt, Gss, Gst, Gtt;
                number qr, qs, qt;

                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_q_points_1d);
                    const int q =
                      (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                    const int r = tid % n_q_points_1d;

                    qr = 0;
                    qs = 0;
                    qt = 0;

                    // step-5 : Load Geometric Factors, coalesced access
                    Grr = G_device[cell_index * 6 * n_q_points_total +
                                   0 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Grs = G_device[cell_index * 6 * n_q_points_total +
                                   1 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Grt = G_device[cell_index * 6 * n_q_points_total +
                                   2 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gss = G_device[cell_index * 6 * n_q_points_total +
                                   3 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gst = G_device[cell_index * 6 * n_q_points_total +
                                   4 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];
                    Gtt = G_device[cell_index * 6 * n_q_points_total +
                                   5 * n_q_points_total +
                                   p * n_q_points_1d * n_q_points_1d +
                                   q * n_q_points_1d + r];

                    // step-6 : Multiply by D
                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qr += s_wsp1[r * n_q_points_1d * n_q_points_1d +
                                     q * n_q_points_1d + n] *
                              co_shape_gradients_scratch[n * n_q_points_1d + p];
                      }

                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qs += s_wsp1[r * n_q_points_1d * n_q_points_1d +
                                     n * n_q_points_1d + p] *
                              co_shape_gradients_scratch[n * n_q_points_1d + q];
                      }

                    for (unsigned int n = 0; n < n_q_points_1d; n++)
                      {
                        qt += s_wsp1[n * n_q_points_1d * n_q_points_1d +
                                     q * n_q_points_1d + p] *
                              co_shape_gradients_scratch[n * n_q_points_1d + r];
                      }

                    // step-7 : Apply chain rule
                    rqr[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d +
                        p] = Grr * qr + Grs * qs + Grt * qt;
                    rqs[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d +
                        p] = Grs * qr + Gss * qs + Gst * qt;
                    rqt[r * n_q_points_1d * n_q_points_1d + q * n_q_points_1d +
                        p] = Grt * qr + Gst * qs + Gtt * qt;
                  }
                team_member.team_barrier();

                // step-8 : Compute out vector in GL nodes
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_q_points_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_q_points_1d);
                    const int q =
                      (tid % (n_q_points_1d * n_q_points_1d)) / n_q_points_1d;
                    const int r = tid % n_q_points_1d;

                    number sum = 0;
                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqr[r * n_q_points_1d * n_q_points_1d +
                                 q * n_q_points_1d + n] *
                             co_shape_gradients_scratch[p * n_q_points_1d + n];

                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqs[r * n_q_points_1d * n_q_points_1d +
                                 n * n_q_points_1d + p] *
                             co_shape_gradients_scratch[q * n_q_points_1d + n];

                    for (unsigned int n = 0; n < n_q_points_1d; ++n)
                      sum += rqt[n * n_q_points_1d * n_q_points_1d +
                                 q * n_q_points_1d + p] *
                             co_shape_gradients_scratch[r * n_q_points_1d + n];

                    s_wsp1[r * n_q_points_1d * n_q_points_1d +
                           q * n_q_points_1d + p] = sum;
                  }
                team_member.team_barrier();

                /*
                Interpolate to GLL nodes
                */

                // step-9 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_q_points_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points_1d * n_local_dofs_1d);
                    const int q = (tid % (n_q_points_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    number sum = 0.0;
                    for (unsigned int r = 0; r < n_q_points_1d; ++r)
                      {
                        sum += s_wsp1[r * n_q_points_1d * n_q_points_1d +
                                      q * n_q_points_1d + p] *
                               shape_values_scratch[k * n_local_dofs_1d + r];
                      }
                    s_wsp0[k * n_q_points_1d * n_local_dofs_1d +
                           q * n_q_points_1d + p] = sum;
                  }
                team_member.team_barrier();

                // step-10 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < n_q_points_1d * n_local_dofs_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int p = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    number sum = 0.0;
                    for (unsigned int q = 0; q < n_q_points_1d; q++)
                      {
                        sum += s_wsp0[k * n_q_points_1d * n_q_points_1d +
                                      q * n_q_points_1d + p] *
                               shape_values_scratch[j * n_q_points_1d + q];
                      }
                    s_wsp1[k * n_q_points_1d * n_local_dofs_1d +
                           j * n_q_points_1d + p] = sum;
                  }
                team_member.team_barrier();

                // step-11 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < n_local_dofs_1d * n_local_dofs_1d * n_local_dofs_1d;
                     tid += blockSize)
                  {
                    const int i = tid / (n_local_dofs_1d * n_local_dofs_1d);
                    const int j = (tid % (n_local_dofs_1d * n_local_dofs_1d)) /
                                  n_local_dofs_1d;
                    const int k = tid % n_local_dofs_1d;

                    number sum = 0.0;
                    for (unsigned int p = 0; p < n_q_points_1d; ++p)
                      {
                        sum += s_wsp1[k * n_q_points_1d * n_local_dofs_1d +
                                      j * n_q_points_1d + p] *
                               shape_values_scratch[i * n_q_points_1d + p];
                      }
                    s_wsp0[k * n_local_dofs_1d * n_local_dofs_1d +
                           j * n_local_dofs_1d + i] = sum;
                  }
                team_member.team_barrier();

                // step-12 : Copy wsp0 to out
                for (unsigned int tid = threadIdx; tid < n_local_dofs_total;
                     tid += blockSize)
                  {
                    const unsigned int dof_index = dof_indices(tid, cell_index);

                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        Kokkos::atomic_add(&out_device[dof_index], s_wsp0[tid]);
                      }
                  }
                team_member.team_barrier();

                cell_index += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

  } // namespace Parallel
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif