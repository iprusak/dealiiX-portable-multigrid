#ifndef laplace_dg_kokkos_kernels_h
#define laplace_dg_kokkos_kernels_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace BK3
{
  namespace DG
  {

    template <typename Number>
    using DeviceView = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>;

    using DoFIndicesView = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename Number>
    void
    compute_cell(
      const DeviceView<Number> d_shape_values,
      const DeviceView<Number> d_co_shape_gradients,
      const DeviceView<Number> d_G,
      const DeviceView<Number> d_in,
      DeviceView<Number>       d_out,
      const Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>
                                                                   interpolate_quad_to_boundary,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_values_at_quads,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_normal_derivatives_at_quads,
      const DoFIndicesView                                         dof_indices,
      const unsigned int                                           n_cells,
      const unsigned int n_cells_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total = Utilities::pow(nq, dim);

      constexpr int nm_total = Utilities::pow(nm, dim);

      constexpr int nq_total_per_face = Utilities::pow(nq, dim - 1);

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 1 + dim; // values and gradients in each direction

      const int nelmt = n_cells;

      const int nelmtPerBatch = std::max(1,
                                         ((n_cells_per_batch == numbers::invalid_unsigned_int) ?
                                            (shmemPerBlock / (n_scratch_arrays * nq_total) /
                                             static_cast<int>(sizeof(Number))) :
                                            static_cast<int>(n_cells_per_batch)));

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));


      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 1) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));


      {
        const int ssize = nm * nq + // shape values
                          nq * nq + // co-shape gradients
                          4 * nq +  // interpolate quad to boundary
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total; // working scratch arrays: scratch_values, scratch_grads_0,
                                      // scratch_grads_1, scratch_grads_2


        const int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];
            Number r_q[nq];
            Number r_r[nq];

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + nq * nm;

            Number *s_quad_to_boundary_value_0 = s_co_shape_gradients + nq * nq;
            Number *s_quad_to_boundary_value_1 = s_quad_to_boundary_value_0 + nq;
            Number *s_quad_to_boundary_grad_0  = s_quad_to_boundary_value_1 + nq;
            Number *s_quad_to_boundary_grad_1  = s_quad_to_boundary_grad_0 + nq;

            Number *scratch_values  = s_quad_to_boundary_grad_1 + nq;
            Number *scratch_grads_0 = scratch_values + nelmtPerBatch * nq_total;
            Number *scratch_grads_1 = scratch_grads_0 + nelmtPerBatch * nq_total;

            Number *scratch_grads_2;
            if (dim == 3)
              scratch_grads_2 = scratch_grads_1 + nelmtPerBatch * nq_total;


            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = d_shape_values[tid];
              }

            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
              }
            team_member.team_barrier();

            for (int tid = threadIdx; tid < nq; tid += blockSize)
              {
                s_quad_to_boundary_value_0[tid] = interpolate_quad_to_boundary(0, tid, 0);
                s_quad_to_boundary_value_1[tid] = interpolate_quad_to_boundary(0, tid, 1);

                s_quad_to_boundary_grad_0[tid] = interpolate_quad_to_boundary(1, tid, 0);
                s_quad_to_boundary_grad_1[tid] = interpolate_quad_to_boundary(1, tid, 1);
              }
            team_member.team_barrier();


            /*
            Interpolate to GL nodes
            */

            // element batch iteration
            int eb = team_member.league_rank();
            while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                        (nelmt - eb * nelmtPerBatch) :
                                        nelmtPerBatch;

                // step-1 : Copy from in to the scratch values
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                    {
                      const int e         = tid / nm_total;
                      const int local_idx = tid % nm_total;

                      const int global_cell_index = eb * nelmtPerBatch + e;

                      // Fetch the global DoF index
                      const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

                      if (dof_index == numbers::invalid_unsigned_int)
                        scratch_values[tid] = 0;
                      else
                        scratch_values[tid] = d_in[dof_index];
                    }
                  team_member.team_barrier();
                }

                // interpolate dof values to quadrature points in each direction
                {
                  // direction 0
                  {
                    constexpr int co_dimension_size = Utilities::pow(nm, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int j = tid % co_dimension_size;

                            for (int i = 0; i < nm; ++i)
                              {
                                r_p[i] = scratch_values[e * nm * nm + j * nm + i];
                              }

                            for (int p = 0; p < nq; ++p)
                              {
                                Number tmp = 0.0;

                                for (int i = 0; i < nm; ++i)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[i];
                                  }

                                scratch_grads_0[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            const int k = (tid % co_dimension_size) / nm;
                            const int j = tid % nm;

                            for (int i = 0; i < nm; ++i)
                              {
                                r_p[i] =
                                  scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + i];
                              }

                            for (int p = 0; p < nq; ++p)
                              {
                                Number tmp = 0.0;

                                for (int i = 0; i < nm; ++i)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[i];
                                  }

                                scratch_grads_0[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 1
                  {
                    constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int j = 0; j < nm; ++j)
                              {
                                r_p[j] = scratch_grads_0[e * nq * nm + j * nq + p];
                              }

                            for (int q = 0; q < nq; ++q)
                              {
                                Number tmp = 0.0;

                                for (int j = 0; j < nm; ++j)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_p[j];
                                  }

                                scratch_values[e * nq * nq + q * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            const int k = (tid % co_dimension_size) / nq;
                            const int p = tid % nq;

                            for (int j = 0; j < nm; ++j)
                              {
                                r_q[j] =
                                  scratch_grads_0[e * nq * nm * nm + k * nq * nm + j * nq + p];
                              }

                            for (int q = 0; q < nq; ++q)
                              {
                                Number tmp = 0.0;

                                for (int j = 0; j < nm; ++j)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[j];
                                  }

                                scratch_grads_1[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 2
                  if (dim == 3)
                    {
                      constexpr int co_dimension_size = nq * nq;

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int q = (tid % (nq * nq)) / nq;
                          const int p = tid % nq;

                          for (int k = 0; k < nm; ++k)
                            {
                              r_r[k] = scratch_grads_1[e * nq * nq * nm + k * nq * nq + q * nq + p];
                            }
                          for (int r = 0; r < nq; ++r)
                            {
                              Number tmp = 0.0;

                              for (int k = 0; k < nm; ++k)
                                {
                                  tmp += s_shape_values[k * nq + r] * r_r[k];
                                }

                              scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }
                }

                // interpolate values to faces
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nq_total_per_face;
                       tid += blockSize)
                    {
                      const int e = tid / nq_total_per_face;
                      if (dim == 2)
                        {
                          const int m = tid % nq_total_per_face;

                          const int offset_0 = e * nq * nq + m * nq;
                          const int offset_1 = e * nq * nq + m;

                          for (int n = 0; n < nq; ++n)
                            {
                              r_p[n] = scratch_values[offset_0 + n];      // x direction
                              r_q[n] = scratch_values[offset_1 + n * nq]; // y direction
                            }

                          Number v_x[2], d_x[2], v_y[2], d_y[2];
                          for (int i = 0; i < 2; ++i)
                            {
                              v_x[i] = 0;
                              d_x[i] = 0;

                              v_y[i] = 0;
                              d_y[i] = 0;
                            }

                          for (int n = 0; n < nq; ++n)
                            {
                              v_x[0] += s_quad_to_boundary_value_0[n] * r_p[n];
                              d_x[0] += s_quad_to_boundary_grad_0[n] * r_p[n];

                              v_x[1] += s_quad_to_boundary_value_1[n] * r_p[n];
                              d_x[1] += s_quad_to_boundary_grad_1[n] * r_p[n];

                              v_y[0] += s_quad_to_boundary_value_0[n] * r_q[n];
                              d_y[0] += s_quad_to_boundary_grad_0[n] * r_q[n];

                              v_y[1] += s_quad_to_boundary_value_1[n] * r_q[n];
                              d_y[1] += s_quad_to_boundary_grad_1[n] * r_q[n];
                            }


                          const int global_cell_id = eb * nelmtPerBatch + e;

                          const int local_q_id = m;

                          face_values_at_quads(local_q_id, 0, global_cell_id)             = v_x[0];
                          face_normal_derivatives_at_quads(local_q_id, 0, global_cell_id) = d_x[0];

                          face_values_at_quads(local_q_id, 1, global_cell_id)             = v_x[1];
                          face_normal_derivatives_at_quads(local_q_id, 1, global_cell_id) = d_x[1];

                          face_values_at_quads(local_q_id, 2, global_cell_id)             = v_y[0];
                          face_normal_derivatives_at_quads(local_q_id, 2, global_cell_id) = d_y[0];

                          face_values_at_quads(local_q_id, 3, global_cell_id)             = v_y[1];
                          face_normal_derivatives_at_quads(local_q_id, 3, global_cell_id) = d_y[1];
                        }
                      else if (dim == 3)
                        {
                          const int m2 = (tid % nq_total_per_face) / nq;
                          const int m1 = tid % nq;

                          const int offset_0 = e * nq * nq * nq + m2 * nq * nq + m1 * nq;
                          const int offset_1 = e * nq * nq * nq + m2 * nq * nq + m1;
                          const int offset_2 = e * nq * nq * nq + m2 * nq + m1;

                          for (int n = 0; n < nq; ++n)
                            {
                              r_p[n] = scratch_values[offset_0 + n];           // x direction
                              r_q[n] = scratch_values[offset_1 + n * nq];      // y direction
                              r_r[n] = scratch_values[offset_2 + n * nq * nq]; // z direction
                            }

                          Number v_x[2], d_x[2], v_y[2], d_y[2], v_z[2], d_z[2];
                          for (int i = 0; i < 2; ++i)
                            {
                              v_x[i] = 0;
                              d_x[i] = 0;

                              v_y[i] = 0;
                              d_y[i] = 0;

                              v_z[i] = 0;
                              d_z[i] = 0;
                            }

                          for (int n = 0; n < nq; ++n)
                            {
                              v_x[0] += s_quad_to_boundary_value_0[n] * r_p[n];
                              d_x[0] += s_quad_to_boundary_grad_0[n] * r_p[n];

                              v_x[1] += s_quad_to_boundary_value_1[n] * r_p[n];
                              d_x[1] += s_quad_to_boundary_grad_1[n] * r_p[n];

                              v_y[0] += s_quad_to_boundary_value_0[n] * r_q[n];
                              d_y[0] += s_quad_to_boundary_grad_0[n] * r_q[n];

                              v_y[1] += s_quad_to_boundary_value_1[n] * r_q[n];
                              d_y[1] += s_quad_to_boundary_grad_1[n] * r_q[n];

                              v_z[0] += s_quad_to_boundary_value_0[n] * r_r[n];
                              d_z[0] += s_quad_to_boundary_grad_0[n] * r_r[n];

                              v_z[1] += s_quad_to_boundary_value_1[n] * r_r[n];
                              d_z[1] += s_quad_to_boundary_grad_1[n] * r_r[n];
                            }



                          const int global_cell_id = eb * nelmtPerBatch + e;
                          const int local_q_id     = m2 * nq + m1;

                          face_values_at_quads(local_q_id, 0, global_cell_id)             = v_x[0];
                          face_normal_derivatives_at_quads(local_q_id, 0, global_cell_id) = d_x[0];

                          face_values_at_quads(local_q_id, 1, global_cell_id)             = v_x[1];
                          face_normal_derivatives_at_quads(local_q_id, 1, global_cell_id) = d_x[1];

                          face_values_at_quads(local_q_id, 2, global_cell_id)             = v_y[0];
                          face_normal_derivatives_at_quads(local_q_id, 2, global_cell_id) = d_y[0];

                          face_values_at_quads(local_q_id, 3, global_cell_id)             = v_y[1];
                          face_normal_derivatives_at_quads(local_q_id, 3, global_cell_id) = d_y[1];

                          face_values_at_quads(local_q_id, 4, global_cell_id)             = v_z[0];
                          face_normal_derivatives_at_quads(local_q_id, 4, global_cell_id) = d_z[0];

                          face_values_at_quads(local_q_id, 5, global_cell_id)             = v_z[1];
                          face_normal_derivatives_at_quads(local_q_id, 5, global_cell_id) = d_z[1];
                        }
                    }
                }
                team_member.team_barrier();

                // apply geometric factors and compute stiffness contributions at quadrature points
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                            }

                          Number Grr, Grs, Gss;
                          Number qr, qs;

                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;

                              // Load Geometric Factors, coalesced access
                              Grr = d_G[eb * nelmtPerBatch * 3 * nq_total + e * 3 * nq_total +
                                        0 * nq_total + q * nq + p];
                              Grs = d_G[eb * nelmtPerBatch * 3 * nq_total + e * 3 * nq_total +
                                        1 * nq_total + q * nq + p];
                              Gss = d_G[eb * nelmtPerBatch * 3 * nq_total + e * 3 * nq_total +
                                        2 * nq_total + q * nq + p];

                              // Multiply by D
                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += r_q[n] * scratch_values[e * nq * nq + n * nq + p];
                                }

                              // Apply chain rule
                              scratch_grads_0[e * nq * nq + q * nq + p] = Grr * qr + Grs * qs;
                              scratch_grads_1[e * nq * nq + q * nq + p] = Grs * qr + Gss * qs;
                            }
                        }
                      else if (dim == 3)
                        {
                          int r = (tid % co_dimension_size) / nq;
                          int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[n * nq + q];
                              r_r[n] = s_co_shape_gradients[n * nq + r];
                            }

                          Number Grr, Grs, Grt, Gss, Gst, Gtt;
                          Number qr, qs, qt;

                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;
                              qt = 0;

                              // Load Geometric Factors, coalesced access
                              Grr = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        0 * nq_total + r * nq * nq + q * nq + p];

                              Grs = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        1 * nq_total + r * nq * nq + q * nq + p];

                              Grt = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        2 * nq_total + r * nq * nq + q * nq + p];

                              Gss = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        3 * nq_total + r * nq * nq + q * nq + p];

                              Gst = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        4 * nq_total + r * nq * nq + q * nq + p];

                              Gtt = d_G[eb * nelmtPerBatch * 6 * nq_total + e * 6 * nq_total +
                                        5 * nq_total + r * nq * nq + q * nq + p];

                              // Multiply by D
                              for (int n = 0; n < nq; n++)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += r_q[n] *
                                        scratch_values[e * nq * nq * nq + r * nq * nq + n * nq + p];
                                  qt += r_r[n] *
                                        scratch_values[e * nq * nq * nq + n * nq * nq + q * nq + p];
                                }

                              // Apply chain rule
                              scratch_grads_0[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grr * qr + Grs * qs + Grt * qt;

                              scratch_grads_1[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grs * qr + Gss * qs + Gst * qt;

                              scratch_grads_2[e * nq * nq * nq + r * nq * nq + q * nq + p] =
                                Grt * qr + Gst * qs + Gtt * qt;
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                // apply D^T
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      if (dim == 2)
                        {
                          const int q = tid % co_dimension_size;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_grads_0[e * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0;

                              for (int n = 0; n < nq; ++n)
                                tmp += s_co_shape_gradients[p * nq + n] * r_p[n];

                              for (int n = 0; n < nq; ++n)
                                tmp += scratch_grads_1[e * nq * nq + q * nq + n] * r_q[n];

                              scratch_values[e * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      else if (dim == 3)
                        {
                          const int r = tid % (nq * nq) / nq;
                          const int q = tid % nq;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_grads_0[e * nq * nq * nq + r * nq * nq + q * nq + n];
                              r_q[n] = s_co_shape_gradients[q * nq + n];
                              r_r[n] = s_co_shape_gradients[r * nq + n];
                            }

                          for (int p = 0; p < nq; ++p)
                            {
                              Number tmp = 0;
                              for (int n = 0; n < nq; ++n)
                                tmp += r_p[n] * s_co_shape_gradients[p * nq + n];

                              for (int n = 0; n < nq; ++n)
                                tmp +=
                                  scratch_grads_1[e * nq * nq * nq + r * nq * nq + n * nq + p] *
                                  r_q[n];

                              for (int n = 0; n < nq; ++n)
                                tmp +=
                                  scratch_grads_2[e * nq * nq * nq + n * nq * nq + q * nq + p] *
                                  r_r[n];

                              scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }
                }

                /*
                Interpolate to GLL nodes
                */
                {
                  // direction 2
                  if (dim == 3)
                    {
                      constexpr int co_dimension_size = Utilities::pow(nq, dim - 1);

                      for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                           tid += blockSize)
                        {
                          const int e = tid / co_dimension_size;

                          const int q = (tid % co_dimension_size) / nq;
                          const int p = tid % nq;

                          for (int r = 0; r < nq; ++r)
                            {
                              r_r[r] = scratch_values[e * nq * nq * nq + r * nq * nq + q * nq + p];
                            }

                          for (int k = 0; k < nm; ++k)
                            {
                              Number tmp = 0.0;

                              for (int r = 0; r < nq; ++r)
                                {
                                  tmp += scratch_grads_0[k * nq + r] * r_r[r];
                                }

                              scratch_grads_0[e * nq * nq * nm + k * nq * nq + q * nq + p] = tmp;
                            }
                        }
                      team_member.team_barrier();
                    }

                  //  direction 1
                  {
                    constexpr int co_dimension_size = nq * Utilities::pow(nm, dim - 2);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int p = tid % co_dimension_size;

                            for (int q = 0; q < nq; ++q)
                              {
                                r_q[q] = scratch_values[e * nq * nq + q * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int q = 0; q < nq; ++q)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[q];
                                  }
                                scratch_grads_1[e * nq * nm + j * nq + p] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nq;
                            int p = tid % nq;

                            for (int j = 0; j < nq; ++j)
                              {
                                r_q[j] =
                                  scratch_grads_0[e * nq * nq * nm + k * nq * nq + j * nq + p];
                              }

                            for (int j = 0; j < nm; ++j)
                              {
                                Number tmp = 0.0;

                                for (int q = 0; q < nq; ++q)
                                  {
                                    tmp += s_shape_values[j * nq + q] * r_q[q];
                                  }
                                scratch_grads_1[e * nq * nm * nm + k * nq * nm + j * nq + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }

                  // direction 0
                  {
                    const int co_dimension_size = Utilities::pow(nm, dim - 1);

                    for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                         tid += blockSize)
                      {
                        const int e = tid / co_dimension_size;

                        if (dim == 2)
                          {
                            const int j = tid % co_dimension_size;
                            for (int p = 0; p < nq; ++p)
                              {
                                r_p[p] = scratch_grads_1[e * nq * nq + j * nq + p];
                              }

                            for (int i = 0; i < nm; ++i)
                              {
                                Number tmp = 0.0;
                                for (int p = 0; p < nq; ++p)
                                  {
                                    tmp += s_shape_values[i * nq + p] * r_p[p];
                                  }
                                scratch_values[e * nm * nm + j * nm + i] = tmp;
                              }
                          }
                        else if (dim == 3)
                          {
                            int k = (tid % co_dimension_size) / nm;
                            int j = tid % nm;

                            for (int i = 0; i < nq; ++i)
                              {
                                r_p[i] =
                                  scratch_grads_1[e * nq * nm * nm + k * nq * nm + j * nq + i];
                              }

                            for (int p = 0; p < nm; ++p)
                              {
                                Number tmp = 0.0;
                                for (int i = 0; i < nq; ++i)
                                  {
                                    tmp += s_shape_values[p * nq + i] * r_p[i];
                                  }
                                scratch_values[e * nm * nm * nm + k * nm * nm + j * nm + p] = tmp;
                              }
                          }
                      }
                    team_member.team_barrier();
                  }
                }

                // step-12 : Copy wsp0 (result) back to global out vector
                for (int tid = threadIdx; tid < c_nelmtPerBatch * nm_total; tid += blockSize)
                  {
                    const int e = tid / nm_total;

                    const int local_idx = tid % nm_total;

                    const int global_cell_index = eb * nelmtPerBatch + e;

                    // Find where this node lives in the global 'd_out'
                    // vector
                    const unsigned int dof_index = dof_indices(local_idx, global_cell_index);

                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        // CRITICAL: Use atomic_add because elements share
                        // nodes!
                        Kokkos::atomic_add(&d_out[dof_index], scratch_values[tid]);
                      }
                  }

                team_member.team_barrier();

                eb += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

    template <int dim, int nm, int nq, typename Number>
    void
    compute_inner_faces(
      const DeviceView<Number>                                            d_shape_values,
      const DeviceView<Number>                                            d_co_shape_gradients,
      const Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jacobian_times_normal,
      const DeviceView<Number>                                            d_in,
      DeviceView<Number>                                                  d_out,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>        face_values_at_quads,
      Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> face_normal_derivatives_at_quads,
      const DoFIndicesView                                         dof_indices,
      Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space> face_info,
      const unsigned int                                                  n_faces,
      const unsigned int n_faces_per_batch = numbers::invalid_unsigned_int,
      const unsigned int n_blocks          = numbers::invalid_unsigned_int,
      const unsigned int threads_per_block = numbers::invalid_unsigned_int)
    {
      constexpr int nq_total_per_face = Utilities::pow(nq, dim - 1);

      constexpr int nm_total_per_face = Utilities::pow(nm, dim - 1);

      // finding the batch size
      constexpr int shmemPerBlock = 10800; // total shared memory used per block (KB)

      constexpr int n_scratch_arrays = 4; // values and normal derivatives on both sides of the face

      const int nelmt = n_faces;

      const int nelmtPerBatch =
        std::max(1,
                 ((n_faces_per_batch == numbers::invalid_unsigned_int) ?
                    (shmemPerBlock / (n_scratch_arrays * nq_total_per_face) /
                     static_cast<int>(sizeof(Number))) :
                    static_cast<int>(n_faces_per_batch)));

      const int numBlocks = std::max(1,
                                     ((n_blocks == numbers::invalid_unsigned_int) ?
                                        ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2) :
                                        static_cast<int>(n_blocks)));


      const int threadsPerBlock = std::max(1,
                                           ((threads_per_block == numbers::invalid_unsigned_int) ?
                                              (Utilities::pow(nq, dim - 2) * nelmtPerBatch) :
                                              static_cast<int>(threads_per_block)));


      // std::cout << "compute_inner_faces: n_faces = " << n_faces << ", nelmt = " << nelmt
      //           << ", nelmtPerBatch = " << nelmtPerBatch << ", numBlocks = " << numBlocks
      //           << ", threadsPerBlock = " << threadsPerBlock << std::endl;
      {
        const int ssize = nm * nq + // shape values
                          nq * nq + // co-shape gradients
                          n_scratch_arrays * nelmtPerBatch *
                            nq_total_per_face; // working scratch arrays: scratch_values,
                                               // scratch_grads_0, scratch_grads_1


        const int shmem_size = ssize * sizeof(Number);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<>                      policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            Number r_p[nq];
            Number r_q[nq];
            Number r_r[nq];


            // Look up tangent directions based on normal_direction (0, 1, or 2)
            constexpr int lookup_tangents_3d[3][2] = {
              {1, 2}, // normal == 0
              {0, 2}, // normal == 1
              {0, 1}  // normal == 2 (default fallback)
            };

            Number *scratch = (Number *)team_member.team_shmem().get_shmem(shmem_size);

            Number *s_shape_values       = scratch;
            Number *s_co_shape_gradients = s_shape_values + nq * nm;

            Number *scratch_values_minus = s_co_shape_gradients + nq * nq;
            Number *scratch_values_plus  = scratch_values_minus + nelmtPerBatch * nq_total_per_face;

            Number *scratch_normal_derivative_minus =
              scratch_values_plus + nelmtPerBatch * nq_total_per_face;
            Number *scratch_normal_derivative_plus =
              scratch_normal_derivative_minus + nelmtPerBatch * nq_total_per_face;

            const int threadIdx = team_member.team_rank();
            const int blockSize = team_member.team_size();

            // copy to shared memory
            for (int tid = threadIdx; tid < nm * nq; tid += blockSize)
              {
                s_shape_values[tid] = d_shape_values[tid];
              }

            for (int tid = threadIdx; tid < nq * nq; tid += blockSize)
              {
                s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
              }
            team_member.team_barrier();

            // element batch iteration
            int eb = team_member.league_rank();
            while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
              {
                // current nelmtPerBatch (edge case, last batch size can be
                // less)
                const int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ?
                                              (nelmt - eb * nelmtPerBatch) :
                                              nelmtPerBatch;

                // std::cout << "compute_inner_faces: eb = " << eb
                //           << ", c_nelmtPerBatch = " << c_nelmtPerBatch << std::endl;

                // step-1 : Copy quad values and normal derivatives
                {
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * nq_total_per_face;
                       tid += blockSize)
                    {
                      const int e = tid / nq_total_per_face;
                      const int q = tid % nq_total_per_face;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int cell_minus = face_info(e_global, 0);
                      const int cell_plus  = face_info(e_global, 1);
                      const int f_minus    = face_info(e_global, 2);
                      const int f_plus     = face_info(e_global, 3);

                      const int normal_direction = f_minus / 2;


                      scratch_values_minus[tid] = face_values_at_quads(q, f_minus, cell_minus);
                      scratch_values_plus[tid]  = face_values_at_quads(q, f_plus, cell_plus);

                      Number jac_per_n[2];
                      jac_per_n[0] =
                        jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                normal_direction * nq_total_per_face + q,
                                              0);
                      jac_per_n[1] =
                        jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                normal_direction * nq_total_per_face + q,
                                              1);

                      scratch_normal_derivative_minus[e * nq_total_per_face + q] =
                        face_normal_derivatives_at_quads(q, f_minus, cell_minus) * jac_per_n[0];

                      scratch_normal_derivative_plus[e * nq_total_per_face + q] =
                        face_normal_derivatives_at_quads(q, f_plus, cell_plus) * jac_per_n[1];

                      // std::cout << "face " << e_global << ": " << jac_per_n[0] << ":"
                      //           << face_normal_derivatives_at_quads(q, f_minus, cell_minus) << " | "
                      //           << jac_per_n[1] << " : "
                      //           << face_normal_derivatives_at_quads(q, f_plus, cell_plus)
                      //           << std::endl;
                    }
                  team_member.team_barrier();
                }



                //  for (int i = 0; i < nq_total_per_face; ++i)
                //   {
                //     std::cout << scratch_normal_derivative_minus[i] << " | "
                //               << scratch_normal_derivative_plus[i] << std::endl;
                //   }
                // step-2 : Evaluate tangential derivatives
                {
                  constexpr int co_dimension_size = Utilities::pow(nq, dim - 2);
                  for (int tid = threadIdx; tid < c_nelmtPerBatch * co_dimension_size;
                       tid += blockSize)
                    {
                      const int e = tid / co_dimension_size;

                      const int e_global = eb * nelmtPerBatch + e;

                      const int f_minus = face_info(e_global, 2);

                      if (dim == 2)
                        {
                          const int normal_direction = f_minus / 2;

                          const int tangent_direction = 1 - normal_direction;

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values_minus[e * nq + n];
                              r_q[n] = scratch_values_plus[e * nq + n];
                            }

                          Number qr, qs;
                          Number jac_per_n[2];
                          for (int p = 0; p < nq; ++p)
                            {
                              qr = 0;
                              qs = 0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                                  qs += s_co_shape_gradients[n * nq + p] * r_q[n];
                                }

                              jac_per_n[0] =
                                jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                        tangent_direction * nq_total_per_face + p,
                                                      0);
                              jac_per_n[1] =
                                jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                        tangent_direction * nq_total_per_face + p,
                                                      1);

                              scratch_normal_derivative_minus[e * nq_total_per_face + p] +=
                                qr * jac_per_n[0];

                              scratch_normal_derivative_plus[e * nq_total_per_face + p] +=
                                qs * jac_per_n[1];
                            }
                        }
                      else if (dim == 3)
                        {
                          const int p = tid % co_dimension_size;

                          const int normal_direction = f_minus / 2;

                          const int tangent_direction_0 = lookup_tangents_3d[normal_direction][0];
                          const int tangent_direction_1 = lookup_tangents_3d[normal_direction][1];

                          // copy to register
                          for (int n = 0; n < nq; n++)
                            {
                              r_p[n] = scratch_values_minus[e * nq * nq + n * nq + p];
                              r_q[n] = scratch_values_plus[e * nq * nq + n * nq + p];
                              r_r[n] = s_co_shape_gradients[n * nq + p];
                            }

                          Number qr, qs, qt, qv;
                          Number jac_per_n[2][2];

                          for (int q = 0; q < nq; ++q)
                            {
                              qr = 0;
                              qs = 0;
                              qt = 0;
                              qv = 0;

                              for (int n = 0; n < nq; ++n)
                                {
                                  qr += s_co_shape_gradients[n * nq + q] * r_p[n];
                                  qs += s_co_shape_gradients[n * nq + q] * r_q[n];

                                  qt += scratch_values_minus[e * nq * nq + q * nq + n] * r_r[n];
                                  qv += scratch_values_plus[e * nq * nq + q * nq + n] * r_r[n];
                                }

                              jac_per_n[0][0] =
                                jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                        tangent_direction_0 * nq_total_per_face + q,
                                                      0);
                              jac_per_n[0][1] =
                                jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                        tangent_direction_0 * nq_total_per_face + q,
                                                      1);
                              jac_per_n[1][0] =
                                jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                        tangent_direction_1 * nq_total_per_face + q,
                                                      0);
                              jac_per_n[1][1] =
                                jacobian_times_normal(e_global * dim * nq_total_per_face +
                                                        tangent_direction_1 * nq_total_per_face + q,
                                                      1);

                              scratch_normal_derivative_minus[e * nq_total_per_face + q * nq + p] +=
                                qr * jac_per_n[0][0] + qt * jac_per_n[1][0];

                              scratch_normal_derivative_plus[e * nq_total_per_face + q * nq + p] +=
                                qs * jac_per_n[0][1] + qv * jac_per_n[1][1];
                            }
                        }
                    }
                  team_member.team_barrier();
                }

                for (int i = 0; i < nq_total_per_face; ++i)
                  {
                    std::cout << scratch_values_minus[i] - scratch_values_plus[i] << std::endl;
                  }


                eb += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }

  } // namespace DG
} // namespace BK3

DEAL_II_NAMESPACE_CLOSE

#endif