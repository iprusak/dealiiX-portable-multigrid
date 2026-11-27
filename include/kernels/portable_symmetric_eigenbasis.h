#ifndef portable_symmetric_eigenbasis_h
#define portable_symmetric_eigenbasis_h

#include <deal.II/base/tensor.h>

#include <Kokkos_Core.hpp>

#include <limits>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  // A = sum_a values[a] vectors[a] (x) vectors[a], values in descending
  // order, vectors[a] orthonormal
  template <int dim, typename Number>
  struct SymmetricEigenbasis
  {
    Tensor<1, dim, Number> values;
    Tensor<2, dim, Number> vectors;
  };



  // Eigenbasis of the symmetric part of A by cyclic Jacobi rotations
  // (Golub, Van Loan, Sec. 8.5): no special cases for repeated eigenvalues,
  // which give an arbitrary orthonormal basis of the eigenspace
  template <int dim, typename Number>
  DEAL_II_HOST_DEVICE SymmetricEigenbasis<dim, Number>
  symmetric_eigenbasis(const Tensor<2, dim, Number> &A)
  {
    constexpr unsigned int max_sweeps = 20;
    constexpr Number       eps        = std::numeric_limits<Number>::epsilon();

    Number a[dim][dim];
    Number v[dim][dim];
    Number norm_sq = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        {
          a[i][j] = Number(0.5) * (A[i][j] + A[j][i]);
          v[i][j] = (i == j) ? Number(1.) : Number(0.);
          norm_sq += a[i][j] * a[i][j];
        }

    // the Frobenius norm is invariant under the rotations
    for (unsigned int sweep = 0; sweep < max_sweeps; ++sweep)
      {
        Number off_sq = 0.;
        for (unsigned int p = 0; p < dim; ++p)
          for (unsigned int q = p + 1; q < dim; ++q)
            off_sq += a[p][q] * a[p][q];
        if (!(off_sq > eps * eps * norm_sq))
          break;

        for (unsigned int p = 0; p < dim; ++p)
          for (unsigned int q = p + 1; q < dim; ++q)
            {
              const Number apq = a[p][q];
              if (apq == Number(0.))
                continue;

              // rotation zeroing a[p][q]: t = tan(phi) of the smaller root,
              // sqrt(theta^2 + 1) without overflow
              const Number theta     = (a[q][q] - a[p][p]) / (Number(2.) * apq);
              const Number abs_theta = Kokkos::abs(theta);
              const Number root =
                (abs_theta > Number(1.)) ?
                  abs_theta * Kokkos::sqrt(Number(1.) + Number(1.) / (theta * theta)) :
                  Kokkos::sqrt(theta * theta + Number(1.));
              const Number t = (theta < Number(0.) ? Number(-1.) : Number(1.)) / (abs_theta + root);
              const Number c = Number(1.) / Kokkos::sqrt(t * t + Number(1.));
              const Number s = t * c;

              a[p][p] -= t * apq;
              a[q][q] += t * apq;
              a[p][q] = a[q][p] = Number(0.);
              for (unsigned int r = 0; r < dim; ++r)
                {
                  if (r != p && r != q)
                    {
                      const Number arp = a[r][p];
                      const Number arq = a[r][q];
                      a[r][p] = a[p][r] = c * arp - s * arq;
                      a[r][q] = a[q][r] = s * arp + c * arq;
                    }
                  const Number vrp = v[r][p];
                  const Number vrq = v[r][q];
                  v[r][p]          = c * vrp - s * vrq;
                  v[r][q]          = s * vrp + c * vrq;
                }
            }
      }

    // eigenvectors are the columns of v; sort descending by swaps with
    // compile-time indices, which keeps a and v in registers
    for (unsigned int i = 0; i + 1 < dim; ++i)
      for (unsigned int j = 0; j + 1 < dim - i; ++j)
        if (a[j][j] < a[j + 1][j + 1])
          {
            const Number tmp = a[j][j];
            a[j][j]          = a[j + 1][j + 1];
            a[j + 1][j + 1]  = tmp;
            for (unsigned int d = 0; d < dim; ++d)
              {
                const Number vd = v[d][j];
                v[d][j]         = v[d][j + 1];
                v[d][j + 1]     = vd;
              }
          }

    SymmetricEigenbasis<dim, Number> result;
    for (unsigned int i = 0; i < dim; ++i)
      {
        result.values[i] = a[i][i];
        for (unsigned int d = 0; d < dim; ++d)
          result.vectors[i][d] = v[d][i];
      }
    return result;
  }
} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
