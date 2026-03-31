#ifndef portable_subdomain_laplace_operator_base_h
#define portable_subdomain_laplace_operator_base_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <memory>

#include "domain_decomposition/subdomain_dof_handler.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  template <int dim, typename number>
  class SubdomainLaplaceOperatorBase : public EnableObserverPointer
  {
  public:

  virtual void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const = 0;

    virtual void
    vmult_dummy(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                &src,
      const bool ghost_exchange_on,
      const bool computation_on) const = 0;

    virtual void
    vmult_interface_cell_range(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const = 0;

    virtual void
    vmult_neumann(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const = 0;

    virtual void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const = 0;

    virtual void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec)
      const = 0;

    virtual void
    compute_diagonal() = 0;

    virtual std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const = 0;

    virtual std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse_neumann() const = 0;

    virtual types::global_dof_index
    m() const = 0;

    virtual types::global_dof_index
    n() const = 0;

    virtual number
    el(const types::global_dof_index row,
       const types::global_dof_index col) const = 0;

    virtual const MatrixFree<dim, number> &
    get_matrix_free() const = 0;

    virtual const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const = 0;

    virtual const Kokkos::View<const unsigned int *,
                               MemorySpace::Default::kokkos_space>
    get_interface_dof_indices_subdomain() const = 0;


    virtual const Kokkos::View<const unsigned int *,
                               MemorySpace::Default::kokkos_space>
    get_physical_boundary_dof_indices_subdomain() const = 0;

    virtual const SubdomainDoFHandler<dim> &
    get_subdomain_dof_handler() const = 0;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif