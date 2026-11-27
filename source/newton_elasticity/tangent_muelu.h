#ifndef newton_elasticity_tangent_muelu_h
#define newton_elasticity_tangent_muelu_h

// MueLu (Tpetra) AMG for the tangent on the degree 1 level of the polynomial
// multigrid: the tangent at u is assembled on the host into a Tpetra matrix,
// smoothed aggregation with Chebyshev smoothing and a direct coarse solve,
// near null space = rigid body modes of the reference configuration. Needs a
// deal.II with Trilinos Tpetra + MueLu.

#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA_MUELU

#  include <deal.II/base/quadrature_lib.h>
#  include <deal.II/base/timer.h>

#  include <deal.II/dofs/dof_handler.h>
#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/fe_values.h>
#  include <deal.II/fe/mapping_q.h>

#  include <deal.II/lac/affine_constraints.h>
#  include <deal.II/lac/dynamic_sparsity_pattern.h>
#  include <deal.II/lac/full_matrix.h>
#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/lac/read_write_vector.h>
#  include <deal.II/lac/sparsity_tools.h>
#  include <deal.II/lac/trilinos_tpetra_precondition.h>
#  include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#  include <deal.II/lac/trilinos_tpetra_vector.h>

#  include <Teuchos_ParameterList.hpp>

DEAL_II_NAMESPACE_OPEN

template <int dim, typename Number, typename Material>
class TangentMueLu
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

  // constraints: those of the level and the prescribed DoFs, all homogeneous
  TangentMueLu(const DoFHandler<dim>           &dof_handler,
               const AffineConstraints<Number> &constraints,
               const Material                  &material);

  // assemble the tangent at u and set up the hierarchy
  void
  rebuild(const VectorType &u);

  // one V-cycle on the free DoFs, identity on the constrained ones (as the
  // matrix-free operator)
  void
  vmult(VectorType &dst, const VectorType &src) const;

  double t_assemble = 0., t_setup = 0.;

private:
  using MemorySpaceType = MemorySpace::Default;
  using MatrixType      = LinearAlgebra::TpetraWrappers::SparseMatrix<Number, MemorySpaceType>;
  using TpetraVector    = LinearAlgebra::TpetraWrappers::Vector<Number, MemorySpaceType>;
  using AMGType = LinearAlgebra::TpetraWrappers::PreconditionAMGMueLu<Number, MemorySpaceType>;
  using MultiVector =
    LinearAlgebra::TpetraWrappers::TpetraTypes::MultiVectorType<Number, MemorySpaceType>;

  const DoFHandler<dim>           &dof_handler;
  const AffineConstraints<Number> &constraints;
  const Material                   material;
  const MappingQ<dim>              mapping;
  const QGauss<dim>                quadrature;
  const MPI_Comm                   comm;
  const IndexSet                   locally_owned_dofs;
  const IndexSet                   locally_relevant_dofs;

  // rigid body modes on the locally owned DoFs
  std::vector<std::vector<double>> modes;

  MatrixType           matrix;
  AMGType              amg;
  mutable TpetraVector src_tpetra, dst_tpetra;

  // 1 on the free DoFs, 0 on the constrained ones
  VectorType         free_mask;
  mutable VectorType masked_src;
};



template <int dim, typename Number, typename Material>
TangentMueLu<dim, Number, Material>::TangentMueLu(const DoFHandler<dim>           &dof_handler,
                                                  const AffineConstraints<Number> &constraints,
                                                  const Material                  &material)
  : dof_handler(dof_handler)
  , constraints(constraints)
  , material(material)
  , mapping(dof_handler.get_fe().degree)
  , quadrature(dof_handler.get_fe().degree + 1)
  , comm(dof_handler.get_mpi_communicator())
  , locally_owned_dofs(dof_handler.locally_owned_dofs())
  , locally_relevant_dofs(DoFTools::extract_locally_relevant_dofs(dof_handler))
{
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, comm, locally_relevant_dofs);
  matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, comm);

  src_tpetra.reinit(locally_owned_dofs, comm);
  dst_tpetra.reinit(locally_owned_dofs, comm);

  free_mask.reinit(locally_owned_dofs, comm);
  masked_src.reinit(locally_owned_dofs, comm);
  {
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = constraints.is_constrained(i) ? 0. : 1.;
    free_mask.import_elements(rw, VectorOperation::insert);
  }

  // not zeroed on constrained DoFs: with a single constrained component per
  // node (symmetry), an aggregate could get a zero null space column
  modes = DoFTools::extract_rigid_body_modes(mapping, dof_handler);
}



template <int dim, typename Number, typename Material>
void
TangentMueLu<dim, Number, Material>::rebuild(const VectorType &u)
{
  Timer timer;

  // 1. u with ghost values on the host
  LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
  rw.import_elements(u, VectorOperation::insert);
  LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> u_host(locally_owned_dofs,
                                                                       locally_relevant_dofs,
                                                                       comm);
  u_host.import_elements(rw, VectorOperation::insert);
  u_host.update_ghost_values();

  // 2. K_ij = (grad phi_i, dP(grad u; grad phi_j))
  const FiniteElement<dim> &fe = dof_handler.get_fe();
  FEValues<dim> fe_values(mapping, fe, quadrature, update_gradients | update_JxW_values);
  const FEValuesExtractors::Vector displacement(0);

  const unsigned int                   n = fe.n_dofs_per_cell();
  FullMatrix<Number>                   cell_matrix(n, n);
  std::vector<types::global_dof_index> dof_indices(n);
  std::vector<Tensor<2, dim, Number>>  grad_u(quadrature.size());
  std::vector<Tensor<2, dim, Number>>  grad_phi(n);

  matrix = 0.;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[displacement].get_function_gradients(u_host, grad_u);
        cell_matrix = 0.;
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            for (unsigned int i = 0; i < n; ++i)
              grad_phi[i] = fe_values[displacement].gradient(i, q);
            for (unsigned int j = 0; j < n; ++j)
              {
                const Tensor<2, dim, Number> dP =
                  material.linearized_stress(grad_u[q], grad_phi[j]) * fe_values.JxW(q);
                for (unsigned int i = 0; i < n; ++i)
                  cell_matrix(i, j) += scalar_product(grad_phi[i], dP);
              }
          }
        cell->get_dof_indices(dof_indices);
        constraints.distribute_local_to_global(cell_matrix, dof_indices, matrix);
      }
  matrix.compress(VectorOperation::add);
  t_assemble += timer.wall_time();

  // 3. hierarchy: dim DoFs per node, rigid body modes as near null space
  timer.restart();
  Teuchos::ParameterList list;
  list.set("multigrid algorithm", "sa");
  list.set("problem: symmetric", true);
  list.set("smoother: type", "CHEBYSHEV");
  list.sublist("smoother: params").set("smoother: sweeps", 2);
  list.set("coarse: type", "KLU2");
  list.set("coarse: max size", 2000);
  list.set("cycle type", "V");
  list.set("verbosity", "none");
  list.set("number of equations", dim);

  Teuchos::RCP<MultiVector> nullspace =
    Teuchos::rcp(new MultiVector(matrix.trilinos_rcp()->getDomainMap(), modes.size()));
  {
    auto view = nullspace->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (unsigned int m = 0; m < modes.size(); ++m)
      for (unsigned int row = 0; row < modes[m].size(); ++row)
        view(row, m) = modes[m][row];
  }
  list.sublist("user data").set("Nullspace", nullspace);

  amg.initialize(matrix, list);
  t_setup += timer.wall_time();
}



template <int dim, typename Number, typename Material>
void
TangentMueLu<dim, Number, Material>::vmult(VectorType &dst, const VectorType &src) const
{
  using DeviceSpace    = MemorySpaceType::kokkos_space;
  const unsigned int n = src.locally_owned_size();

  // an aggregate holds whole nodes, so without the mask the V-cycle would
  // return values on the constrained components of partly constrained nodes
  masked_src = src;
  masked_src.scale(free_mask);

  // the locally owned DoFs of both vector types are in the same order
  {
    Kokkos::View<const Number *, DeviceSpace> in(masked_src.get_values(), n);
    auto out = src_tpetra.trilinos_vector().getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::deep_copy(Kokkos::subview(out, Kokkos::ALL(), 0), in);
  }

  amg.vmult(dst_tpetra, src_tpetra);

  {
    auto in = dst_tpetra.trilinos_vector().getLocalViewDevice(Tpetra::Access::ReadOnly);
    Kokkos::View<Number *, DeviceSpace> out(dst.get_values(), n);
    Kokkos::deep_copy(out, Kokkos::subview(in, Kokkos::ALL(), 0));
  }

  // dst = mask * dst + (1 - mask) * src
  dst.scale(free_mask);
  dst += src;
  dst -= masked_src;
}

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA_MUELU

#endif
