// Correctness check for Portable::LaplaceOperatorBK3 (portable_laplace_operator_bk3.h)
// on a curved, sheared, non-axis-aligned mesh: vmult(), Tvmult() and
// compute_diagonal() are compared against a CPU FEValues reference built with
// the same MappingQ(fe_degree) the operator uses internally.
//
// Every other test in correctness_tests/ runs on an axis-aligned hyper-cube,
// where the inverse Jacobian is diagonal and constant, so a wrong contraction
// index in the G-tensor (J^{-1} J^{-T} vs J^{-T} J^{-1}) or a wrong Jacobian
// layout would go unnoticed. Here all dim*(dim+1)/2 entries of G vary per
// quadrature point.
//
// The reference does exactly what the operator does with Dirichlet
// constraints: constrained source entries are read as zero, constrained rows
// of the result are not accumulated, and afterwards dst[c] = src[c] on every
// constrained DoF (copy_constrained_values). The diagonal is expected to be 1
// on constrained DoFs.
//
// Runs for fe_degree = 1..4 in double (tolerance 1e-12) and float (1e-5),
// dim = 2 and 3, with and without overlap_communication_computation. The
// overlap configuration is skipped (not failed) when the build cannot
// support it, see run_test().

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "operators/portable_laplace_operator_bk3.h"

namespace multigrid
{
  using namespace dealii;

  // Smooth, anisotropic, sheared and non-affine diffeomorphism of the unit
  // cube with an analytic inverse. The coarse hyper_cube is pushed through
  // map_forward() and the manifold is attached, so refinement produces
  // genuinely curved cells for MappingQ(p > 1).
  template <int dim>
  class CurvedShearedManifold : public ChartManifold<dim, dim, dim>
  {
  public:
    static Point<dim>
    map_forward(const Point<dim> &chart_point)
    {
      if constexpr (dim == 2)
        {
          const double xi  = chart_point[0];
          const double eta = chart_point[1];

          const double y = 1.1 * eta;
          const double x = 1.3 * xi + 0.18 * y + 0.15 * std::sin(numbers::PI * eta);
          return Point<dim>(x, y);
        }
      else
        {
          const double xi   = chart_point[0];
          const double eta  = chart_point[1];
          const double zeta = chart_point[2];

          const double z = 0.7 * zeta;
          const double y = 1.1 * eta + 0.12 * std::sin(numbers::PI * zeta);
          const double x = 2.3 * xi + 0.18 * y +
                           0.15 * std::sin(numbers::PI * eta) * std::cos(numbers::PI * zeta);
          return Point<dim>(x, y, z);
        }
    }

    static Point<dim>
    map_backward(const Point<dim> &space_point)
    {
      if constexpr (dim == 2)
        {
          const double x = space_point[0];
          const double y = space_point[1];

          const double eta = y / 1.1;
          const double xi  = (x - 0.18 * y - 0.15 * std::sin(numbers::PI * eta)) / 1.3;
          return Point<dim>(xi, eta);
        }
      else
        {
          const double x = space_point[0];
          const double y = space_point[1];
          const double z = space_point[2];

          const double zeta = z / 0.7;
          const double eta  = (y - 0.12 * std::sin(numbers::PI * zeta)) / 1.1;
          const double xi =
            (x - 0.18 * y - 0.15 * std::sin(numbers::PI * eta) * std::cos(numbers::PI * zeta)) /
            2.3;
          return Point<dim>(xi, eta, zeta);
        }
    }

    Point<dim>
    push_forward(const Point<dim> &chart_point) const override
    {
      return map_forward(chart_point);
    }

    Point<dim>
    pull_back(const Point<dim> &space_point) const override
    {
      return map_backward(space_point);
    }

    std::unique_ptr<Manifold<dim, dim>>
    clone() const override
    {
      return std::make_unique<CurvedShearedManifold<dim>>();
    }
  };



  // CPU reference: dst = A * src and diag = diag(A) with the operator's
  // Dirichlet convention, A assembled cell by cell with FEValues in double
  // and never stored globally.
  template <int dim, typename Number>
  void
  reference_vmult_and_diagonal(
    const Mapping<dim>                                                     &mapping,
    const DoFHandler<dim>                                                  &dof_handler,
    const AffineConstraints<Number>                                        &constraints,
    const unsigned int                                                      n_q_points_1d,
    const IndexSet                                                         &locally_owned_dofs,
    const IndexSet                                                         &locally_relevant_dofs,
    const MPI_Comm                                                          mpi_communicator,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src_device,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst_device,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &diag_device)
  {
    using HostVector = LinearAlgebra::distributed::Vector<double, MemorySpace::Host>;

    HostVector src_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    {
      LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
      rw.import_elements(src_device, VectorOperation::insert);
      for (const auto i : locally_owned_dofs)
        src_host(i) = rw(i);
    }
    src_host.update_ghost_values();

    HostVector dst_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    HostVector diag_host(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const FiniteElement<dim> &fe = dof_handler.get_fe();
    const QGauss<dim>         quadrature(n_q_points_1d);
    FEValues<dim> fe_values(mapping, fe, quadrature, update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();

    FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                       x_local(dofs_per_cell), y_local(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0.;

          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) +=
                  fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            x_local(j) = constraints.is_constrained(local_dof_indices[j]) ?
                           0. :
                           src_host(local_dof_indices[j]);

          cell_matrix.vmult(y_local, x_local);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (!constraints.is_constrained(local_dof_indices[i]))
              {
                dst_host(local_dof_indices[i]) += y_local(i);
                diag_host(local_dof_indices[i]) += cell_matrix(i, i);
              }
        }

    dst_host.compress(VectorOperation::add);
    diag_host.compress(VectorOperation::add);

    for (const auto i : locally_owned_dofs)
      if (constraints.is_constrained(i))
        {
          dst_host(i)  = src_host(i);
          diag_host(i) = 1.;
        }

    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
    for (const auto i : locally_owned_dofs)
      rw(i) = static_cast<Number>(dst_host(i));
    dst_device.import_elements(rw, VectorOperation::insert);

    for (const auto i : locally_owned_dofs)
      rw(i) = static_cast<Number>(diag_host(i));
    diag_device.import_elements(rw, VectorOperation::insert);
  }



  template <typename Number>
  double
  relative_l2_difference(
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &a,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &b)
  {
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> diff(a);
    diff -= b;
    const double norm_b = b.l2_norm();
    return norm_b > 0 ? diff.l2_norm() / norm_b : diff.l2_norm();
  }



  // Returns the largest relative error over vmult, Tvmult and the diagonal
  // for one configuration, or -1 if the configuration was skipped.
  template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout, const bool overlap)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_q_points_1d = " << n_q_points_1d << ", "
          << (std::is_same_v<Number, double> ? "double" : "float")
          << ", overlap = " << (overlap ? "true" : "false") << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    GridTools::transform(&CurvedShearedManifold<dim>::map_forward, triangulation);

    const CurvedShearedManifold<dim> manifold;
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, manifold);
    triangulation.refine_global(dim == 2 ? 3 : 2);

    const FE_Q<dim>     fe(fe_degree);
    const MappingQ<dim> mapping(fe_degree);
    DoFHandler<dim>     dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<Number> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Functions::ZeroFunction<dim, Number>(), constraints);
    constraints.close();

    std::unique_ptr<Portable::LaplaceOperatorBK3<dim, fe_degree, Number, n_q_points_1d>> op;
    try
      {
        op = std::make_unique<Portable::LaplaceOperatorBK3<dim, fe_degree, Number, n_q_points_1d>>(
          dof_handler, constraints, overlap);
      }
    catch (const ExceptionBase &exc)
      {
        // deal.II refuses overlap_communication_computation without
        // device-aware MPI. That is a property of the build, not a bug in
        // the operator, so report the configuration as skipped.
        if (overlap && std::string(exc.what()).find("device-aware MPI") != std::string::npos)
          {
            pcout << "  SKIPPED: " << exc.get_exc_name() << " (overlap needs device-aware MPI)"
                  << std::endl
                  << std::endl;
            return -1.;
          }
        throw;
      }

    AssertThrow(op->m() == dof_handler.n_dofs() && op->n() == dof_handler.n_dofs(),
                ExcMessage("m()/n() do not match the number of DoFs."));

    using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    VectorType src, dst_gpu, dst_tvmult_gpu, dst_ref, diag_ref;
    op->initialize_dof_vector(src);
    op->initialize_dof_vector(dst_gpu);
    op->initialize_dof_vector(dst_tvmult_gpu);
    op->initialize_dof_vector(dst_ref);
    op->initialize_dof_vector(diag_ref);

    // Deterministic, rank-independent fill; deliberately nonzero on Dirichlet
    // DoFs so copy_constrained_values() is exercised.
    {
      LinearAlgebra::ReadWriteVector<Number> rw(locally_owned_dofs);
      for (const auto i : locally_owned_dofs)
        rw(i) = static_cast<Number>(std::sin(0.37 * static_cast<double>(i + 1)) +
                                    0.5 * std::cos(0.13 * static_cast<double>(i + 3)));
      src.import_elements(rw, VectorOperation::insert);
    }

    op->vmult(dst_gpu, src);
    op->Tvmult(dst_tvmult_gpu, src);
    op->compute_diagonal();
    Kokkos::fence();

    reference_vmult_and_diagonal<dim, Number>(mapping,
                                              dof_handler,
                                              constraints,
                                              n_q_points_1d,
                                              locally_owned_dofs,
                                              locally_relevant_dofs,
                                              mpi_communicator,
                                              src,
                                              dst_ref,
                                              diag_ref);

    const double err_vmult  = relative_l2_difference(dst_gpu, dst_ref);
    const double err_tvmult = relative_l2_difference(dst_tvmult_gpu, dst_ref);

    // Diagonal: invert the operator's inverse diagonal on the host and
    // compare entry by entry (constrained entries must be exactly 1).
    double err_diag = 0.;
    {
      const VectorType &inverse_diagonal = op->get_matrix_diagonal_inverse()->get_vector();

      LinearAlgebra::ReadWriteVector<Number> rw_inv(locally_owned_dofs), rw_ref(locally_owned_dofs);
      rw_inv.import_elements(inverse_diagonal, VectorOperation::insert);
      rw_ref.import_elements(diag_ref, VectorOperation::insert);

      for (const auto i : locally_owned_dofs)
        {
          const double d_gpu = 1. / static_cast<double>(rw_inv(i));
          const double d_ref = static_cast<double>(rw_ref(i));
          err_diag           = std::max(err_diag, std::abs(d_gpu - d_ref) / std::abs(d_ref));
        }
      err_diag = Utilities::MPI::max(err_diag, mpi_communicator);
    }

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_ref| = " << dst_ref.l2_norm()
          << ", rel_err(vmult) = " << err_vmult << ", rel_err(Tvmult) = " << err_tvmult
          << ", max_rel_err(diag) = " << err_diag << std::endl
          << std::endl;

    return std::max({err_vmult, err_tvmult, err_diag});
  }



  template <int dim, int fe_degree, typename Number, int n_q_points_1d = fe_degree + 1>
  void
  run_both_overlap_modes(const MPI_Comm      mpi_communicator,
                         ConditionalOStream &pcout,
                         double             &max_err,
                         unsigned int       &n_skipped)
  {
    for (const bool overlap : {false, true})
      {
        const double err =
          run_test<dim, fe_degree, Number, n_q_points_1d>(mpi_communicator, pcout, overlap);
        if (err < 0.)
          ++n_skipped;
        else
          max_err = std::max(max_err, err);
      }
  }
} // namespace multigrid



int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  double       max_err_fp64 = 0., max_err_fp32 = 0.;
  unsigned int n_skipped = 0;

  try
    {
      // 3D, fp64, default quadrature (nq == p + 1)
      run_both_overlap_modes<3, 1, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 2, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 3, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 4, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);

      // 3D, fp64, over-integrated (nq == p + 2), what tests/bp3 uses
      run_both_overlap_modes<3, 2, double, 4>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<3, 3, double, 5>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);

      // 3D, fp32 (nq == p + 1 is odd for even p)
      run_both_overlap_modes<3, 1, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);
      run_both_overlap_modes<3, 2, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);
      run_both_overlap_modes<3, 3, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);
      run_both_overlap_modes<3, 4, float>(MPI_COMM_WORLD, pcout, max_err_fp32, n_skipped);

      // 2D, fp64
      run_both_overlap_modes<2, 1, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<2, 2, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<2, 3, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
      run_both_overlap_modes<2, 4, double>(MPI_COMM_WORLD, pcout, max_err_fp64, n_skipped);
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance_fp64 = 1e-12;
  const double tolerance_fp32 = 1e-5;

  if (n_skipped > 0)
    pcout << "Note: " << n_skipped
          << " overlap configuration(s) skipped (build without device-aware MPI)." << std::endl;

  if (max_err_fp64 > tolerance_fp64 || max_err_fp32 > tolerance_fp32)
    {
      pcout << "FAILED: max relative error fp64 " << max_err_fp64 << " (tolerance "
            << tolerance_fp64 << "), fp32 " << max_err_fp32 << " (tolerance " << tolerance_fp32
            << ")" << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error fp64 " << max_err_fp64 << " (tolerance " << tolerance_fp64
        << "), fp32 " << max_err_fp32 << " (tolerance " << tolerance_fp32 << ")" << std::endl;

  return 0;
}
