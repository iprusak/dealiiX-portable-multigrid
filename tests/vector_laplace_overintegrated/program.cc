// Correctness check for Portable::VectorLaplaceOperator's n_q_points_1d
// template parameter (portable_vector_laplace_operator.h): unlike tests/
// vector_laplace_bk4 and tests/vector_laplace_tensor_core, which both use
// the operator's default n_q_points_1d == fe_degree + 1, this exercises
// n_q_points_1d > fe_degree + 1 (over-integration, nq != nm) -- the same
// configuration tests/bp4/program.cc now uses (fe_degree + 2) -- on all
// three backends: vmult_dealii() (ground truth), vmult_bk4() (the custom
// Kokkos kernel), and vmult_tensor_core() (the FP64 Tensor Core kernel,
// only when compiled by nvcc against a CUDA-enabled Kokkos -- see the
// __CUDACC__ guard below, same graceful-degradation pattern as tests/
// vector_laplace_tensor_core/program.cc).
//
// Also runs one case at the default n_q_points_1d (5th template argument
// omitted) as a regression check that adding the template parameter didn't
// change the pre-existing nq == nm behavior.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>
#include <random>

#include "operators/portable_vector_laplace_operator.h"

namespace multigrid
{
  using namespace dealii;

  // Runs the comparison for one (dim, fe_degree, n_components,
  // n_q_points_1d) combination on a small hyper-cube mesh. Returns the max
  // relative error over whichever checks actually ran (rel_err(bk4 vs
  // dealii) always; rel_err(tensor_core vs dealii) only for dim == 3 when
  // built with nvcc against a CUDA-enabled Kokkos), all of which should be
  // at machine precision.
  template <int dim, int fe_degree, int n_components, int n_q_points_1d = fe_degree + 1>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree
          << ", n_components = " << n_components << ", n_q_points_1d = " << n_q_points_1d
          << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(dim == 2 ? 5 : 3);

    const FE_Q<dim> scalar_fe(fe_degree);
    FESystem<dim>   fe(scalar_fe, n_components);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim>                        zero(n_components);
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(dof_handler,
                                              dirichlet_boundary_functions,
                                              constraints);
    constraints.close();

    Portable::VectorLaplaceOperator<dim, fe_degree, n_components, double, n_q_points_1d> op(
      dof_handler, constraints, /* overlap_communication_computation = */ false);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src, dst_dealii, dst_bk4;
    op.initialize_dof_vector(src);
    op.initialize_dof_vector(dst_dealii);
    op.initialize_dof_vector(dst_bk4);

    {
      std::mt19937                           gen(42);
      std::uniform_real_distribution<double> dist(-1., 1.);

      LinearAlgebra::ReadWriteVector<double> rw(locally_owned_dofs);
      for (const auto idx : locally_owned_dofs)
        rw(idx) = dist(gen);
      src.import_elements(rw, VectorOperation::insert);
    }

    // vmult_dealii() reads src at constrained DoFs directly (it relies on
    // copy_constrained_values() afterwards), while vmult_bk4()/
    // vmult_tensor_core() never read constrained entries at all -- zero
    // them up front so every variant sees the same effective input, exactly
    // as tests/vector_laplace_bk4/program.cc and tests/
    // vector_laplace_tensor_core/program.cc do.
    op.get_matrix_free().set_constrained_values(0., src);

    op.vmult_dealii(dst_dealii, src);
    op.vmult_bk4(dst_bk4, src);

    op.get_matrix_free().set_constrained_values(0., dst_dealii);
    op.get_matrix_free().set_constrained_values(0., dst_bk4);

    const double norm_dealii = dst_dealii.l2_norm();

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = dst_bk4;
    diff -= dst_dealii;
    const double rel_err_bk4 = norm_dealii > 0 ? diff.l2_norm() / norm_dealii : diff.l2_norm();

    pcout << "  dofs = " << dof_handler.n_dofs() << ", |dst_dealii| = " << norm_dealii
          << ", rel_err(bk4 vs dealii) = " << rel_err_bk4 << std::endl;

    double max_rel_err = rel_err_bk4;

    // vmult_tensor_core() is dim == 3 + double only (static_assert()s
    // otherwise) and needs an actual nvcc/CUDA build to do anything -- on a
    // Serial/OpenMP-only Kokkos build (such as this project's own dev box)
    // it Assert()s out at runtime, so it's only ever actually called here
    // under __CUDACC__, exactly as tests/vector_laplace_tensor_core/
    // program.cc does.
    if constexpr (dim == 3)
      {
#ifdef __CUDACC__
        LinearAlgebra::distributed::Vector<double, MemorySpace::Default> dst_tensor_core;
        op.initialize_dof_vector(dst_tensor_core);

        op.vmult_tensor_core(dst_tensor_core, src);
        op.get_matrix_free().set_constrained_values(0., dst_tensor_core);

        LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff_tc = dst_tensor_core;
        diff_tc -= dst_dealii;
        const double rel_err_tc =
          norm_dealii > 0 ? diff_tc.l2_norm() / norm_dealii : diff_tc.l2_norm();

        pcout << "  rel_err(tensor_core vs dealii) = " << rel_err_tc << std::endl;

        max_rel_err = std::max(max_rel_err, rel_err_tc);
#else
        pcout << "  rel_err(tensor_core vs dealii) = SKIPPED "
                 "(this translation unit was not compiled by nvcc)"
              << std::endl;
#endif
      }

    // Sanity-check compute_diagonal(): a Laplacian's diagonal must be
    // strictly positive everywhere (compute_diagonal() itself Asserts this
    // in debug mode already -- this just makes the check visible here too).
    op.compute_diagonal();
    const auto &inverse_diagonal = op.get_matrix_diagonal_inverse()->get_vector();
    LinearAlgebra::ReadWriteVector<double> diagonal_host(locally_owned_dofs);
    diagonal_host.import_elements(inverse_diagonal, VectorOperation::insert);
    double min_inverse_diagonal = std::numeric_limits<double>::max();
    for (const auto idx : locally_owned_dofs)
      min_inverse_diagonal = std::min(min_inverse_diagonal, diagonal_host(idx));
    pcout << "  min(1/diagonal) = " << min_inverse_diagonal << std::endl << std::endl;

    return max_rel_err;
  }
} // namespace multigrid

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace multigrid;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

#ifdef __CUDACC__
  pcout << "Built with nvcc: vmult_tensor_core() will actually run on the GPU." << std::endl
        << std::endl;
#else
  pcout << "NOT built with nvcc: vmult_tensor_core() checks below are skipped -- "
           "rebuild this target with a CUDA-enabled Kokkos to exercise them."
        << std::endl
        << std::endl;
#endif

  double max_rel_err = 0.;

  try
    {
      // Regression case: n_q_points_1d omitted, should behave exactly like
      // the pre-templating operator (nq == nm).
      max_rel_err = std::max(max_rel_err, run_test<3, 2, 3>(MPI_COMM_WORLD, pcout));

      // Over-integrated cases: nq != nm, the new behavior. fe_degree + 2
      // matches what tests/bp4/program.cc now uses; fe_degree + 3 pushes
      // further to make sure this isn't a fluke of a single offset.
      max_rel_err =
        std::max(max_rel_err, run_test<3, 2, 3, /* n_q_points_1d = */ 4>(MPI_COMM_WORLD, pcout));
      max_rel_err =
        std::max(max_rel_err, run_test<3, 3, 2, /* n_q_points_1d = */ 5>(MPI_COMM_WORLD, pcout));
      max_rel_err =
        std::max(max_rel_err, run_test<2, 2, 2, /* n_q_points_1d = */ 6>(MPI_COMM_WORLD, pcout));
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }

  const double tolerance = 1e-10;

  if (max_rel_err > tolerance)
    {
      pcout << "FAILED: max relative error " << max_rel_err << " exceeds tolerance " << tolerance
            << std::endl;
      return 1;
    }

  pcout << "PASSED: max relative error " << max_rel_err << " (tolerance " << tolerance << ")"
        << std::endl;

  return 0;
}
