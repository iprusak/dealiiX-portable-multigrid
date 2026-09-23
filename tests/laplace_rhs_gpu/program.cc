// Correctness check for the scalar operators' new GPU RHS assembly:
// Portable::LaplaceOperator::compute_rhs_bk3_abstracted()
// (portable_laplace_operator.h) and
// Portable::LaplaceOperatorBK3::compute_rhs() (portable_laplace_operator_bk3.h).
// Both are built on BK3::Parallel::KokkosRHSAbstracted() (bk3_kokkos_kernels.h)
// -- the same Custom::Parallel abstraction layer their own vmult_bk3_abstracted()/
// vmult() already use -- replacing the CPU FEValues loop every source/ and
// tests/ program used to run through host memory. Compares each against
// that same CPU FEValues assembly (real deal.II, taken as ground truth) on
// a couple of mesh/degree combinations.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>

#include "operators/portable_laplace_operator.h"
#include "operators/portable_laplace_operator_bk3.h"

namespace multigrid
{
  using namespace dealii;

  // CPU FEValues assembly of the same constant-unit-load RHS every
  // source/tests program used before these GPU paths existed -- taken as
  // ground truth here.
  template <int dim>
  void
  compute_rhs_fevalues(
    const DoFHandler<dim>                                            &dof_handler,
    const FE_Q<dim>                                                  &fe,
    const AffineConstraints<double>                                  &constraints,
    const IndexSet                                                   &locally_owned_dofs,
    const IndexSet                                                   &locally_relevant_dofs,
    const MPI_Comm                                                   &mpi_communicator,
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &rhs_device)
  {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> rhs_host(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim>      fe_values(fe, quadrature_formula, update_values | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += fe_values.shape_value(i, q_index) * 1.0 * fe_values.JxW(q_index);

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs, local_dof_indices, rhs_host);
        }

    rhs_host.compress(VectorOperation::add);
    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import_elements(rhs_host, VectorOperation::insert);
    rhs_device.import_elements(rw_vector, VectorOperation::insert);
  }

  // Runs the comparison for one (dim, fe_degree) combination on a small
  // hyper-cube mesh: compute_rhs_bk3_abstracted() (LaplaceOperator) and
  // compute_rhs() (LaplaceOperatorBK3) against the CPU FEValues assembly.
  // Returns the max relative error over both, which should be at machine
  // precision.
  template <int dim, int fe_degree>
  double
  run_test(const MPI_Comm mpi_communicator, ConditionalOStream &pcout)
  {
    pcout << "dim = " << dim << ", fe_degree = " << fe_degree << std::endl;

    parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(dim == 2 ? 5 : 3);

    const FE_Q<dim> fe(fe_degree);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs    = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    Functions::ZeroFunction<dim>                        zero;
    std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary_functions = {
      {types::boundary_id(0), &zero}};
    VectorTools::interpolate_boundary_values(dof_handler,
                                              dirichlet_boundary_functions,
                                              constraints);
    constraints.close();

    double max_rel_err = 0.;

    // Portable::LaplaceOperator::compute_rhs_bk3_abstracted()
    {
      Portable::LaplaceOperator<dim, fe_degree, double> op(
        dof_handler, constraints, /* overlap_communication_computation = */ false);

      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> rhs_cpu, rhs_gpu;
      op.initialize_dof_vector(rhs_cpu);
      op.initialize_dof_vector(rhs_gpu);

      compute_rhs_fevalues<dim>(dof_handler,
                                fe,
                                constraints,
                                locally_owned_dofs,
                                locally_relevant_dofs,
                                mpi_communicator,
                                rhs_cpu);
      const double norm_cpu = rhs_cpu.l2_norm();

      op.compute_rhs_bk3_abstracted(rhs_gpu);

      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = rhs_gpu;
      diff -= rhs_cpu;
      const double rel_err = norm_cpu > 0 ? diff.l2_norm() / norm_cpu : diff.l2_norm();

      pcout << "  [LaplaceOperator]     dofs = " << dof_handler.n_dofs()
            << ", rel_err(gpu vs cpu) = " << rel_err << std::endl;

      max_rel_err = std::max(max_rel_err, rel_err);
    }

    // Portable::LaplaceOperatorBK3::compute_rhs()
    {
      Portable::LaplaceOperatorBK3<dim, fe_degree, double> op(
        dof_handler, constraints, /* overlap_communication_computation = */ false);

      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> rhs_cpu, rhs_gpu;
      op.initialize_dof_vector(rhs_cpu);
      op.initialize_dof_vector(rhs_gpu);

      compute_rhs_fevalues<dim>(dof_handler,
                                fe,
                                constraints,
                                locally_owned_dofs,
                                locally_relevant_dofs,
                                mpi_communicator,
                                rhs_cpu);
      const double norm_cpu = rhs_cpu.l2_norm();

      op.compute_rhs(rhs_gpu);

      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> diff = rhs_gpu;
      diff -= rhs_cpu;
      const double rel_err = norm_cpu > 0 ? diff.l2_norm() / norm_cpu : diff.l2_norm();

      pcout << "  [LaplaceOperatorBK3]  dofs = " << dof_handler.n_dofs()
            << ", rel_err(gpu vs cpu) = " << rel_err << std::endl;

      max_rel_err = std::max(max_rel_err, rel_err);
    }

    pcout << std::endl;

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

  double max_rel_err = 0.;

  try
    {
      max_rel_err = std::max(max_rel_err, run_test<2, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 2>(MPI_COMM_WORLD, pcout));
      max_rel_err = std::max(max_rel_err, run_test<3, 3>(MPI_COMM_WORLD, pcout));
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
