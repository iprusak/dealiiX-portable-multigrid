#ifndef portable_laplace_operator_dg_h
#define portable_laplace_operator_dg_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"
#include "kernels/laplace_dg_kokkos_kernels.h"
#include "kernels/portable_local_laplace_operator.h"
#include "operators/portable_laplace_operator_quad.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  class LaplaceOperatorDG : public LaplaceOperatorBase<dim, Number>
  {
  public:
    LaplaceOperatorDG() = default;

    void
    reinit(const Mapping<dim>              &mapping,
           const DoFHandler<dim>           &dof_handler,
           const AffineConstraints<Number> &constraints,
           bool                             overlap_communication_computation);

    void
    vmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    vmult_dummy(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
                const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
                const bool ghost_exchange_on,
                const bool computation_on) const override;


    void
    Tvmult(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const override;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const override;

    void
    compute_diagonal() override;

    void
    setup_dof_indices_per_color();

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const override;

    types::global_dof_index
    m() const override;

    types::global_dof_index
    n() const override;

    Number
    el(const types::global_dof_index row, const types::global_dof_index col) const override;

    const MatrixFree<dim, Number> &
    get_matrix_free() const override;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const override;

    void
    compute_G_tensors();

  private:
    using TeamHandle =
      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues =
      Kokkos::View<Number *,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients =
      Kokkos::View<Number **,
                   MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static constexpr unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int     n_q_points   = Utilities::pow(n_q_points_1d, dim);

    MatrixFree<dim, Number> matrix_free;

    ObserverPointer<const AffineConstraints<Number>> constraints;

    std::shared_ptr<
      DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
      inverse_diagonal_entries;

    std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dof_indices_per_color;

    // Kokkos::View<Number **, MemorySpace::Default::kokkos_space>  quad_values;
    std::vector<Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>> face_values_at_quads;
    std::vector<Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>>
      face_normal_derivatives_at_quads;

    Kokkos::View<Number ***, MemorySpace::Default::kokkos_space> interpolate_quad_to_boundary;

    std::vector<Kokkos::View<Number *, MemorySpace::Default::kokkos_space>> G_tensors;
  };

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::reinit(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<Number> &constraints,
    bool                             overlap_communication_computation)
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    this->constraints = &constraints;

    QGauss<1> quadrature_1d(n_q_points_1d);

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    matrix_free.reinit(mapping, dof_handler, constraints, quadrature_1d, additional_data);

    setup_dof_indices_per_color();

    // We need to compute the values of the shape functions at the quadrature points on the boundary
    // for the face integrals
    {
      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quadrature_1d.get_points());

      interpolate_quad_to_boundary = Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("interpolate_quad_to_boundary", Kokkos::WithoutInitializing),
        2,            // location at 0 or 1
        basis.size(), // quad points
        2);           // values and derivative

      AssertDimension(basis.size(), n_q_points_1d);


      auto interpolate_quad_to_boundary_host =
        Kokkos::create_mirror_view(interpolate_quad_to_boundary);

      std::vector<double> val_and_der(2);

      for (unsigned int i = 0; i < basis.size(); ++i)
        {
          basis[i].value(0., val_and_der);
          interpolate_quad_to_boundary_host(0, i, 0) = val_and_der[0];
          interpolate_quad_to_boundary_host(1, i, 0) = val_and_der[1];

          basis[i].value(1., val_and_der);
          interpolate_quad_to_boundary_host(0, i, 1) = val_and_der[0];
          interpolate_quad_to_boundary_host(1, i, 1) = val_and_der[1];
        }

      Kokkos::deep_copy(interpolate_quad_to_boundary, interpolate_quad_to_boundary_host);
      Kokkos::fence();


      const auto        &colored_graph       = matrix_free.get_colored_graph();
      const unsigned int n_colors            = colored_graph.size();
      constexpr int      n_q_points_per_face = Utilities::pow(n_q_points_1d, dim - 1);

      face_values_at_quads.resize(n_colors);
      face_normal_derivatives_at_quads.resize(n_colors);

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          face_values_at_quads[color] =
            Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("face_values_at_quads", Kokkos::WithoutInitializing),
              n_q_points_per_face,
              2 * dim,
              colored_graph[color].size());
          face_normal_derivatives_at_quads[color] =
            Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("face_normal_derivatives_at_quads", Kokkos::WithoutInitializing),
              n_q_points_per_face,
              2 * dim,
              colored_graph[color].size());
        }
    }

    compute_G_tensors();
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::vmult(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    dst = 0.;

    DeviceVector<Number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    constexpr bool is_serial =
      std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

    unsigned int n_cells_per_batch = numbers::invalid_unsigned_int;
    unsigned int n_blocks          = numbers::invalid_unsigned_int;
    unsigned int threads_per_block = numbers::invalid_unsigned_int;
    if (is_serial)
      {
        n_cells_per_batch = 1u;
        n_blocks          = 1u;
        threads_per_block = 1u;
      }

    // helper to process one color
    auto do_color = [&](const unsigned int color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            BK3::DG::compute_cell<dim, fe_degree + 1, n_q_points_1d, Number>(
              precomputed_data.shape_values,
              precomputed_data.co_shape_gradients,
              G_tensors[color],
              src_device,
              dst_device,
              interpolate_quad_to_boundary,
              face_values_at_quads[color],
              face_normal_derivatives_at_quads[color],
              dof_indices_per_color[color],
              n_cells,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);
          }
      };

    if (matrix_free.use_overlap_communication_computation())
      {
        src.update_ghost_values_start(0);

        // In parallel, it's possible that some processors do not own any
        // cells.
        if (colored_graph.size() > 0 && colored_graph[0].size() > 0)
          do_color(0);

        src.update_ghost_values_finish();

        // In serial this color does not exist because there are no ghost
        // cells
        if (colored_graph.size() > 1 && colored_graph[1].size() > 0)
          {
            do_color(1);

            // We need a synchronization point because we don't want
            // device-aware MPI to start the MPI communication until the
            // kernel is done.
            Kokkos::fence();
          }

        dst.compress_start(0, VectorOperation::add);
        // When the mesh is coarse it is possible that some processors do
        // not own any cells
        if (colored_graph.size() > 2 && colored_graph[2].size() > 0)
          do_color(2);
        dst.compress_finish(VectorOperation::add);
      }
    else
      {
        src.update_ghost_values();

        for (unsigned int color = 0; color < n_colors; ++color)
          {
            if (colored_graph[color].size() > color)
              do_color(color);
          }
        dst.compress(VectorOperation::add);
      }

    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);

    // dst = 0.;

    // LocalLaplaceOperator<dim, fe_degree, Number> cell_operator;

    // this->cell_loop(cell_operator, src, dst);

    // matrix_free.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::vmult_dummy(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src,
    const bool                                                              ghost_exchange_on,
    const bool                                                              computation_on) const
  {
    (void)dst;
    (void)src;
    (void)ghost_exchange_on;
    (void)computation_on;



    // dst = 0.;

    // LocalLaplaceOperator<dim, fe_degree, n_q_points_1d, Number> cell_operator;

    // this->cell_loop_dummy(cell_operator, src, dst, ghost_exchange_on, computation_on);

    // matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::setup_dof_indices_per_color()
  {
    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    const auto &dof_handler = matrix_free.get_dof_handler();

    std::vector<unsigned int> lex_numbering(n_local_dofs);

    {
      const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;


      shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
      lex_numbering = shape_info.lexicographic_numbering;
    }

    this->dof_indices_per_color.clear();
    this->dof_indices_per_color.resize(n_colors);

    std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
    std::vector<types::global_dof_index> subdomain_local_dof_indices(n_local_dofs);

    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto &mf_data = matrix_free.get_data(color);

            const auto &graph = colored_graph[color];

            this->dof_indices_per_color[color] =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("dof_indices_" + std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_local_dofs,
                mf_data.n_cells);

            auto dof_indices_host = Kokkos::create_mirror_view(this->dof_indices_per_color[color]);

            for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
              {
                auto triacell = graph[cell_id];

                typename DoFHandler<dim>::cell_iterator cell =
                  triacell->as_dof_handler_iterator(dof_handler);

                cell->get_dof_indices(local_dof_indices);

                triacell->get_dof_indices(subdomain_local_dof_indices);

                if (partitioner)
                  for (auto &index : local_dof_indices)
                    index = partitioner->global_to_local(index);

                for (unsigned int i = 0; i < n_local_dofs; ++i)
                  {
                    const auto global_dof          = local_dof_indices[lex_numbering[i]];
                    const auto subdomain_local_dof = subdomain_local_dof_indices[lex_numbering[i]];

                    if (constraints->is_constrained(subdomain_local_dof))
                      dof_indices_host(i, cell_id) = numbers::invalid_unsigned_int;
                    else
                      dof_indices_host(i, cell_id) = global_dof;
                  }
              }

            Kokkos::deep_copy(exec_space, this->dof_indices_per_color[color], dof_indices_host);
            Kokkos::fence();
          }
      }
  }


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::Tvmult(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &src) const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }


  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  const MatrixFree<dim, Number> &
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::get_matrix_free() const
  {
    return matrix_free;
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &inverse_diagonal =
      inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    internal::LaplaceOperatorQuad<dim, fe_degree, n_q_points_1d, Number> operator_quad;

    MatrixFreeTools::compute_diagonal<dim, fe_degree, n_q_points_1d, 1, Number>(
      matrix_free,
      inverse_diagonal,
      operator_quad,
      EvaluationFlags::gradients,
      EvaluationFlags::gradients);

    Number *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::compute_G_tensors()
  {
    constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    G_tensors.resize(n_colors);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto        &precomputed_data = matrix_free.get_data(color);
            const unsigned int n_cells          = precomputed_data.n_cells;

            const auto &inv_jacobian = precomputed_data.inv_jacobian;
            const auto &JxW          = precomputed_data.JxW;

            G_tensors[color] = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("G_tensor_color_" + std::to_string(color),
                                 Kokkos::WithoutInitializing),
              symmetric_tensor_dim * n_cells * n_q_points);

            auto G = G_tensors[color];

            Kokkos::parallel_for(
              "Fill_G_tensor_color" + std::to_string(color),
              Kokkos::RangePolicy<dealii::MemorySpace::Default::kokkos_space::execution_space>(
                0, n_cells),
              KOKKOS_LAMBDA(const int cell_id) {
                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    Number components[symmetric_tensor_dim];

                    int idx = 0;
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = d1; d2 < dim; ++d2)
                        {
                          Number sum = 0;
                          for (int k = 0; k < dim; ++k)
                            sum += inv_jacobian(q_point, cell_id, k, d1) *
                                   inv_jacobian(q_point, cell_id, k, d2);
                          components[idx] = JxW(q_point, cell_id) * sum;
                          ++idx;
                        }

                    for (int c = 0; c < symmetric_tensor_dim; ++c)
                      {
                        G[cell_id * symmetric_tensor_dim * n_q_points + c * n_q_points + q_point] =
                          components[c];
                      }
                  }
              });
            Kokkos::fence();
          }
      }
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  std::shared_ptr<DiagonalMatrix<LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>>>
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  types::global_dof_index
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  types::global_dof_index
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  Number
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr && inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());

    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif