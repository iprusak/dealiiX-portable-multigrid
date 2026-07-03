#ifndef portable_laplace_operator_dg_h
#define portable_laplace_operator_dg_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <memory>

#include "base/portable_laplace_operator_base.h"
#include "kernels/laplace_dg_kokkos_kernels.h"
#include "kernels/portable_local_laplace_operator.h"
#include "matrix_free/portable_shape_info.h"
#include "operators/portable_laplace_operator_quad.h"


DEAL_II_NAMESPACE_OPEN

namespace Portable
{

  namespace internal
  {

    struct FaceToCellTopology
    {
      unsigned int global_face_index;

      unsigned int cell_interior;

      unsigned int cell_exterior;

      unsigned int exterior_face_no;

      unsigned int interior_face_no;

      unsigned int face_orientation;
    };

  } // namespace internal

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

    void
    compute_face_info();

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

    static constexpr unsigned int n_local_dofs    = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int     n_q_points      = Utilities::pow(n_q_points_1d, dim);
    static const unsigned int     n_q_points_face = Utilities::pow(n_q_points_1d, dim - 1);


    MatrixFree<dim, Number> matrix_free;

    ObserverPointer<const AffineConstraints<Number>> constraints;

    ObserverPointer<const Mapping<dim>> mapping;

    ObserverPointer<const DoFHandler<dim>> dof_handler;

    std::unique_ptr<QGauss<1>> quadrature_1d;

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

    // 0 - inner faces, 1 - boundary faces
    Kokkos::Array<Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space>, 2> face_info;
    Kokkos::View<unsigned int * [2 * dim], MemorySpace::Default::kokkos_space> face_info_per_cell;
    Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jacobians_times_normal_inner_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> jacobians_times_normal_boundary_face;
    Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space> jxw_inner_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space>    jxw_boundary_face;


    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> penalty_parameters_inner_face;
    Kokkos::View<Number *, MemorySpace::Default::kokkos_space> penalty_parameters_boundary_face;

    internal::MatrixFreeFunctions::UnivariateShapeData<Number> shape_data;

    Number
    get_penalty_factor(const double cell_extent_minus, const double cell_extent_plus) const
    {
      return fe_degree * (fe_degree + 1.) * 0.5 * (1. / cell_extent_minus + 1. / cell_extent_plus);
    }
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
    this->mapping     = &mapping;
    this->dof_handler = &dof_handler;


    this->quadrature_1d = std::make_unique<QGauss<1>>(n_q_points_1d);

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation = overlap_communication_computation;

    matrix_free.reinit(mapping, dof_handler, constraints, *quadrature_1d, additional_data);

    {
      dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(*quadrature_1d,
                                                                          dof_handler.get_fe());

      shape_data.reinit(shape_info.get_shape_data());
    }

    setup_dof_indices_per_color();

    compute_G_tensors();

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    // We need to compute the values of the shape functions at the quadrature points on the boundary
    // for the face integrals
    {
      std::vector<Polynomials::Polynomial<double>> basis =
        Polynomials::generate_complete_Lagrange_basis(quadrature_1d->get_points());

      AssertDimension(basis.size(), n_q_points_1d);

      interpolate_quad_to_boundary = Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("interpolate_quad_to_boundary", Kokkos::WithoutInitializing),
        2,            // location at 0 or 1
        basis.size(), // quad points
        2);           // values and derivative

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

      constexpr int n_q_points_per_face = Utilities::pow(n_q_points_1d, dim - 1);

      face_values_at_quads.resize(n_colors);
      face_normal_derivatives_at_quads.resize(n_colors);

      for (unsigned int color = 0; color < n_colors; ++color)
        {
          face_values_at_quads[color] =
            Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("face_values_at_quads_color_" + std::to_string(color),
                                 Kokkos::WithoutInitializing),
              n_q_points_per_face,
              2 * dim,
              colored_graph[color].size());

          face_normal_derivatives_at_quads[color] =
            Kokkos::View<Number ***, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("face_normal_derivatives_at_quads_color_" + std::to_string(color),
                                 Kokkos::WithoutInitializing),
              n_q_points_per_face,
              2 * dim,
              colored_graph[color].size());
        }
    }

    compute_face_info();
  }

  template <int dim, int fe_degree, int n_q_points_1d, typename Number>
  void
  LaplaceOperatorDG<dim, fe_degree, n_q_points_1d, Number>::compute_face_info()
  {
    unsigned int cell_counter          = 0;
    unsigned int inner_face_counter    = 0;
    unsigned int boundary_face_counter = 0;

    std::vector<std::array<unsigned int, 5>>       inner_faces_v;
    std::vector<std::array<unsigned int, 5>>       boundary_faces_v;
    std::vector<std::array<unsigned int, 2 * dim>> faces_per_cell;

    std::vector<std::pair<unsigned int, unsigned int>> cell_level_index;


    const auto &triangulation = dof_handler->get_triangulation();

    // get interior and boundary faces information
    // 0 - interior cell
    // 1 - exterior cell
    // 2 - interior face no
    // 3 - exterior face no
    // 4 - face orientation
    {
      std::vector<bool> visited_face(triangulation.n_raw_faces());

      std::array<unsigned int, 2 * dim> f_ids_on_cell;

      for (const auto &cell : triangulation.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;


          for (const auto f : cell->face_indices())
            {
              const auto &face = cell->face(f);

              const unsigned int face_index = face->index();

              f_ids_on_cell[f] = face_index;

              if (visited_face[face_index])
                continue;

              visited_face[face_index] = true;

              if (face->at_boundary())
                {
                  ++boundary_face_counter;
                  std::array<unsigned int, 5> info;
                  info[0] = cell_counter;
                  info[1] = numbers::invalid_unsigned_int;
                  info[2] = f;
                  info[3] = face->boundary_id();
                  info[4] = 0;

                  boundary_faces_v.push_back(info);
                }
              else
                {
                  ++inner_face_counter;
                  typename dealii::Triangulation<dim>::cell_iterator neighbor =
                    cell->neighbor_or_periodic_neighbor(f);

                  const unsigned int neighbor_face_no = cell->has_periodic_neighbor(f) ?
                                                          cell->periodic_neighbor_face_no(f) :
                                                          cell->neighbor_face_no(f);

                  std::array<unsigned int, 5> info;
                  info[0] = cell_counter;
                  info[1] = neighbor->index();
                  info[2] = f;
                  info[3] = neighbor_face_no;
                  info[4] = 0;

                  inner_faces_v.push_back(info);
                }
            }

          ++cell_counter;
          faces_per_cell.push_back(f_ids_on_cell);
          cell_level_index.push_back({cell->level(), cell->index()});
        }

      Kokkos::View<unsigned int *[5],
                   Kokkos::LayoutRight,
                   Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        inner_faces_info_host(inner_faces_v.data()->data(), inner_faces_v.size());

      Kokkos::View<unsigned int *[5],
                   Kokkos::LayoutRight,
                   Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        boundary_faces_info_host(boundary_faces_v.data()->data(), boundary_faces_v.size());

      Kokkos::View<unsigned int * [2 * dim],
                   Kokkos::LayoutRight,
                   Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        face_info_per_cell_host(faces_per_cell.data()->data(), faces_per_cell.size());

      face_info[0] = Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("inner_faces_info", Kokkos::WithoutInitializing), inner_faces_v.size());

      face_info[1] = Kokkos::View<unsigned int *[5], MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("boundary_faces_info", Kokkos::WithoutInitializing),
        boundary_faces_v.size());

      face_info_per_cell =
        Kokkos::View<unsigned int * [2 * dim], MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("face_info_per_cell", Kokkos::WithoutInitializing),
          faces_per_cell.size());

      Kokkos::deep_copy(face_info[0], inner_faces_info_host);
      Kokkos::deep_copy(face_info[1], boundary_faces_info_host);
      Kokkos::deep_copy(face_info_per_cell, face_info_per_cell_host);

      Kokkos::fence();
    }
    {
      Quadrature<dim - 1> face_quadrature(*quadrature_1d);
      FEFaceValues<dim>   fe_face_values(*mapping,
                                         dof_handler->get_fe(),
                                         face_quadrature,
                                         update_gradients | update_JxW_values |
                                           update_normal_vectors | update_jacobians);
      FEFaceValues<dim>   fe_face_values_neighbor(*mapping,
                                                  dof_handler->get_fe(),
                                                  face_quadrature,
                                                  update_gradients | update_JxW_values |
                                                    update_normal_vectors | update_jacobians);


      jacobians_times_normal_inner_face =
        Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("jacobian_times_normal_inner_face", Kokkos::WithoutInitializing),
          inner_faces_v.size() * n_q_points_face * dim);

      jacobians_times_normal_boundary_face =
        Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("jacobian_times_normal_boundary_face", Kokkos::WithoutInitializing),
          boundary_faces_v.size() * n_q_points_face * dim);

      jxw_inner_face = Kokkos::View<Number *[2], MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("jxw_inner_face", Kokkos::WithoutInitializing),
        inner_faces_v.size() * n_q_points_face);

      jxw_boundary_face = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("jxw_boundary_face", Kokkos::WithoutInitializing),
        boundary_faces_v.size() * n_q_points_face);

      penalty_parameters_inner_face = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("penalty_parameters_inner_face", Kokkos::WithoutInitializing),
        inner_faces_v.size());

      penalty_parameters_boundary_face = Kokkos::View<Number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("penalty_parameters_boundary_face", Kokkos::WithoutInitializing),
        boundary_faces_v.size());

      auto jacobians_times_normal_inner_face_host =
        Kokkos::create_mirror_view(jacobians_times_normal_inner_face);
      auto jacobians_times_normal_boundary_face_host =
        Kokkos::create_mirror_view(jacobians_times_normal_boundary_face);
      auto jxw_inner_face_host    = Kokkos::create_mirror_view(jxw_inner_face);
      auto jxw_boundary_face_host = Kokkos::create_mirror_view(jxw_boundary_face);
      auto penalty_parameters_inner_face_host =
        Kokkos::create_mirror_view(penalty_parameters_inner_face);
      auto penalty_parameters_boundary_face_host =
        Kokkos::create_mirror_view(penalty_parameters_boundary_face);

      for (unsigned int f = 0; f < inner_faces_v.size(); ++f)
        {
          const unsigned int cell_minus = inner_faces_v[f][0];
          const unsigned int cell_plus  = inner_faces_v[f][1];
          const unsigned int f_minus    = inner_faces_v[f][2];
          const unsigned int f_plus     = inner_faces_v[f][3];

          const typename dealii::Triangulation<dim>::cell_iterator cell_it_minus(
            &triangulation,
            cell_level_index[cell_minus].first,
            cell_level_index[cell_minus].second);

          const typename dealii::Triangulation<dim>::cell_iterator cell_it_plus(
            &triangulation, cell_level_index[cell_plus].first, cell_level_index[cell_plus].second);

          // compute penalty factors
          {
            const double extent1 =
              cell_it_minus->measure() / cell_it_minus->face(f_minus)->measure();
            const double extent2 = cell_it_plus->measure() / cell_it_plus->face(f_plus)->measure();

            penalty_parameters_inner_face_host(f) = get_penalty_factor(extent1, extent2);

            fe_face_values.reinit(cell_it_minus, f_minus);
            fe_face_values_neighbor.reinit(cell_it_plus, f_plus);

            for (unsigned int q = 0; q < n_q_points_face; ++q)
              {
                const Tensor<1, dim> n_minus = fe_face_values.normal_vector(q);

                const Tensor<2, dim, Number> inv_jacobian_minus(
                  fe_face_values.jacobian(q).covariant_form());

                const Tensor<2, dim, Number> inv_jacobian_plus(
                  fe_face_values_neighbor.jacobian(q).covariant_form());

                Tensor<1, dim, Number> jac_x_n_minus = inv_jacobian_minus * n_minus;

                Tensor<1, dim, Number> jac_x_n_plus = inv_jacobian_plus * n_minus;

                jxw_inner_face_host(f * n_q_points_face + q, 0) = fe_face_values.JxW(q);

                jxw_inner_face_host(f * n_q_points_face + q, 1) = fe_face_values_neighbor.JxW(q);

                for (unsigned int d = 0; d < dim; d++)
                  {
                    jacobians_times_normal_inner_face_host(f * dim * n_q_points_face +
                                                             d * n_q_points_face + q,
                                                           0) = jac_x_n_minus[d];
                    jacobians_times_normal_inner_face_host(f * dim * n_q_points_face +
                                                             d * n_q_points_face + q,
                                                           1) = jac_x_n_plus[d];
                  }
              }
          }
        }
      for (unsigned int f = 0; f < boundary_faces_v.size(); ++f)
        {
          const unsigned int cell = boundary_faces_v[f][0];
          const unsigned int face = boundary_faces_v[f][2];

          Assert(boundary_faces_v[f][1] == numbers::invalid_unsigned_int, ExcInternalError());

          const typename dealii::Triangulation<dim>::cell_iterator cell_it(
            &triangulation, cell_level_index[cell].first, cell_level_index[cell].second);

          // compute penalty factors
          {
            const double extent = cell_it->measure() / cell_it->face(face)->measure();

            penalty_parameters_boundary_face_host(f) = get_penalty_factor(extent, extent);
          }

          fe_face_values.reinit(cell_it, face);

          for (unsigned int q = 0; q < n_q_points_face; ++q)
            {
              const Tensor<1, dim> n = fe_face_values.normal_vector(q);

              const Tensor<2, dim, Number> inv_jacobian(
                fe_face_values.jacobian(q).covariant_form());

              Tensor<1, dim, Number> jac_x_n = inv_jacobian * n;

              jxw_boundary_face_host(f * n_q_points_face + q) = fe_face_values.JxW(q);

              for (unsigned int d = 0; d < dim; d++)
                {
                  jacobians_times_normal_boundary_face_host(f * dim * n_q_points_face +
                                                            d * n_q_points_face + q) = jac_x_n[d];
                }
            }
        }
      Kokkos::deep_copy(jacobians_times_normal_inner_face, jacobians_times_normal_inner_face_host);
      Kokkos::deep_copy(jacobians_times_normal_boundary_face,
                        jacobians_times_normal_boundary_face_host);
      Kokkos::deep_copy(jxw_inner_face, jxw_inner_face_host);
      Kokkos::deep_copy(jxw_boundary_face, jxw_boundary_face_host);
      Kokkos::deep_copy(penalty_parameters_inner_face, penalty_parameters_inner_face_host);
      Kokkos::deep_copy(penalty_parameters_boundary_face, penalty_parameters_boundary_face_host);

      Kokkos::fence();
    }
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
              shape_data.shape_values,
              shape_data.shape_gradients_collocation,
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

            const unsigned int n_inner_faces = face_info[0].extent(0);

            BK3::DG::compute_inner_faces<dim, fe_degree + 1, n_q_points_1d, Number>(
              shape_data.shape_gradients_collocation,
              jacobians_times_normal_inner_face,
              jxw_inner_face,
              penalty_parameters_inner_face,
              face_values_at_quads[color],
              face_normal_derivatives_at_quads[color],
              face_info[0],
              n_inner_faces,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);

            const unsigned int n_boundary_faces = face_info[1].extent(0);

            BK3::DG::compute_boundary_faces<dim, fe_degree + 1, n_q_points_1d, Number>(
              shape_data.shape_gradients_collocation,
              jacobians_times_normal_boundary_face,
              jxw_boundary_face,
              penalty_parameters_boundary_face,
              face_values_at_quads[color],
              face_normal_derivatives_at_quads[color],
              face_info[1],
              n_boundary_faces,
              n_cells_per_batch,
              n_blocks,
              threads_per_block);



            BK3::DG::distribute_face_to_global<dim, fe_degree + 1, n_q_points_1d, Number>(
              shape_data.shape_values,
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
                const auto triacell = graph[cell_id];

                const typename DoFHandler<dim>::cell_iterator cell =
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