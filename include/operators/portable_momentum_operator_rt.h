#ifndef portable_momentum_operator_rt_h
#define portable_momentum_operator_rt_h

#include <deal.II/base/enable_observer_pointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

#include "matrix_free/portable_shape_info.h"



DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  namespace RT
  {
    template <int dim, typename Number = double>
    class RaviartThomasOperatorBase : public EnableObserverPointer
    {
    public:
      using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;

      using KokkosScratchSpace =
        MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space;

      using ViewValues =
        Kokkos::View<Number *, KokkosScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      RaviartThomasOperatorBase() = default;


      template <typename OtherNumber>
      void
      reinit(const Mapping<dim>                   &mapping,
             const DoFHandler<dim>                &dof_handler,
             const AffineConstraints<OtherNumber> &constraints,
             const Quadrature<1>                  &quadrature)
      {
        this->mapping     = &mapping;
        this->dof_handler = &dof_handler;
        this->quadrature  = &quadrature;

        const FiniteElement<dim> &fe = dof_handler.get_fe();


        AssertThrow(dynamic_cast<const FE_RaviartThomasNodal<dim> *>(&fe),
                    ExcMessage("This class only works for Raviart-Thomas elements."));


        this->shape_info_cpu.reinit(quadrature, fe);

        AssertDimension(shape_info_cpu.data.size(), 2);

        this->shape_info[0].reinit(shape_info_cpu.data[0]);
        this->shape_info[1].reinit(shape_info_cpu.data[1]);

        {
          const MPI_Comm comm = dof_handler.get_mpi_communicator();

          // calculate the number of cells
          unsigned int n_cells = 0;
          for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              ++n_cells;

          IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

          unsigned int n_constrained_locally_owned_dofs = 0;
          for (const auto &line : constraints.get_lines())
            if (locally_owned_dofs.is_element(line.index))
              ++n_constrained_locally_owned_dofs;

          const types::global_dof_index n_unconstrained_owned_dofs =
            locally_owned_dofs.n_elements() - n_constrained_locally_owned_dofs;

          // Assign DoF numbers for the unconstrained degrees of freedom only
          std::pair<types::global_dof_index, types::global_dof_index> positions =
            Utilities::MPI::partial_and_total_sum(n_unconstrained_owned_dofs, comm);

          std::vector<types::global_dof_index> dof_numbers(locally_owned_dofs.n_elements(),
                                                           numbers::invalid_dof_index);
          types::global_dof_index              counter = positions.first;
          for (unsigned int i = 0; i < n_unconstrained_owned_dofs; ++i)
            dof_numbers[i] = counter++;

          // Extract ghost entries for which we need to query the numbers from
          // remote processes - we do not know the start indices of the
          // respective processes even though we could query the DoF index owner
          // through the triangulation, so we need to perform a lookup anyway
          // and do that by a ghost exchange of all data. While there, also
          // extract a compressed representation, taking the first number on
          // each entity (face, cell) in global index space, which we later
          // translate to local numbers


          std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
          std::vector<types::global_dof_index> lexicographic_dof_indices(fe.dofs_per_cell);

          const std::vector<unsigned int> &lexicographic_numbering =
            shape_info_cpu.lexicographic_numbering;

          // We store the start of the indices per each geometric entity (first
          // 2*dim faces and then the cell dofs).
          std::vector<std::array<types::global_dof_index, 2 * dim + 1>> dof_indices_per_entity(
            n_cells);

          std::vector<types::global_dof_index> ghost_indices;

          unsigned int cell_counter = 0;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(local_dof_indices);

                  // Adjust dof indices due to periodicity
                  for (types::global_dof_index &a : local_dof_indices)
                    {
                      const auto line = constraints.get_constraint_entries(a);
                      if (line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
                        a = (*line)[0].first;
                    }

                  for (types::global_dof_index a : local_dof_indices)
                    if (!locally_owned_dofs.is_element(a) && !constraints.is_constrained(a))
                      ghost_indices.push_back(a);

                  const unsigned int dofs_per_face = fe.dofs_per_face;

                  for (unsigned int f = 0; f < 2 * dim; ++f)
                    {
                      for (unsigned int i = 0; i < dofs_per_face; ++i)
                        AssertThrow(local_dof_indices[f * dofs_per_face + i] ==
                                      local_dof_indices[f * dofs_per_face] + i,
                                    ExcInternalError());

                      dof_indices_per_entity[cell_counter][f] =
                        local_dof_indices[f * dofs_per_face];
                    }

                  const unsigned int start_cell_dofs = 2 * dim * dofs_per_face;
                  for (unsigned int i = 0; i < fe.dofs_per_cell - start_cell_dofs; ++i)
                    AssertThrow(local_dof_indices[start_cell_dofs + i] ==
                                  local_dof_indices[start_cell_dofs] + i,
                                ExcInternalError());

                  dof_indices_per_entity[cell_counter][2 * dim] =
                    local_dof_indices[start_cell_dofs];

                  ++cell_counter;
                }
            }

          IndexSet ghost_index_set(locally_owned_dofs.size());
          ghost_index_set.add_indices(ghost_indices.begin(), ghost_indices.end());
          ghost_index_set.compress();
          Utilities::MPI::Partitioner partitioner_dofs(locally_owned_dofs, ghost_index_set, comm);

          std::vector<types::global_dof_index> tmp_array(partitioner_dofs.n_import_indices());
          std::vector<types::global_dof_index> numbers_ghosts(partitioner_dofs.n_ghost_indices());
          std::vector<MPI_Request>             requests;
          partitioner_dofs.export_to_ghosted_array_start(3,
                                                         make_const_array_view(dof_numbers),
                                                         make_array_view(tmp_array),
                                                         make_array_view(numbers_ghosts),
                                                         requests);
          partitioner_dofs.export_to_ghosted_array_finish(make_array_view(numbers_ghosts),
                                                          requests);

          IndexSet owned_dofs(positions.second);
          owned_dofs.add_range(positions.first, positions.first + n_unconstrained_owned_dofs);
          std::vector<types::global_dof_index> compressed_ghost;
          compressed_ghost.reserve(numbers_ghosts.size());
          for (const types::global_dof_index a : numbers_ghosts)
            if (a != numbers::invalid_dof_index)
              compressed_ghost.push_back(a);
          IndexSet ghost_dofs(positions.second);
          ghost_dofs.add_indices(compressed_ghost.begin(), compressed_ghost.end());
          ghost_dofs.compress();

          partitioner = std::make_shared<Utilities::MPI::Partitioner>(owned_dofs, ghost_dofs, comm);


          dof_indices = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dof_indices", Kokkos::WithoutInitializing), 2 * dim + 1, n_cells);

          neighbor_cells = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("neighbor_cells", Kokkos::WithoutInitializing), 2 * dim, n_cells);

          auto dof_indices_host = Kokkos::create_mirror_view(dof_indices);

          auto neighbor_cells_host = Kokkos::create_mirror_view(neighbor_cells);

          dof_indices_per_cell = Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dof_indices_per_cell", Kokkos::WithoutInitializing),
            fe.dofs_per_cell,
            n_cells);

          auto dof_indices_per_cell_host = Kokkos::create_mirror_view(dof_indices_per_cell);

          {
            for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
              {
                for (unsigned int i = 0; i < 2 * dim; ++i)
                  {
                    neighbor_cells_host(i, cell_id) = numbers::invalid_unsigned_int;
                    dof_indices_host(i, cell_id)    = numbers::invalid_unsigned_int;
                  }

                for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                  dof_indices_per_cell_host(i, cell_id) = numbers::invalid_unsigned_int;

                dof_indices_host(2 * dim, cell_id) = numbers::invalid_unsigned_int;
              }
          }

          std::array<std::vector<unsigned int>, 2 * dim> cells_at_dirichlet_boundary_by_face;
          std::vector<unsigned int> cell_indices(dof_handler.get_triangulation().n_active_cells(),
                                                 numbers::invalid_unsigned_int);

          cell_counter = 0;
          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  for (unsigned int f = 0; f < 2 * dim + 1; ++f)
                    {
                      const types::global_dof_index index = dof_indices_per_entity[cell_counter][f];
                      const unsigned int      my_index    = partitioner_dofs.global_to_local(index);
                      types::global_dof_index number_compressed;

                      if (my_index < locally_owned_dofs.n_elements())
                        number_compressed = dof_numbers[my_index];
                      else
                        number_compressed = numbers_ghosts[my_index - dof_numbers.size()];

                      Assert(number_compressed != numbers::invalid_dof_index ||
                               constraints.is_constrained(index),
                             ExcInternalError());

                      if (number_compressed != numbers::invalid_dof_index)
                        dof_indices_host(f, cell_counter) =
                          partitioner->global_to_local(number_compressed);
                      else
                        dof_indices_host(f, cell_counter) = numbers::invalid_unsigned_int;
                    }

                  cell_indices[cell->active_cell_index()] = cell_counter;


                  for (unsigned int f = 0; f < 2 * dim; ++f)
                    if (dof_indices_host(f, cell_counter) == numbers::invalid_unsigned_int)
                      cells_at_dirichlet_boundary_by_face[f].push_back(cell_counter);


                  cell->get_dof_indices(local_dof_indices);

                  // Adjust dof indices due to periodicity
                  for (types::global_dof_index &a : local_dof_indices)
                    {
                      const auto line = constraints.get_constraint_entries(a);
                      if (line != nullptr && line->size() == 1 && (*line)[0].second == Number(1.0))
                        a = (*line)[0].first;
                    }

                  if (partitioner)
                    for (auto &index : local_dof_indices)
                      index = partitioner->global_to_local(index);

                  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    {
                      const types::global_dof_index dof_index =
                        local_dof_indices[lexicographic_numbering[i]];

                      if (constraints.is_constrained(dof_index))
                        dof_indices_per_cell_host(i, cell_counter) = numbers::invalid_unsigned_int;
                      else
                        dof_indices_per_cell_host(i, cell_counter) = dof_index;
                    }

                  ++cell_counter;
                }
            }

          // put cells at Dirichlet boundary together in one data structure
          unsigned int n_cells_at_dirichlet_boundary = 0;
          for (unsigned int f = 0; f < 2 * dim; ++f)
            n_cells_at_dirichlet_boundary += cells_at_dirichlet_boundary_by_face[f].size();

          std::map<unsigned int, std::vector<std::array<types::global_dof_index, 5>>>
            proc_neighbors;

          cell_counter = 0;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  for (unsigned int f = 0; f < 2 * dim; ++f)
                    {
                      const bool at_boundary = cell->at_boundary(f);
                      const bool has_periodic_neighbor =
                        at_boundary && cell->has_periodic_neighbor(f);

                      if (at_boundary == false || has_periodic_neighbor)
                        {
                          const auto neighbor =
                            has_periodic_neighbor ? cell->periodic_neighbor(f) : cell->neighbor(f);

                          if (neighbor->is_locally_owned())
                            {
                              AssertIndexRange(neighbor->active_cell_index(), cell_indices.size());
                              neighbor_cells_host(f, cell_counter) =
                                cell_indices[neighbor->active_cell_index()];
                            }
                          else
                            {
                              std::array<types::global_dof_index, 5> neighbor_data;
                              neighbor_data[0] = cell_counter;
                              neighbor_data[1] = has_periodic_neighbor ?
                                                   cell->periodic_neighbor_face_no(f) :
                                                   cell->neighbor_face_no(f);
                              neighbor_data[2] = cell->global_active_cell_index();
                              neighbor_data[3] = neighbor->global_active_cell_index();
                              neighbor_data[4] = f;
                              proc_neighbors[neighbor->subdomain_id()].push_back(neighbor_data);
                              // set dummy
                              neighbor_cells_host(f, cell_counter) =
                                cell_indices[cell->active_cell_index()];
                            }
                        }
                    }

                  ++cell_counter;
                }
            }
          Kokkos::deep_copy(dof_indices, dof_indices_host);
          Kokkos::fence();

          Kokkos::deep_copy(neighbor_cells, neighbor_cells_host);
          Kokkos::fence();

          Kokkos::deep_copy(dof_indices_per_cell, dof_indices_per_cell_host);
          Kokkos::fence();
        }
      }

      void
      compute_geometric_tensor(const Mapping<dim>    &mapping,
                               const Quadrature<1>   &quadrature_1d,
                               const DoFHandler<dim> &dof_handler,
                               const unsigned int     n_cells)
      {
        const unsigned int n_q_points_1d = quadrature_1d.size();
        const unsigned int n_q_points    = Utilities::pow(n_q_points_1d, dim);

        constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

        geometric_tensor_mass =
          DeviceVector<Number>(Kokkos::view_alloc("geometric_tensor_mass",
                                                  Kokkos::WithoutInitializing),
                               n_cells * symmetric_tensor_dim * n_q_points);

        auto geometric_tensor_mass_host = Kokkos::create_mirror_view(geometric_tensor_mass);

        // build tensor product quadrature
        Quadrature<dim> quadrature(quadrature_1d);

        std::vector<Number> quad_weights = quadrature.get_weights();

        const FiniteElement<dim> &fe = dof_handler.get_fe();

        FEValues<dim> fe_values(mapping,
                                fe,
                                quadrature,
                                update_jacobians | update_jacobian_grads |
                                  update_quadrature_points);

        unsigned cell_counter = 0;
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);

                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const DerivativeForm<1, dim, dim> jacobian = fe_values.jacobian(q);

                    Number components[symmetric_tensor_dim];

                    // we only need the symmetric part of the tensor, so we store it
                    // in a compressed format
                    int index = 0;
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = d1; d2 < dim; ++d2)
                        {
                          Number sum = 0;
                          for (int k = 0; k < dim; ++k)
                            sum += jacobian[k][d1] * jacobian[k][d2];
                          components[index] = sum * quad_weights[q];
                          ++index;
                        }

                    for (unsigned int c = 0; c < symmetric_tensor_dim; ++c)
                      geometric_tensor_mass_host(cell_counter * symmetric_tensor_dim * n_q_points +
                                                 c * n_q_points + q) =
                        components[c] / jacobian.determinant() * quad_weights[q];
                  }
                ++cell_counter;
              }
          }

        Kokkos::deep_copy(geometric_tensor_mass, geometric_tensor_mass_host);
        Kokkos::fence();
      }

      //   template <int nq, int nm_t>
      //   void
      //   compute_cell_mass_operator(const DeviceVector<Number> shape_values_n,
      //   const  DeviceVector<Number> shape_values_t,
      //   const
      // const Number * __restrict__ )
      //   {
      //     constexpr int nm_n = nm_t + 1;


      //   }

    private:
      enum class CellOperation
      {
        helmholtz,
        convective,
        convective_and_divergence
      };

      ObserverPointer<const Mapping<dim>>    mapping;
      ObserverPointer<const DoFHandler<dim>> dof_handler;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> dof_indices_per_cell;

      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space> neighbor_cells;

      std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

      mutable std::array<double, 15> timings;

      dealii::internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info_cpu;

      Kokkos::Array<internal::UnivariateShapeData<Number>, 2> shape_info;

      ObserverPointer<const Quadrature<1>> quadrature;

      DeviceVector<Number> geometric_tensor_mass;
    };

  } // namespace RT

} // namespace Portable


DEAL_II_NAMESPACE_CLOSE

#endif