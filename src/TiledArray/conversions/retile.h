/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  retile.h
 *  May 11, 2020
 */

#ifndef TILEDARRAY_RETILE_H
#define TILEDARRAY_RETILE_H

#include "TiledArray/special/diagonal_array.h"
#include "TiledArray/special/kronecker_delta.h"
#include "TiledArray/util/annotation.h"

/// \name Retile function
/// \brief Retiles a tensor with a provided TiledRange

/// Retiles the data of the input tensor by comparing each dimension of its
/// TiledRange with the corresponding dimension of the input TiledRange. If they
/// are not equivalent, a suitably tiled identity matrix is used to convert that
/// dimension to the desired tiling.
/// \param tensor The tensor whose data is to be retiled
/// \param new_trange The desired TiledRange of the output tensor
/// \return A new tensor with appropriately tiled data

namespace TiledArray {

namespace detail {

template <typename Tile, typename Policy>
auto retile_v0(const DistArray<Tile, Policy>& tensor,
               const TiledRange& new_trange) {
  // Make sure ranks match
  auto rank = new_trange.rank();
  auto tensor_rank = tensor.trange().rank();
  assert((rank == tensor_rank) && "TiledRanges are of different ranks");

  // Makes the annotations for the contraction step
  auto annotations =
      [&](std::size_t target_dim) -> std::tuple<std::string, std::string> {
    std::ostringstream final, switcher;
    switcher << "i" << target_dim << ",iX";
    if (target_dim == 0) {
      final << "iX";
    } else {
      final << "i0";
    }
    for (unsigned int d = 1; d < rank; ++d) {
      if (d == target_dim) {
        final << ",iX";
      } else {
        final << ",i" << d;
      }
    }
    return {final.str(), switcher.str()};
  };

  // Check the different dimensions and contract when needed
  using tensor_type = DistArray<Tile, Policy>;
  auto start = detail::dummy_annotation(rank);
  tensor_type output_tensor;
  for (auto i = 0; i < rank; ++i) {
    if (i == 0) {
      output_tensor(start) = tensor(start);
    }
    if (new_trange.dim(i) != tensor.trange().dim(i)) {
      // Make identity for contraction
      TiledRange retiler{tensor.trange().dim(i), new_trange.dim(i)};
      auto identity = diagonal_array<tensor_type>(tensor.world(), retiler);

      // Make indices for contraction
      auto [finish, change] = annotations(i);

      // Retile
      output_tensor(finish) = output_tensor(start) * identity(change);
    }
  }

  return output_tensor;
}

template <typename Tile, typename Policy>
auto retile_v1(const DistArray<Tile, Policy>& tensor,
               const TiledRange& new_trange) {
  // Make sure ranks match
  auto rank = new_trange.rank();
  auto tensor_rank = tensor.trange().rank();
  assert((rank == tensor_rank) && "TiledRanges are of different ranks");

  // Makes the annotations for the contraction step
  auto annotations = [&]() -> std::tuple<std::string, std::string> {
    std::ostringstream final, switcher;
    final << "j0";
    switcher << "j0";
    for (unsigned int d = 1; d < rank; ++d) {
      final << ",j" << d;
      switcher << ",j" << d;
    }
    for (unsigned int d = 0; d < rank; ++d) {
      switcher << ",i" << d;
    }
    return {final.str(), switcher.str()};
  };

  // Check the different dimensions and contract when needed
  using Array = DistArray<Tile, Policy>;
  container::svector<TiledRange1> retiler_ranges;
  for (auto i = 0; i < rank; ++i) {
    retiler_ranges.emplace_back(new_trange.dim(i));
  }
  for (auto i = 0; i < rank; ++i) {
    retiler_ranges.emplace_back(tensor.trange().dim(i));
  }
  TA::TiledRange retiler_range(retiler_ranges);
  TA::DistArray<KroneckerDeltaTile, Policy> retiler(
      tensor.world(), retiler_range,
      SparseShape(kronecker_shape(retiler_range), retiler_range),
      std::make_shared<detail::ReplicatedPmap>(
          tensor.world(), retiler_range.tiles_range().volume()));
  retiler.init_tiles([=](const TiledArray::Range& range) {
    return KroneckerDeltaTile(range);
  });

  // Make indices for contraction

  // Retile
  Array output;
  auto start = detail::dummy_annotation(rank);
  auto [finish, change] = annotations();
  output(finish) = retiler(change) * tensor(start);

  return output;
}

template <typename Tile, typename Policy>
void write_tile_block(madness::uniqueidT target_array_id,
                      std::size_t target_tile_ord,
                      const Tile& target_tile_contribution) {
  auto* world_ptr = World::world_from_id(target_array_id.get_world_id());
  auto target_array_ptr_opt = world_ptr->ptr_from_id<
      typename DistArray<Tile, Policy>::impl_type::storage_type>(
      target_array_id);
  TA_ASSERT(target_array_ptr_opt);
  TA_ASSERT((*target_array_ptr_opt)->is_local(target_tile_ord));
  (*target_array_ptr_opt)
      ->get_local(target_tile_ord)
      .get()
      .block(target_tile_contribution.range()) = target_tile_contribution;
}

template <typename Tile, typename Policy>
auto retile_v2(const DistArray<Tile, Policy>& source_array,
               const TiledRange& target_trange) {
  auto& world = source_array.world();
  const auto rank = source_array.trange().rank();
  TA_ASSERT(rank == target_trange.rank());

  // compute metadata
  // - list of target tile indices and the corresponding Range1 for each 1-d
  // source tile
  using target_tiles_t = std::vector<std::pair<TA_1INDEX_TYPE, Range1>>;
  using mode_target_tiles_t = std::vector<target_tiles_t>;
  using all_target_tiles_t = std::vector<mode_target_tiles_t>;

  all_target_tiles_t all_target_tiles(target_trange.rank());
  // for each mode ...
  for (auto d = 0; d != target_trange.rank(); ++d) {
    mode_target_tiles_t& mode_target_tiles = all_target_tiles[d];
    auto& target_tr1 = target_trange.dim(d);
    auto& target_element_range = target_tr1.elements_range();
    // ... and each tile in that mode ...
    for (auto&& source_tile : source_array.trange().dim(d)) {
      mode_target_tiles.emplace_back();
      auto& target_tiles = mode_target_tiles.back();
      auto source_tile_lo = source_tile.lobound();
      auto source_tile_up = source_tile.upbound();
      auto source_element_idx = source_tile_lo;
      // ... find all target tiles what overlap with it
      if (target_element_range.overlaps_with(source_tile)) {
        while (source_element_idx < source_tile_up) {
          if (target_element_range.includes(source_element_idx)) {
            auto target_tile_idx =
                target_tr1.element_to_tile(source_element_idx);
            auto target_tile = target_tr1.tile(target_tile_idx);
            auto target_lo =
                std::max(source_element_idx, target_tile.lobound());
            auto target_up = std::min(source_tile_up, target_tile.upbound());
            target_tiles.emplace_back(target_tile_idx,
                                      Range1(target_lo, target_up));
            source_element_idx = target_up;
          } else if (source_element_idx < target_element_range.lobound()) {
            source_element_idx = target_element_range.lobound();
          } else if (source_element_idx >= target_element_range.upbound())
            break;
        }
      }
    }
  }

  // estimate the shape, if sparse
  // use max value for each nonzero tile, then will recompute after tiles are
  // assigned
  using shape_type = typename Policy::shape_type;
  shape_type target_shape;
  const auto& target_tiles_range = target_trange.tiles_range();
  if constexpr (!is_dense_v<Policy>) {
    // each rank computes contributions to the shape norms from its local tiles
    Tensor<float> target_shape_norms(target_tiles_range, 0);
    auto& source_trange = source_array.trange();
    const auto e = source_array.end();
    for (auto it = source_array.begin(); it != e; ++it) {
      auto source_tile_idx = it.index();

      // make range for iterating over all possible target tile idx combinations
      TA::Index target_tile_ord_extent_range(rank);
      for (auto d = 0; d != rank; ++d) {
        target_tile_ord_extent_range[d] =
            all_target_tiles[d][source_tile_idx[d]].size();
      }

      // loop over every target tile combination
      TA::Range target_tile_ord_extent(target_tile_ord_extent_range);
      for (auto& target_tile_ord : target_tile_ord_extent) {
        TA::Index target_tile_idx(rank);
        for (auto d = 0; d != rank; ++d) {
          target_tile_idx[d] =
              all_target_tiles[d][source_tile_idx[d]][target_tile_ord[d]].first;
        }
        target_shape_norms(target_tile_idx) = std::numeric_limits<float>::max();
      }
    }
    world.gop.max(target_shape_norms.data(), target_shape_norms.size());
    target_shape = SparseShape(target_shape_norms, target_trange);
  }

  using Array = DistArray<Tile, Policy>;
  Array target_array(source_array.world(), target_trange, target_shape);
  target_array.fill_local(0.0);
  target_array.world().gop.fence();

  // loop over local tile and sends its contributions to the targets
  {
    auto& source_trange = source_array.trange();
    const auto e = source_array.end();
    auto& target_tiles_range = target_trange.tiles_range();
    for (auto it = source_array.begin(); it != e; ++it) {
      const auto& source_tile = *it;
      auto source_tile_idx = it.index();

      // make range for iterating over all possible target tile idx combinations
      TA::Index target_tile_ord_extent_range(rank);
      for (auto d = 0; d != rank; ++d) {
        target_tile_ord_extent_range[d] =
            all_target_tiles[d][source_tile_idx[d]].size();
      }

      // loop over every target tile combination
      TA::Range target_tile_ord_extent(target_tile_ord_extent_range);
      for (auto& target_tile_ord : target_tile_ord_extent) {
        TA::Index target_tile_idx(rank);
        container::svector<TA::Range1> target_tile_rngs1(rank);
        for (auto d = 0; d != rank; ++d) {
          std::tie(target_tile_idx[d], target_tile_rngs1[d]) =
              all_target_tiles[d][source_tile_idx[d]][target_tile_ord[d]];
        }
        TA_ASSERT(source_tile.future().probe());
        Tile target_tile_contribution(
            source_tile.get().block(target_tile_rngs1));
        auto target_tile_idx_ord = target_tiles_range.ordinal(target_tile_idx);
        auto target_proc = target_array.pmap()->owner(target_tile_idx_ord);
        world.taskq.add(target_proc, &write_tile_block<Tile, Policy>,
                        target_array.id(), target_tile_idx_ord,
                        target_tile_contribution);
      }
    }
  }
  // data is mutated in place, so must wait for all tasks to complete
  target_array.world().gop.fence();
  // recompute norms/trim away zeros
  target_array.truncate();

  return target_array;
}

}  // namespace detail

/// Creates a new DistArray with the same data as the input tensor, but with a
/// different trange. The primary use-case is to change tiling while keeping the
/// element range the same, but it can be used to select blocks of the data as
/// well as increasing the element range (with the new elements initialized to
/// zero)
/// \param array The DistArray whose data is to be retiled
/// \param target_trange The desired TiledRange of the output tensor
template <typename Tile, typename Policy>
auto retile(const DistArray<Tile, Policy>& array,
            const TiledRange& target_trange) {
  return detail::retile_v0(array, target_trange);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_RETILE_H
