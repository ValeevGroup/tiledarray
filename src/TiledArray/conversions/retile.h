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
auto retile_v2(const DistArray<Tile, Policy>& source_array,
               const TiledRange& target_trange) {
  return DistArray<Tile, Policy>(source_array, target_trange);
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
