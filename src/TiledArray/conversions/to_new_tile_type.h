/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  Drew Lewis
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_to_dense.h
 *  Jan 22, 2015
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_TO_NEW_TILE_TYPE_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_TO_NEW_TILE_TYPE_H__INCLUDED

#include "../dist_array.h"

namespace TiledArray {

namespace detail {
template <typename DstTile, typename SrcTile, typename Op,
          typename Enabler = void>
struct cast_then_op;

template <typename Tile, typename Op>
struct cast_then_op<Tile, Tile, Op, void> {
  cast_then_op(const Op& op) : op_(op) {}

  auto operator()(const Tile& tile) const { return op_(tile); }

  Op op_;
};

template <typename DstTile, typename SrcTile, typename Op>
struct cast_then_op<DstTile, SrcTile, Op,
                    std::enable_if_t<!std::is_same<DstTile, SrcTile>::value>> {
  cast_then_op(const Op& op) : op_(op) {}

  auto operator()(const SrcTile& tile) const {
    return op_(Cast<DstTile, SrcTile>{}(tile));
  }

  Op op_;
};

}  // namespace detail

/// Function to convert an array to a new array with a different tile type.

/// \tparam Tile The array tile type
/// \tparam ConvTile The tile type to which Tile can be converted to and
///                  for which \c Op(ConvTile) is well-formed
/// \tparam Policy The array policy type
/// \tparam Op The tile conversion operation type
/// \param array The array to be converted
/// \param op The tile type conversion operation
template <typename Tile, typename ConvTile = Tile, typename Policy, typename Op>
inline decltype(auto) to_new_tile_type(DistArray<Tile, Policy> const& old_array,
                                       Op&& op) {
  using OutTileType = std::invoke_result_t<Op, ConvTile>;

  static_assert(!std::is_same<Tile, OutTileType>::value,
                "Can't call new tile type if tile type does not change.");

  const detail::cast_then_op<ConvTile, Tile, std::remove_reference_t<Op>>
      cast_op(std::forward<Op>(op));

  return foreach<OutTileType>(
      old_array, [cast_op](auto& out, const auto& in) { out = cast_op(in); });
}

}  // namespace TiledArray
#endif  // TILEDARRAY_CONVERSIONS_TO_NEW_TILE_TYPE_H__INCLUDED
