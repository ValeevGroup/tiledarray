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

  /// Function to convert an array to a new array with a different tile type.

  /// \tparam Tile The array tile type
  /// \tparam Policy The array policy type
  /// \tparam Op The tile conversion operation type
  /// \param array The array to be converted
  /// \param op The tile type conversion operation
  template <typename Tile, typename Policy, typename Op>
  inline DistArray<typename std::result_of<Op(Tile)>::type, Policy>
  to_new_tile_type(DistArray<Tile, Policy> const &old_array, Op &&op) {
    using OutTileType = typename std::result_of<Op(Tile)>::type;
    using OutArray = DistArray<OutTileType, Policy>;

    static_assert(!std::is_same<Tile, OutTileType>::value,
        "Can't call new tile type if tile type does not change.");

    auto &world = old_array.world();

    // Create new array
    OutArray new_array(world, old_array.trange(), old_array.shape(), old_array.pmap());

    using pmap_iter = decltype(old_array.pmap()->begin());
    pmap_iter it = old_array.pmap()->begin();
    pmap_iter end = old_array.pmap()->end();

    for(; it != end; ++it) {
      // Must check for zero because pmap_iter does not.
      if(!old_array.is_zero(*it)) {
        // Spawn a task to evaluate the tile
        Future<OutTileType> tile = world.taskq.add(op, old_array.find(*it));
        new_array.set(*it, tile);
      }
    }

    return new_array;
  }

} // namespace TiledArray
#endif // TILEDARRAY_CONVERSIONS_TO_NEW_TILE_TYPE_H__INCLUDED
