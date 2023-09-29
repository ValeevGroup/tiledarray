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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  clone.h
 *  Nov 29, 2015
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_CLONE_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_CLONE_H__INCLUDED

#ifdef TILEDARRAY_HAS_DEVICE
#include "TiledArray/device/device_task_fn.h"
#endif

namespace TiledArray {

/// Forward declarations
template <typename, typename>
class DistArray;
class DensePolicy;
class SparsePolicy;

/// Create a deep copy of an array

/// \tparam Tile The tile type of the array
/// \tparam Policy The policy of the array
/// \param arg The array to be cloned
template <typename Tile, typename Policy>
inline DistArray<Tile, Policy> clone(const DistArray<Tile, Policy>& arg) {
  typedef typename DistArray<Tile, Policy>::value_type value_type;

  World& world = arg.world();

  // Make an empty result array
  DistArray<Tile, Policy> result(world, arg.trange(), arg.shape(), arg.pmap());

  // Iterate over local tiles of arg
  for (auto index : *arg.pmap()) {
    if (arg.is_zero(index)) continue;

    // Spawn a task to clone the tiles

    Future<value_type> tile;
    if constexpr (!detail::is_device_tile_v<value_type>) {
      tile = world.taskq.add(
          [](const value_type& tile) -> value_type {
            using TiledArray::clone;
            return clone(tile);
          },
          arg.find(index));
    } else {
#ifdef TILEDARRAY_HAS_DEVICE
      tile = madness::add_device_task(
          world,
          [](const value_type& tile) -> value_type {
            using TiledArray::clone;
            return clone(tile);
          },
          arg.find(index));
#else
      abort();  // unreachable
#endif
    }

    // Store result tile
    result.set(index, tile);
  }

  return result;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_CLONE_H__INCLUDED
