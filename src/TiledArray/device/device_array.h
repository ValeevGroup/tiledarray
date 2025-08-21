/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2025  Virginia Tech
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
 *  Ajay Melekamburath
 *  Department of Chemistry, Virginia Tech
 *  July 31, 2025
 *
 */

#ifndef TILEDARRAY_DEVICE_ARRAY_H
#define TILEDARRAY_DEVICE_ARRAY_H

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE
#include <TiledArray/fwd.h>

#include <TiledArray/device/um_storage.h>
#include <TiledArray/external/device.h>
#include <TiledArray/tile.h>

namespace TiledArray {

/// @brief Array-level to_device operation for DistArrays
/// @tparam UMT Device (UM) Tile type
/// @tparam Policy Policy for DistArray
/// @param um_array input array
template <typename UMT, typename Policy>
void to_device(TiledArray::DistArray<TiledArray::Tile<UMT>, Policy> &um_array) {
  auto to_device_fn = [](TiledArray::Tile<UMT> &tile) {
    auto stream = device::stream_for(tile.range());

    // Check if UMT has storage() method (BTAS-based) or use tensor directly
    // (UMTensor)
    if constexpr (requires { tile.tensor().storage(); }) {
      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
          tile.tensor().storage(), stream);
    } else {
      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
          tile.tensor(), stream);
    }
  };

  auto &world = um_array.world();
  auto start = um_array.pmap()->begin();
  auto end = um_array.pmap()->end();

  for (; start != end; ++start) {
    if (!um_array.is_zero(*start)) {
      world.taskq.add(to_device_fn, um_array.find(*start));
    }
  }

  world.gop.fence();
  DeviceSafeCall(device::deviceSynchronize());
}

/// @brief Array-level to_host operation for DistArrays
/// @tparam UMT Device (UM) Tile type
/// @tparam Policy Policy for DistArray
/// @param um_array input array
template <typename UMT, typename Policy>
void to_host(TiledArray::DistArray<TiledArray::Tile<UMT>, Policy> &um_array) {
  auto to_host_fn = [](TiledArray::Tile<UMT> &tile) {
    auto stream = device::stream_for(tile.range());

    // Check if UMT has storage() method (BTAS-based) or use tensor directly
    // (UMTensor)
    if constexpr (requires { tile.tensor().storage(); }) {
      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Host>(
          tile.tensor().storage(), stream);
    } else {
      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Host>(
          tile.tensor(), stream);
    }
  };

  auto &world = um_array.world();
  auto start = um_array.pmap()->begin();
  auto end = um_array.pmap()->end();

  for (; start != end; ++start) {
    if (!um_array.is_zero(*start)) {
      world.taskq.add(to_host_fn, um_array.find(*start));
    }
  }

  world.gop.fence();
  DeviceSafeCall(device::deviceSynchronize());
}

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_ARRAY_H
