/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 */

#ifndef TILEDARRAY_DEVICE_TENSOR_H
#define TILEDARRAY_DEVICE_TENSOR_H

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/external/device.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tile.h>

namespace TiledArray {
namespace detail {

/// `UMTensor` lives in unified memory; the expression engine must route its
/// tile ops through `madness::add_device_task`. The pass-through specs for
/// `Tile<T>` and `LazyArrayTile<T, Op>` in tensor/type_traits.h pick this up.
template <typename T>
struct is_device_tile<TiledArray::UMTensor<T>> : public std::true_type {};

/// Prefetch a UMTensor's storage to the device associated with its tile range.
/// Mirrors the pattern in device/btas_um_tensor.h but reaches the storage via
/// `.data()` + `.total_size()` since `TA::Tensor`'s buffer is a
/// `shared_ptr<T[]>` rather than a varray-like container.
template <typename T>
inline void to_device(const TiledArray::UMTensor<T>& tile) {
  if (tile.empty()) return;
  auto stream = device::stream_for(tile.range());
  if (deviceEnv::instance()->concurrent_managed_access()) {
    DeviceSafeCall(device::memPrefetchAsync(tile.data(),
                                            tile.total_size() * sizeof(T),
                                            stream.device, stream.stream));
  }
}

/// Prefetch a UMTensor's storage back to the host.
template <typename T>
inline void to_host(const TiledArray::UMTensor<T>& tile) {
  if (tile.empty()) return;
  auto stream = device::stream_for(tile.range());
  if (deviceEnv::instance()->concurrent_managed_access()) {
    DeviceSafeCall(device::memPrefetchAsync(tile.data(),
                                            tile.total_size() * sizeof(T),
                                            device::CpuDeviceId, stream.stream));
  }
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_TENSOR_H
