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

#include <TiledArray/conversions/to_new_tile_type.h>
#include <TiledArray/device/dispatch.h>
#include <TiledArray/external/device.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tile.h>

namespace TiledArray {
namespace detail {

/// prefetches the \p n elements at \p p to \p device_id
/// (device::CpuDeviceId for the host) on \p stream, if the device supports it
template <typename T>
inline void prefetch(const T* p, std::size_t n, int device_id,
                     device::stream_t stream) {
  if (n == 0) return;
  if (deviceEnv::instance()->concurrent_managed_access())
    DeviceSafeCall(
        device::memPrefetchAsync(p, n * sizeof(T), device_id, stream));
}

/// prefetches a UMTensor's storage to the device of the stream its range
/// maps to
/// \return that stream
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline device::Stream to_device(const TiledArray::UMTensor<T>& tile) {
  const auto stream = device::stream_for(tile.range());
  if (!tile.empty())
    prefetch(tile.data(), tile.total_size(), stream.device, stream.stream);
  return stream;
}

/// prefetches a UMTensor's storage back to the host
/// \return the stream the prefetch is queued on
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline device::Stream to_host(const TiledArray::UMTensor<T>& tile) {
  const auto stream = device::stream_for(tile.range());
  if (!tile.empty())
    prefetch(tile.data(), tile.total_size(), device::CpuDeviceId,
             stream.stream);
  return stream;
}

/// runs \p prefetch_tile on every local nonzero tile of \p array and returns
/// once all of them have completed; fences the array's world
template <typename T, typename Policy, typename Prefetch>
inline void prefetch_local_tiles(
    TiledArray::DistArray<UMTensor<T>, Policy>& array, Prefetch prefetch_tile) {
  // a plain task: sync_madness_task_with blocks until the prefetch completes
  auto task = [prefetch_tile](UMTensor<T>& tile) {
    device::sync_madness_task_with(prefetch_tile(tile));
  };
  auto& world = array.world();
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (!array.is_zero(*it)) world.taskq.add(task, array.find(*it));
  }
  world.gop.fence();
}

}  // namespace detail

/// Array-level helpers: bulk to-host / to-device prefetch and conversions
/// between UMTensor-backed and host-Tensor-backed DistArrays.

/// Prefetch every local tile of `array` to the host. Fences on the
/// containing world; every prefetch has completed on return.
template <typename T, typename Policy>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_host(TiledArray::DistArray<UMTensor<T>, Policy>& array) {
  detail::prefetch_local_tiles(
      array, [](const UMTensor<T>& tile) { return detail::to_host(tile); });
}

/// Prefetch every local tile of `array` to the device. Fences on the
/// containing world; every prefetch has completed on return.
template <typename T, typename Policy>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_device(TiledArray::DistArray<UMTensor<T>, Policy>& array) {
  detail::prefetch_local_tiles(
      array, [](const UMTensor<T>& tile) { return detail::to_device(tile); });
}

/// Convert a UMTensor-backed `DistArray` to one backed by a host tile type.
template <typename UMTile, typename HostTile, typename Policy>
inline TiledArray::DistArray<HostTile, Policy> um_tensor_to_ta_tensor(
    const TiledArray::DistArray<UMTile, Policy>& um_array) {
  // a no-op if the tile types are the same (e.g. in callers generic over the
  // tile type)
  if constexpr (std::is_same_v<UMTile, HostTile>)
    return um_array;
  else {
    auto convert_tile = [](const UMTile& tile) {
      device::sync_madness_task_with(detail::to_host(tile));
      HostTile result(tile.range(), typename HostTile::nbatches(tile.nbatch()));
      std::copy_n(tile.data(), tile.total_size(), result.data());
      return result;
    };
    auto out = to_new_tile_type<UMTile>(um_array, convert_tile);
    um_array.world().gop.fence();
    return out;
  }
}

/// Convert a host-tile-backed `DistArray` to a UMTensor-backed one.
template <typename UMTile, typename HostTile, typename Policy>
inline TiledArray::DistArray<UMTile, Policy> ta_tensor_to_um_tensor(
    const TiledArray::DistArray<HostTile, Policy>& host_array) {
  // a no-op if the tile types are the same
  if constexpr (std::is_same_v<UMTile, HostTile>)
    return host_array;
  else {
    auto convert_tile = [](const HostTile& tile) {
      UMTile result(tile.range(), typename UMTile::nbatches(tile.nbatch()));
      std::copy_n(tile.data(), tile.total_size(), result.data());
      detail::to_device(result);
      return result;
    };
    auto out = to_new_tile_type<HostTile>(host_array, convert_tile);
    host_array.world().gop.fence();
    return out;
  }
}

}  // namespace TiledArray

// N.B. archiving a UMTensor needs no specialization: Tensor::serialize
// waits for device work and prefetches the data to the host before storing.
// The extern template declarations for UMTensor live in tensor/tensor.h.

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_TENSOR_H
