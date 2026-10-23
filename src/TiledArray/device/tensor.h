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
#include <TiledArray/device/blas.h>
#include <TiledArray/external/device.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tile.h>

#include <madness/world/archive.h>
#include <blas.hh>

namespace TiledArray {
namespace detail {

/// UMTensor lives in unified memory; it is identified as a device_tile and
/// the expression engine must route its tile ops through
/// madness::add_device_task.
template <typename T>
struct is_device_tile<TiledArray::UMTensor<T>>
    : public std::bool_constant<TiledArray::detail::is_numeric_v<T>> {};

/// Prefetch a UMTensor's storage to the device associated with its tile range.
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
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
  requires TiledArray::detail::is_numeric_v<T>
inline void to_host(const TiledArray::UMTensor<T>& tile) {
  if (tile.empty()) return;
  auto stream = device::stream_for(tile.range());
  if (deviceEnv::instance()->concurrent_managed_access()) {
    DeviceSafeCall(
        device::memPrefetchAsync(tile.data(), tile.total_size() * sizeof(T),
                                 device::CpuDeviceId, stream.stream));
  }
}

}  // namespace detail

/// dot product: scalar = sum_i arg1[i] * arg2[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline T dot(const UMTensor<T>& arg1, const UMTensor<T>& arg2) {
  TA_ASSERT(!arg1.empty());
  TA_ASSERT(!arg2.empty());
  TA_ASSERT(arg1.nbatch() == 1 && arg2.nbatch() == 1);
  TA_ASSERT(arg1.size() == arg2.size());

  auto& queue = blasqueue_for(arg1.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(arg1);
  detail::to_device(arg2);

  T result(0);
  ::blas::dot(arg1.size(), arg1.data(), 1, arg2.data(), 1, &result, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// scalar = sum_i arg[i] * arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline auto squared_norm(const UMTensor<T>& arg) {
  return dot(arg, arg);
}

/// scalar = sqrt(squared_norm(arg))
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline auto norm(const UMTensor<T>& arg) {
  using std::sqrt;
  using ResultType = TiledArray::detail::scalar_t<T>;
  return static_cast<ResultType>(sqrt(squared_norm(arg)));
}

/// Array-level helpers: bulk to-host / to-device prefetch and conversions
/// between UMTensor-backed and host-Tensor-backed DistArrays.

/// Prefetch every local tile of `array` to the host. Fences on the
/// containing world and globally synchronizes the device on exit.
template <typename T, typename Policy>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_host(TiledArray::DistArray<UMTensor<T>, Policy>& array) {
  auto prefetch = [](UMTensor<T>& tile) {
    auto stream = device::stream_for(tile.range());
    detail::to_host(tile);
    device::sync_madness_task_with(stream);
  };
  auto& world = array.world();
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (!array.is_zero(*it)) world.taskq.add(prefetch, array.find(*it));
  }
  world.gop.fence();
  DeviceSafeCall(device::deviceSynchronize());
}

/// Prefetch every local tile of `array` to the device. Fences on the
/// containing world and globally synchronizes the device on exit.
template <typename T, typename Policy>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_device(TiledArray::DistArray<UMTensor<T>, Policy>& array) {
  auto prefetch = [](UMTensor<T>& tile) {
    auto stream = device::stream_for(tile.range());
    detail::to_device(tile);
    device::sync_madness_task_with(stream);
  };
  auto& world = array.world();
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (!array.is_zero(*it)) world.taskq.add(prefetch, array.find(*it));
  }
  world.gop.fence();
  DeviceSafeCall(device::deviceSynchronize());
}

/// Convert a UMTensor-backed `DistArray` to one backed by a host tile type.
template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<!std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<HostTile, Policy>>
um_tensor_to_ta_tensor(const TiledArray::DistArray<UMTile, Policy>& um_array) {
  auto convert_tile = [](const UMTile& tile) {
    auto stream = device::stream_for(tile.range());
    detail::to_host(tile);
    device::sync_madness_task_with(stream);
    HostTile result(tile.range());
    std::copy_n(tile.data(), tile.total_size(), result.data());
    return result;
  };
  auto out = to_new_tile_type<UMTile>(um_array, convert_tile);
  um_array.world().gop.fence();
  return out;
}

template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<UMTile, Policy>>
um_tensor_to_ta_tensor(const TiledArray::DistArray<UMTile, Policy>& um_array) {
  return um_array;
}

/// Convert a host-tile-backed `DistArray` to a UMTensor-backed one.
template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<!std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<UMTile, Policy>>
ta_tensor_to_um_tensor(
    const TiledArray::DistArray<HostTile, Policy>& host_array) {
  auto convert_tile = [](const HostTile& tile) {
    UMTile result(tile.range());
    std::copy_n(tile.data(), tile.total_size(), result.data());
    detail::to_device(result);
    return result;
  };
  auto out = to_new_tile_type<HostTile>(host_array, convert_tile);
  host_array.world().gop.fence();
  return out;
}

template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<UMTile, Policy>>
ta_tensor_to_um_tensor(
    const TiledArray::DistArray<HostTile, Policy>& host_array) {
  return host_array;
}

}  // namespace TiledArray

/// MADNESS archive specializations for UMTensor.
///
/// `TA::Tensor::serialize(ar)` works on any allocator (the member just walks
/// `data() + range().volume() * nbatch()`), but UM data may be stale on the
/// host if a device kernel is in flight. The Store specialization prefetches
/// the tile back to the host before reading.
namespace madness {
namespace archive {

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::UMTensor<T>& t) {
    if constexpr (TiledArray::detail::is_numeric_v<T>) {
      if (!t.empty()) {
        auto stream = TiledArray::device::stream_for(t.range());
        TiledArray::detail::to_host(t);
        TiledArray::device::sync_madness_task_with(stream);
      }
    }
    const bool empty = t.empty();
    ar & empty;
    if (!empty) {
      ar & t.range();
      ar & t.nbatch();
      ar& madness::archive::wrap(t.data(), t.range().volume() * t.nbatch());
    }
  }
};

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void load(const Archive& ar, TiledArray::UMTensor<T>& t) {
    bool empty = false;
    ar & empty;
    if (!empty) {
      TiledArray::Range range;
      std::size_t nbatch = 1;
      ar & range;
      ar & nbatch;
      t = TiledArray::UMTensor<T>(
          std::move(range), typename TiledArray::UMTensor<T>::nbatches(nbatch));
      ar& madness::archive::wrap(t.data(), t.range().volume() * t.nbatch());
    } else {
      t = TiledArray::UMTensor<T>();
    }
  }
};

}  // namespace archive
}  // namespace madness

/// extern template declarations for the UMTensor class.
namespace TiledArray {

extern template class Tensor<double, device_um_allocator<double>>;
extern template class Tensor<float, device_um_allocator<float>>;
extern template class Tensor<std::complex<double>,
                             device_um_allocator<std::complex<double>>>;
extern template class Tensor<std::complex<float>,
                             device_um_allocator<std::complex<float>>>;
extern template class Tensor<int, device_um_allocator<int>>;
extern template class Tensor<long, device_um_allocator<long>>;

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_TENSOR_H
