/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *  Feb 6, 2018
 *
 */

#ifndef TILEDARRAY_DEVICE_UM_VECTOR_H__INCLUDED
#define TILEDARRAY_DEVICE_UM_VECTOR_H__INCLUDED

#include <TiledArray/external/device.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <btas/array_adaptor.h>
#include <btas/varray/varray.h>

#include <TiledArray/device/platform.h>
#include <TiledArray/utility.h>

#include <madness/world/archive.h>

namespace TiledArray {

/// @return true if @c dev_vec is present in space @space
template <MemorySpace Space, typename Storage>
bool in_memory_space(const Storage& vec) noexcept {
  return overlap(MemorySpace::Device_UM, Space);
}

/**
 * @tparam Space
 * @tparam Storage  the Storage type of the vector, such as
 * device_um_btas_varray
 */
template <ExecutionSpace Space, typename Storage>
void to_execution_space(Storage& vec, const device::Stream& s) {
  switch (Space) {
    case ExecutionSpace::Host: {
      using std::data;
      using std::size;
      using value_type = typename Storage::value_type;
      if (deviceEnv::instance()->concurrent_managed_access()) {
        DeviceSafeCall(device::memPrefetchAsync(data(vec),
                                                size(vec) * sizeof(value_type),
                                                device::CpuDeviceId, s.stream));
      }
      break;
    }
    case ExecutionSpace::Device: {
      using std::data;
      using std::size;
      using value_type = typename Storage::value_type;
      if (deviceEnv::instance()->concurrent_managed_access()) {
        DeviceSafeCall(device::memPrefetchAsync(
            data(vec), size(vec) * sizeof(value_type), s.device, s.stream));
      }
      break;
    }
    default:
      throw std::runtime_error("invalid execution space");
  }
}

/**
 * create UM storage and prefetch it to device
 *
 * @param storage UM Storage type object
 * @param n size of um storage object
 * @param stream device stream used to perform prefetch
 */
template <typename Storage>
void make_device_storage(Storage& storage, std::size_t n,
                         const device::Stream& s) {
  storage = Storage(n);
  TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(storage,
                                                                     s);
}

/**
 *  return the device pointer for UM storage object
 *
 * @param storage UM Storage type object
 * @return data pointer of UM Storage object
 */
template <typename Storage>
typename Storage::value_type* device_data(Storage& storage) {
  return storage.data();
}

/**
 *  return the const pointer for UM storage object
 *
 * @param storage UM Storage type object
 * @return const data pointer of UM Storage object
 */
template <typename Storage>
const typename Storage::value_type* device_data(const Storage& storage) {
  return storage.data();
}

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_DEVICE_UM_VECTOR_H__INCLUDED
