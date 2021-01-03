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

#ifndef TILEDARRAY_CUDA_UM_VECTOR_H__INCLUDED
#define TILEDARRAY_CUDA_UM_VECTOR_H__INCLUDED

#include <TiledArray/cuda/thrust.h>
#include <TiledArray/cuda/um_allocator.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <btas/array_adaptor.h>
#include <btas/varray/varray.h>

#include <TiledArray/cuda/platform.h>
#include <TiledArray/utility.h>

namespace TiledArray {

template <typename T>
using cuda_um_thrust_vector =
    thrust::device_vector<T, TiledArray::cuda_um_allocator<T>>;

/// @return true if @c dev_vec is present in space @space
template <MemorySpace Space, typename Storage>
bool in_memory_space(const Storage& vec) noexcept {
  return overlap(MemorySpace::CUDA_UM, Space);
}

/**
 * @tparam Space
 * @tparam Storage  the Storage type of the vector, such as cuda_um_btas_varray
 */
template <ExecutionSpace Space, typename Storage>
void to_execution_space(Storage& vec, cudaStream_t stream = 0) {
  switch (Space) {
    case ExecutionSpace::CPU: {
      using std::data;
      using std::size;
      using value_type = typename Storage::value_type;
      if (cudaEnv::instance()->concurrent_managed_access()) {
        CudaSafeCall(cudaMemPrefetchAsync(data(vec),
                                          size(vec) * sizeof(value_type),
                                          cudaCpuDeviceId, stream));
      }
      break;
    }
    case ExecutionSpace::CUDA: {
      using std::data;
      using std::size;
      using value_type = typename Storage::value_type;
      int device = -1;
      if (cudaEnv::instance()->concurrent_managed_access()) {
        CudaSafeCall(cudaGetDevice(&device));
        CudaSafeCall(cudaMemPrefetchAsync(
            data(vec), size(vec) * sizeof(value_type), device, stream));
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
 * @param stream cuda stream used to perform prefetch
 */
template <typename Storage>
void make_device_storage(Storage& storage, std::size_t n,
                         const cudaStream_t& stream = 0) {
  storage = Storage(n);
  TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(storage,
                                                                   stream);
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

namespace madness {
namespace archive {

// forward decls
template <class Archive, typename T>
struct ArchiveLoadImpl;
template <class Archive, typename T>
struct ArchiveStoreImpl;

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::cuda_um_thrust_vector<T>> {
  static inline void load(const Archive& ar,
                          TiledArray::cuda_um_thrust_vector<T>& x) {
    typename thrust::device_vector<
        T, TiledArray::cuda_um_allocator<T>>::size_type n(0);
    ar& n;
    x.resize(n);
    for (auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::cuda_um_thrust_vector<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::cuda_um_thrust_vector<T>& x) {
    ar& x.size();
    for (const auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::cuda_um_btas_varray<T>> {
  static inline void load(const Archive& ar,
                          TiledArray::cuda_um_btas_varray<T>& x) {
    typename TiledArray::cuda_um_btas_varray<T>::size_type n(0);
    ar& n;
    x.resize(n);
    for (auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::cuda_um_btas_varray<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::cuda_um_btas_varray<T>& x) {
    ar& x.size();
    for (const auto& xi : x) ar& xi;
  }
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_UM_VECTOR_H__INCLUDED
