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
 *  Mar 16, 2018
 *
 */

#ifndef TILEDARRAY_DEVICE_THRUST_H__INCLUDED
#define TILEDARRAY_DEVICE_THRUST_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#ifdef TILEDARRAY_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

// rocthrust headers rely on THRUST_DEVICE_SYSTEM being defined, which is only
// defined by the HIP-specific compilers to be usable with host compiler define
// it here explicitly
#ifdef TILEDARRAY_HAS_HIP
#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_HIP
#endif
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// thrust::device_vector::data() returns a proxy, provide an overload for
// std::data() to provide raw ptr
namespace thrust {

// thrust::device_malloc_allocator name changed to device_allocator after
// version 10
#ifdef TILEDARRAY_HAS_CUDA
#if CUDART_VERSION < 10000
template <typename T>
using device_allocator = thrust::device_malloc_allocator<T>;
#endif
#endif  // TILEDARRAY_HAS_CUDA

template <typename T, typename Alloc>
const T* data(const thrust::device_vector<T, Alloc>& dev_vec) {
  return thrust::raw_pointer_cast(dev_vec.data());
}
template <typename T, typename Alloc>
T* data(thrust::device_vector<T, Alloc>& dev_vec) {
  return thrust::raw_pointer_cast(dev_vec.data());
}

// this must be instantiated in a .cu file
template <typename T, typename Alloc>
void resize(thrust::device_vector<T, Alloc>& dev_vec, size_t size);
}  // namespace thrust

namespace TiledArray::device {

#ifdef TILEDARRAY_HAS_CUDA
namespace thrust_system = thrust::cuda;
#elif TILEDARRAY_HAS_HIP
namespace thrust_system = thrust::hip;
#endif

}  // namespace TiledArray::device

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_THRUST_H__INCLUDED
