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

#ifndef TILEDARRAY_TENSOR_CUDA_THRUST_H__INCLUDED
#define TILEDARRAY_TENSOR_CUDA_THRUST_H__INCLUDED


#include <TiledArray/config.h>


#ifdef TILEDARRAY_HAS_CUDA


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// thrust::device_vector::data() returns a proxy, provide an overload for std::data() to provide raw ptr
namespace thrust {
template<typename T, typename Alloc>
const T* data (const thrust::device_vector<T, Alloc>& dev_vec) {
  return thrust::raw_pointer_cast(dev_vec.data());
}
template<typename T, typename Alloc>
T* data (thrust::device_vector<T, Alloc>& dev_vec) {
  return thrust::raw_pointer_cast(dev_vec.data());
}

// this must be instantiated in a .cu file
template <typename T, typename Alloc>
void resize(thrust::device_vector<T, Alloc>& dev_vec, size_t size);
}  // namespace thrust


#endif // TILEDARRAY_HAS_CUDA

#endif //TILEDARRAY_TENSOR_CUDA_THRUST_H__INCLUDED
