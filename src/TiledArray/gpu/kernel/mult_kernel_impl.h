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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  Apir 11, 2018
 *
 */

#ifndef TILEDARRAY_CUDA_MULT_KERNEL_IMPL_H__INCLUDED
#define TILEDARRAY_CUDA_MULT_KERNEL_IMPL_H__INCLUDED

#include <TiledArray/external/cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace TiledArray {

/// result[i] = result[i] * arg[i]
template <typename T>
void mult_to_cuda_kernel_impl(T *result, const T *arg, std::size_t n,
                              cudaStream_t stream, int device_id) {
  CudaSafeCall(cudaSetDevice(device_id));

  thrust::multiplies<T> mul_op;
  thrust::transform(
      thrust::cuda::par.on(stream), thrust::device_pointer_cast(arg),
      thrust::device_pointer_cast(arg) + n, thrust::device_pointer_cast(result),
      thrust::device_pointer_cast(result), mul_op);
}

/// result[i] = arg1[i] * arg2[i]
template <typename T>
void mult_cuda_kernel_impl(T *result, const T *arg1, const T *arg2,
                           std::size_t n, cudaStream_t stream, int device_id) {
  CudaSafeCall(cudaSetDevice(device_id));

  thrust::multiplies<T> mul_op;
  thrust::transform(
      thrust::cuda::par.on(stream), thrust::device_pointer_cast(arg1),
      thrust::device_pointer_cast(arg1) + n, thrust::device_pointer_cast(arg2),
      thrust::device_pointer_cast(result), mul_op);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CUDA_MULT_KERNEL_IMPL_H__INCLUDED
