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

#ifndef TILEDARRAY_CUDA_REDUCE_KERNEL_IMPL_H__INCLUDED
#define TILEDARRAY_CUDA_REDUCE_KERNEL_IMPL_H__INCLUDED

#include <limits>

#include <TiledArray/external/cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

namespace TiledArray {

namespace detail {

template <typename T>
struct absolute_value : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(const T &x) const {
    return x < T(0) ? -x : x;
  }
};

}  // namespace detail

/// T = reduce(T* arg)
template <typename T, typename ReduceOp>
T reduce_cuda_kernel_impl(ReduceOp &&op, const T *arg, std::size_t n, T init,
                          cudaStream_t stream, int device_id) {
  CudaSafeCall(cudaSetDevice(device_id));

  auto arg_p = thrust::device_pointer_cast(arg);

  auto result = thrust::reduce(thrust::cuda::par.on(stream), arg_p, arg_p + n,
                               init, std::forward<ReduceOp>(op));

  return result;
}

template <typename T>
T product_reduce_cuda_kernel_impl(const T *arg, std::size_t n,
                                  cudaStream_t stream, int device_id) {
  T init(1);
  thrust::multiplies<T> mul_op;
  return reduce_cuda_kernel_impl(mul_op, arg, n, init, stream, device_id);
}

template <typename T>
T sum_reduce_cuda_kernel_impl(const T *arg, std::size_t n, cudaStream_t stream,
                              int device_id) {
  T init(0);
  thrust::plus<T> plus_op;
  return reduce_cuda_kernel_impl(plus_op, arg, n, init, stream, device_id);
}

template <typename T>
T max_reduce_cuda_kernel_impl(const T *arg, std::size_t n, cudaStream_t stream,
                              int device_id) {
  T init = std::numeric_limits<T>::lowest();
  thrust::maximum<T> max_op;
  return reduce_cuda_kernel_impl(max_op, arg, n, init, stream, device_id);
}

template <typename T>
T min_reduce_cuda_kernel_impl(const T *arg, std::size_t n, cudaStream_t stream,
                              int device_id) {
  T init = std::numeric_limits<T>::max();
  thrust::minimum<T> min_op;
  return reduce_cuda_kernel_impl(min_op, arg, n, init, stream, device_id);
}

template <typename T>
T absmax_reduce_cuda_kernel_impl(const T *arg, std::size_t n,
                                 cudaStream_t stream, int device_id) {
  T init(0);
  thrust::maximum<T> max_op;
  detail::absolute_value<T> abs_op;

  CudaSafeCall(cudaSetDevice(device_id));

  auto arg_p = thrust::device_pointer_cast(arg);

  auto result = thrust::transform_reduce(thrust::cuda::par.on(stream), arg_p,
                                         arg_p + n, abs_op, init, max_op);

  return result;
}

template <typename T>
T absmin_reduce_cuda_kernel_impl(const T *arg, std::size_t n,
                                 cudaStream_t stream, int device_id) {
  T init(0);
  thrust::minimum<T> min_op;
  detail::absolute_value<T> abs_op;

  CudaSafeCall(cudaSetDevice(device_id));

  auto arg_p = thrust::device_pointer_cast(arg);

  auto result = thrust::transform_reduce(thrust::cuda::par.on(stream), arg_p,
                                         arg_p + n, abs_op, init, min_op);
  return result;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CUDA_REDUCE_KERNEL_IMPL_H__INCLUDED
