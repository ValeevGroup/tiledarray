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
 *  July 24, 2018
 *
 */

#ifndef TILEDARRAY_BTAS_CUDA_CUBLAS_H__INCLUDED
#define TILEDARRAY_BTAS_CUDA_CUBLAS_H__INCLUDED

#include <TiledArray/cuda/cublas.h>
#include <TiledArray/math/blas.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/external/cuda.h>
#include <btas/tensor.h>

#include <TiledArray/cuda/kernel/mult_kernel.h>
#include <TiledArray/cuda/kernel/reduce_kernel.h>
#include <TiledArray/cuda/platform.h>
#include <TiledArray/cuda/um_storage.h>
#include <TiledArray/math/gemm_helper.h>

namespace TiledArray {

template <typename T, typename Scalar, typename Range, typename Storage,
    typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btas::Tensor<T, Range, Storage> btas_tensor_gemm_cuda_impl(
    const btas::Tensor<T, Range, Storage> &left,
    const btas::Tensor<T, Range, Storage> &right, Scalar factor,
    const TiledArray::math::GemmHelper &gemm_helper) {
  // Check that the arguments are not empty and have the correct ranks
  TA_ASSERT(!left.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(!right.empty());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  // Check that the inner dimensions of left and right match
  TA_ASSERT(
      ignore_tile_position() ||
      gemm_helper.left_right_congruent(std::cbegin(left.range().lobound()),
                                       std::cbegin(right.range().lobound())));
  TA_ASSERT(
      ignore_tile_position() ||
      gemm_helper.left_right_congruent(std::cbegin(left.range().upbound()),
                                       std::cbegin(right.range().upbound())));
  TA_ASSERT(gemm_helper.left_right_congruent(
      std::cbegin(left.range().extent()), std::cbegin(right.range().extent())));

  // Compute gemm dimensions
  using TiledArray::math::blas::integer;
  integer m = 1, n = 1, k = 1;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  // Get the leading dimension for left and right matrices.
  const integer lda =
      (gemm_helper.left_op() == TiledArray::math::blas::Op::NoTrans ? k : m);
  const integer ldb =
      (gemm_helper.right_op() == TiledArray::math::blas::Op::NoTrans ? n : k);

  T factor_t = T(factor);
  T zero(0);

  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

  //  typedef typename Tensor::storage_type storage_type;
  auto result_range =
      gemm_helper.make_result_range<Range>(left.range(), right.range());

  auto &cuda_stream = detail::get_stream_based_on_range(result_range);

  // the result Tensor type
  typedef btas::Tensor<T, Range, Storage> Tensor;
  Tensor result;

  // check if stream is busy
  //  auto stream_status = cudaStreamQuery(cuda_stream);

  // if stream is completed, use GPU
  //  if (stream_status == cudaSuccess) {
  if (true) {
    Storage result_storage;
    make_device_storage(result_storage, result_range.area(), cuda_stream);
    result = Tensor(std::move(result_range), std::move(result_storage));

    // left and right are readonly!!
    //    cudaMemAdvise(device_data(left), left.size() * sizeof(T),
    //                  cudaMemAdviseSetReadMostly,
    //                  cudaEnv::instance()->current_cuda_device_id());
    //    cudaMemAdvise(device_data(right), right.size() * sizeof(T),
    //                  cudaMemAdviseSetReadMostly,
    //                  cudaEnv::instance()->current_cuda_device_id());

    // prefetch data
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        left.storage(), cuda_stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        right.storage(), cuda_stream);

    const auto &handle = cuBLASHandlePool::handle();
    CublasSafeCall(cublasSetStream(handle, cuda_stream));

    CublasSafeCall(cublasGemm(handle, to_cublas_op(gemm_helper.right_op()),
                              to_cublas_op(gemm_helper.left_op()), n, m, k,
                              &factor_t, device_data(right.storage()), ldb,
                              device_data(left.storage()), lda, &zero,
                              device_data(result.storage()), n));

    // wait for cuda calls to finish
    //    detail::thread_wait_cuda_stream(cuda_stream);
    synchronize_stream(&cuda_stream);
  }
  // otherwise, use CPU
  else {
    Storage result_storage(result_range.area());
    result = Tensor(std::move(result_range), std::move(result_storage));

    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        result.storage(), cuda_stream);

    // left and right are readonly!!
    cudaMemAdvise(device_data(left), left.size() * sizeof(T),
                  cudaMemAdviseSetReadMostly,
                  cudaEnv::instance()->current_cuda_device_id());
    cudaMemAdvise(device_data(right), right.size() * sizeof(T),
                  cudaMemAdviseSetReadMostly,
                  cudaEnv::instance()->current_cuda_device_id());

    // prefetch data
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        left.storage(), cuda_stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        right.storage(), cuda_stream);

    TiledArray::math::blas::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n,
                           k, factor_t, left.data(), lda, right.data(), ldb,
                           zero, result.data(), n);
  }

  return result;
}

template <typename T, typename Scalar, typename Range, typename Storage, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void btas_tensor_gemm_cuda_impl(
    btas::Tensor<T, Range, Storage> &result,
    const btas::Tensor<T, Range, Storage> &left,
    const btas::Tensor<T, Range, Storage> &right, Scalar factor,
    const TiledArray::math::GemmHelper &gemm_helper) {
  // Check that the result is not empty and has the correct rank
  TA_ASSERT(!result.empty());
  TA_ASSERT(result.range().rank() == gemm_helper.result_rank());

  // Check that the arguments are not empty and have the correct ranks
  TA_ASSERT(!left.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(!right.empty());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  // Check that the outer dimensions of left match the the corresponding
  // dimensions in result
  TA_ASSERT(
      ignore_tile_position() ||
      gemm_helper.left_result_congruent(std::cbegin(left.range().lobound()),
                                        std::cbegin(result.range().lobound())));
  TA_ASSERT(
      ignore_tile_position() ||
      gemm_helper.left_result_congruent(std::cbegin(left.range().upbound()),
                                        std::cbegin(result.range().upbound())));
  TA_ASSERT(
      gemm_helper.left_result_congruent(std::cbegin(left.range().extent()),
                                        std::cbegin(result.range().extent())));

  // Check that the outer dimensions of right match the the corresponding
  // dimensions in result
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.right_result_congruent(
                std::cbegin(right.range().lobound()),
                std::cbegin(result.range().lobound())));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.right_result_congruent(
                std::cbegin(right.range().upbound()),
                std::cbegin(result.range().upbound())));
  TA_ASSERT(
      gemm_helper.right_result_congruent(std::cbegin(right.range().extent()),
                                         std::cbegin(result.range().extent())));

  // Check that the inner dimensions of left and right match
  TA_ASSERT(
      ignore_tile_position() ||
      gemm_helper.left_right_congruent(std::cbegin(left.range().lobound()),
                                       std::cbegin(right.range().lobound())));
  TA_ASSERT(
      ignore_tile_position() ||
      gemm_helper.left_right_congruent(std::cbegin(left.range().upbound()),
                                       std::cbegin(right.range().upbound())));
  TA_ASSERT(gemm_helper.left_right_congruent(
      std::cbegin(left.range().extent()), std::cbegin(right.range().extent())));

  // Compute gemm dimensions
  using TiledArray::math::blas::integer;
  integer m, n, k;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  // Get the leading dimension for left and right matrices.
  const integer lda =
      (gemm_helper.left_op() == TiledArray::math::blas::Op::NoTrans ? k : m);
  const integer ldb =
      (gemm_helper.right_op() == TiledArray::math::blas::Op::NoTrans ? n : k);

  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());

  T factor_t = T(factor);
  T one(1);
  // check if stream is busy
  //  auto stream_status = cudaStreamQuery(cuda_stream);

  // if stream is completed, use GPU
  //  if (stream_status == cudaSuccess) {
  if (true) {
    // left and right are readonly!!
    //    cudaMemAdvise(device_data(left), left.size() * sizeof(T),
    //                  cudaMemAdviseSetReadMostly,
    //                  cudaEnv::instance()->current_cuda_device_id());
    //    cudaMemAdvise(device_data(right), right.size() * sizeof(T),
    //                  cudaMemAdviseSetReadMostly,
    //                  cudaEnv::instance()->current_cuda_device_id());

    // prefetch all data
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        left.storage(), cuda_stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        right.storage(), cuda_stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        result.storage(), cuda_stream);

    const auto &handle = cuBLASHandlePool::handle();
    CublasSafeCall(cublasSetStream(handle, cuda_stream));
    CublasSafeCall(cublasGemm(handle, to_cublas_op(gemm_helper.right_op()),
                              to_cublas_op(gemm_helper.left_op()), n, m, k,
                              &factor_t, device_data(right.storage()), ldb,
                              device_data(left.storage()), lda, &one,
                              device_data(result.storage()), n));
    synchronize_stream(&cuda_stream);

    //    detail::thread_wait_cuda_stream(cuda_stream);

  } else {
    // left and right are readonly!!
    cudaMemAdvise(device_data(left), left.size() * sizeof(T),
                  cudaMemAdviseSetReadMostly,
                  cudaEnv::instance()->current_cuda_device_id());
    cudaMemAdvise(device_data(right), right.size() * sizeof(T),
                  cudaMemAdviseSetReadMostly,
                  cudaEnv::instance()->current_cuda_device_id());

    // prefetch data
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        left.storage(), cuda_stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        right.storage(), cuda_stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        result.storage(), cuda_stream);

    TiledArray::math::blas::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n,
                           k, factor_t, left.data(), lda, right.data(), ldb,
                           one, result.data(), n);
  }
}

/// result[i] = arg[i]
template <typename T, typename Range, typename Storage>
btas::Tensor<T, Range, Storage> btas_tensor_clone_cuda_impl(
    const btas::Tensor<T, Range, Storage> &arg) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

  Storage result_storage;
  auto result_range = arg.range();
  auto &cuda_stream = detail::get_stream_based_on_range(result_range);

  make_device_storage(result_storage, arg.size(), cuda_stream);
  btas::Tensor<T, Range, Storage> result(std::move(result_range),
                                         std::move(result_storage));

  // call cublasCopy
  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));

  CublasSafeCall(cublasCopy(handle, result.size(), device_data(arg.storage()),
                            1, device_data(result.storage()), 1));

  synchronize_stream(&cuda_stream);
  return result;
}

/// result[i] = a * arg[i]
template <typename T, typename Range, typename Storage, typename Scalar,
    typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btas::Tensor<T, Range, Storage> btas_tensor_scale_cuda_impl(
    const btas::Tensor<T, Range, Storage> &arg, const Scalar a) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  //  std::cout << "scale, tile offset: " << arg.range().offset() << " stream: "
  //  << arg.range().offset() % cudaEnv::instance()->num_cuda_streams() << "\n";

  auto result = btas_tensor_clone_cuda_impl(arg);

  // call cublasScale
  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));
  CublasSafeCall(
      cublasScal(handle, result.size(), &a, device_data(result.storage()), 1));

  synchronize_stream(&cuda_stream);

  return result;
}

/// result[i] *= a
template <typename T, typename Range, typename Storage, typename Scalar,
    typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void btas_tensor_scale_to_cuda_impl(btas::Tensor<T, Range, Storage> &result,
                                    const Scalar a) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());
  // call cublasScale
  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));
  CublasSafeCall(
      cublasScal(handle, result.size(), &a, device_data(result.storage()), 1));

  synchronize_stream(&cuda_stream);
}

/// result[i] = arg1[i] - a * arg2[i]
template <typename T, typename Scalar, typename Range, typename Storage,
    typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btas::Tensor<T, Range, Storage> btas_tensor_subt_cuda_impl(
    const btas::Tensor<T, Range, Storage> &arg1,
    const btas::Tensor<T, Range, Storage> &arg2, const Scalar a) {
  auto result = btas_tensor_clone_cuda_impl(arg1);

  // revert the sign of a
  auto b = -a;

  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());

  if (in_memory_space<MemorySpace::CUDA>(result.storage())) {
    const auto &handle = cuBLASHandlePool::handle();
    CublasSafeCall(cublasSetStream(handle, cuda_stream));
    CublasSafeCall(cublasAxpy(handle, result.size(), &b,
                              device_data(arg2.storage()), 1,
                              device_data(result.storage()), 1));
  } else {
    TA_ASSERT(false);
    //    btas::axpy(1.0, arg, result);
  }

  synchronize_stream(&cuda_stream);
  return result;
}

/// result[i] -= a * arg1[i]
template <typename T, typename Scalar, typename Range, typename Storage, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void btas_tensor_subt_to_cuda_impl(btas::Tensor<T, Range, Storage> &result,
                                   const btas::Tensor<T, Range, Storage> &arg1,
                                   const Scalar a) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());

  // revert the sign of a
  auto b = -a;

  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));
  CublasSafeCall(cublasAxpy(handle, result.size(), &b,
                            device_data(arg1.storage()), 1,
                            device_data(result.storage()), 1));
  synchronize_stream(&cuda_stream);
}

/// result[i] = arg1[i] + a * arg2[i]
template <typename T, typename Scalar, typename Range, typename Storage, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btas::Tensor<T, Range, Storage> btas_tensor_add_cuda_impl(
    const btas::Tensor<T, Range, Storage> &arg1,
    const btas::Tensor<T, Range, Storage> &arg2, const Scalar a) {
  auto result = btas_tensor_clone_cuda_impl(arg1);

  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());

  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));
  CublasSafeCall(cublasAxpy(handle, result.size(), &a,
                            device_data(arg2.storage()), 1,
                            device_data(result.storage()), 1));

  synchronize_stream(&cuda_stream);
  return result;
}

/// result[i] += a * arg[i]
template <typename T, typename Scalar, typename Range, typename Storage, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void btas_tensor_add_to_cuda_impl(btas::Tensor<T, Range, Storage> &result,
                                  const btas::Tensor<T, Range, Storage> &arg,
                                  const Scalar a) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());

  //   TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(result.storage(),cuda_stream);
  //   TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(arg.storage(),cuda_stream);

  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));
  CublasSafeCall(cublasAxpy(handle, result.size(), &a,
                            device_data(arg.storage()), 1,
                            device_data(result.storage()), 1));

  synchronize_stream(&cuda_stream);
}

/// result[i] = result[i] * arg[i]
template <typename T, typename Range, typename Storage>
void btas_tensor_mult_to_cuda_impl(btas::Tensor<T, Range, Storage> &result,
                                   const btas::Tensor<T, Range, Storage> &arg) {
  auto device_id = cudaEnv::instance()->current_cuda_device_id();
  auto &cuda_stream = detail::get_stream_based_on_range(result.range());

  std::size_t n = result.size();

  TA_ASSERT(n == arg.size());

  mult_to_cuda_kernel(result.data(), arg.data(), n, cuda_stream, device_id);
  synchronize_stream(&cuda_stream);
}

/// result[i] = arg1[i] * arg2[i]
template <typename T, typename Range, typename Storage>
btas::Tensor<T, Range, Storage> btas_tensor_mult_cuda_impl(
    const btas::Tensor<T, Range, Storage> &arg1,
    const btas::Tensor<T, Range, Storage> &arg2) {
  std::size_t n = arg1.size();

  TA_ASSERT(arg2.size() == n);

  auto device_id = cudaEnv::instance()->current_cuda_device_id();
  CudaSafeCall(cudaSetDevice(device_id));
  auto &cuda_stream = detail::get_stream_based_on_range(arg1.range());

  Storage result_storage;
  make_device_storage(result_storage, n, cuda_stream);
  btas::Tensor<T, Range, Storage> result(arg1.range(),
                                         std::move(result_storage));

  mult_cuda_kernel(result.data(), arg1.data(), arg2.data(), n, cuda_stream,
                   device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

// foreach(i) result += arg[i] * arg[i]
template <typename T, typename Range, typename Storage>
typename btas::Tensor<T, Range, Storage>::value_type
btas_tensor_squared_norm_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());

  auto &storage = arg.storage();
  using TiledArray::math::blas::integer;
  integer size = storage.size();
  T result = 0;
  if (in_memory_space<MemorySpace::CUDA>(storage)) {
    const auto &handle = cuBLASHandlePool::handle();
    CublasSafeCall(cublasSetStream(handle, cuda_stream));
    CublasSafeCall(cublasDot(handle, size, device_data(storage), 1,
                             device_data(storage), 1, &result));
  } else {
    TA_ASSERT(false);
    //    result = TiledArray::math::dot(size, storage.data(), storage.data());
  }
  synchronize_stream(&cuda_stream);
  return result;
}

// foreach(i) result += arg1[i] * arg2[i]
template <typename T, typename Range, typename Storage>
typename btas::Tensor<T, Range, Storage>::value_type btas_tensor_dot_cuda_impl(
    const btas::Tensor<T, Range, Storage> &arg1,
    const btas::Tensor<T, Range, Storage> &arg2) {
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

  auto &cuda_stream = detail::get_stream_based_on_range(arg1.range());

  using TiledArray::math::blas::integer;
  integer size = arg1.storage().size();

  TA_ASSERT(size == arg2.storage().size());

  T result = 0;
  if (in_memory_space<MemorySpace::CUDA>(arg1.storage()) &&
      in_memory_space<MemorySpace::CUDA>(arg2.storage())) {
    const auto &handle = cuBLASHandlePool::handle();
    CublasSafeCall(cublasSetStream(handle, cuda_stream));
    CublasSafeCall(cublasDot(handle, size, device_data(arg1.storage()), 1,
                             device_data(arg2.storage()), 1, &result));
  } else {
    TA_ASSERT(false);
    //    result = TiledArray::math::dot(size, storage.data(), storage.data());
  }
  synchronize_stream(&cuda_stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T btas_tensor_sum_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  auto device_id = cudaEnv::instance()->current_cuda_device_id();

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = sum_cuda_kernel(arg.data(), n, cuda_stream, device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T btas_tensor_product_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  auto device_id = cudaEnv::instance()->current_cuda_device_id();

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = product_cuda_kernel(arg.data(), n, cuda_stream, device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T btas_tensor_min_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  auto device_id = cudaEnv::instance()->current_cuda_device_id();

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = min_cuda_kernel(arg.data(), n, cuda_stream, device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T btas_tensor_max_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  auto device_id = cudaEnv::instance()->current_cuda_device_id();

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = max_cuda_kernel(arg.data(), n, cuda_stream, device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T btas_tensor_absmin_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  auto device_id = cudaEnv::instance()->current_cuda_device_id();

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = absmin_cuda_kernel(arg.data(), n, cuda_stream, device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T btas_tensor_absmax_cuda_impl(const btas::Tensor<T, Range, Storage> &arg) {
  auto &cuda_stream = detail::get_stream_based_on_range(arg.range());
  auto device_id = cudaEnv::instance()->current_cuda_device_id();

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = absmax_cuda_kernel(arg.data(), n, cuda_stream, device_id);

  synchronize_stream(&cuda_stream);
  return result;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_BTAS_CUDA_CUBLAS_H__INCLUDED
