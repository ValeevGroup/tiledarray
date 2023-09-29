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

#ifndef TILEDARRAY_DEVICE_BTAS_H__INCLUDED
#define TILEDARRAY_DEVICE_BTAS_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/math/blas.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/device/blas.h>

#include <TiledArray/external/device.h>
#include <btas/tensor.h>

#include <TiledArray/device/kernel/mult_kernel.h>
#include <TiledArray/device/kernel/reduce_kernel.h>
#include <TiledArray/device/platform.h>
#include <TiledArray/device/um_storage.h>
#include <TiledArray/math/gemm_helper.h>

namespace TiledArray {

namespace device {

namespace btas {

template <typename T, typename Scalar, typename Range, typename Storage,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
::btas::Tensor<T, Range, Storage> gemm(
    const ::btas::Tensor<T, Range, Storage> &left,
    const ::btas::Tensor<T, Range, Storage> &right, Scalar factor,
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

  //  typedef typename Tensor::storage_type storage_type;
  auto result_range =
      gemm_helper.make_result_range<Range>(left.range(), right.range());

  auto &queue = blasqueue_for(result_range);
  const auto device = queue.device();
  const auto str = queue.stream();
  const device::Stream stream(device, str);
  DeviceSafeCall(device::setDevice(device));

  // the result Tensor type
  typedef ::btas::Tensor<T, Range, Storage> Tensor;
  Tensor result;

  if (true) {
    Storage result_storage;
    make_device_storage(result_storage, result_range.area(), stream);
    result = Tensor(std::move(result_range), std::move(result_storage));

    // prefetch data
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
        left.storage(), stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
        right.storage(), stream);

    static_assert(::btas::boxrange_iteration_order<Range>::value ==
                  ::btas::boxrange_iteration_order<Range>::row_major);
    blas::gemm(blas::Layout::ColMajor, gemm_helper.right_op(),
               gemm_helper.left_op(), n, m, k, factor_t,
               device_data(right.storage()), ldb, device_data(left.storage()),
               lda, zero, device_data(result.storage()), n, queue);

    device::sync_madness_task_with(stream);
  }

  return result;
}

template <typename T, typename Scalar, typename Range, typename Storage,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void gemm(::btas::Tensor<T, Range, Storage> &result,
          const ::btas::Tensor<T, Range, Storage> &left,
          const ::btas::Tensor<T, Range, Storage> &right, Scalar factor,
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

  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  T factor_t = T(factor);
  T one(1);
  if (true) {
    // prefetch all data
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
        left.storage(), stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
        right.storage(), stream);
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
        result.storage(), stream);

    static_assert(::btas::boxrange_iteration_order<Range>::value ==
                  ::btas::boxrange_iteration_order<Range>::row_major);
    blas::gemm(blas::Layout::ColMajor, gemm_helper.right_op(),
               gemm_helper.left_op(), n, m, k, factor_t,
               device_data(right.storage()), ldb, device_data(left.storage()),
               lda, one, device_data(result.storage()), n, queue);
    device::sync_madness_task_with(stream);
  }
}

/// result[i] = arg[i]
template <typename T, typename Range, typename Storage>
::btas::Tensor<T, Range, Storage> clone(
    const ::btas::Tensor<T, Range, Storage> &arg) {
  Storage result_storage;
  auto result_range = arg.range();
  auto &queue = blasqueue_for(result_range);
  const auto stream = Stream{queue.device(), queue.stream()};

  make_device_storage(result_storage, arg.size(), stream);
  ::btas::Tensor<T, Range, Storage> result(std::move(result_range),
                                           std::move(result_storage));

  blas::copy(result.size(), device_data(arg.storage()), 1,
             device_data(result.storage()), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] = a * arg[i]
template <typename T, typename Range, typename Storage, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
::btas::Tensor<T, Range, Storage> scale(
    const ::btas::Tensor<T, Range, Storage> &arg, const Scalar a) {
  auto &queue = blasqueue_for(arg.range());
  const device::Stream stream(queue.device(), queue.stream());

  auto result = clone(arg);

  if constexpr (TiledArray::detail::is_blas_numeric_v<Scalar> ||
                std::is_arithmetic_v<Scalar>) {
    blas::scal(result.size(), a, device_data(result.storage()), 1, queue);
  } else {
    if constexpr (TiledArray::detail::is_complex_v<T>) {
      abort();  // fused conjugation requires custom kernels, not yet supported
    } else {
      if constexpr (std::is_same_v<
                        Scalar, TiledArray::detail::ComplexConjugate<void>>) {
      } else if constexpr (std::is_same_v<
                               Scalar,
                               TiledArray::detail::ComplexConjugate<
                                   TiledArray::detail::ComplexNegTag>>) {
        blas::scal(result.size(), static_cast<T>(-1),
                   device_data(result.storage()), 1, queue);
      }
    }
  }

  device::sync_madness_task_with(stream);

  return result;
}

/// result[i] *= a
template <typename T, typename Range, typename Storage, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void scale_to(::btas::Tensor<T, Range, Storage> &result, const Scalar a) {
  auto &queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());

  if constexpr (TiledArray::detail::is_blas_numeric_v<Scalar> ||
                std::is_arithmetic_v<Scalar>) {
    blas::scal(result.size(), a, device_data(result.storage()), 1, queue);
  } else {
    if constexpr (TiledArray::detail::is_complex_v<T>) {
      abort();  // fused conjugation requires custom kernels, not yet supported
    } else {
      if constexpr (std::is_same_v<
                        Scalar, TiledArray::detail::ComplexConjugate<void>>) {
      } else if constexpr (std::is_same_v<
                               Scalar,
                               TiledArray::detail::ComplexConjugate<
                                   TiledArray::detail::ComplexNegTag>>) {
        blas::scal(result.size(), static_cast<T>(-1),
                   device_data(result.storage()), 1, queue);
      }
    }
  }

  device::sync_madness_task_with(stream);
}

/// result[i] = arg1[i] - a * arg2[i]
template <typename T, typename Scalar, typename Range, typename Storage,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
::btas::Tensor<T, Range, Storage> subt(
    const ::btas::Tensor<T, Range, Storage> &arg1,
    const ::btas::Tensor<T, Range, Storage> &arg2, const Scalar a) {
  auto result = clone(arg1);

  // revert the sign of a
  auto b = -a;

  auto &queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());

  if (in_memory_space<MemorySpace::Device>(result.storage())) {
    blas::axpy(result.size(), b, device_data(arg2.storage()), 1,
               device_data(result.storage()), 1, queue);
  } else {
    TA_ASSERT(false);
  }

  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] -= a * arg1[i]
template <typename T, typename Scalar, typename Range, typename Storage,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void subt_to(::btas::Tensor<T, Range, Storage> &result,
             const ::btas::Tensor<T, Range, Storage> &arg1, const Scalar a) {
  auto &queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());

  // revert the sign of a
  auto b = -a;

  blas::axpy(result.size(), b, device_data(arg1.storage()), 1,
             device_data(result.storage()), 1, queue);
  device::sync_madness_task_with(stream);
}

/// result[i] = arg1[i] + a * arg2[i]
template <typename T, typename Scalar, typename Range, typename Storage,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
::btas::Tensor<T, Range, Storage> add(
    const ::btas::Tensor<T, Range, Storage> &arg1,
    const ::btas::Tensor<T, Range, Storage> &arg2, const Scalar a) {
  auto result = clone(arg1);

  auto &queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());

  blas::axpy(result.size(), a, device_data(arg2.storage()), 1,
             device_data(result.storage()), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] += a * arg[i]
template <typename T, typename Scalar, typename Range, typename Storage,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void add_to(::btas::Tensor<T, Range, Storage> &result,
            const ::btas::Tensor<T, Range, Storage> &arg, const Scalar a) {
  auto &queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());

  //   TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(result.storage(),stream);
  //   TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(arg.storage(),stream);

  blas::axpy(result.size(), a, device_data(arg.storage()), 1,
             device_data(result.storage()), 1, queue);

  device::sync_madness_task_with(stream);
}

/// result[i] = result[i] * arg[i]
template <typename T, typename Range, typename Storage>
void mult_to(::btas::Tensor<T, Range, Storage> &result,
             const ::btas::Tensor<T, Range, Storage> &arg) {
  auto &queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());

  std::size_t n = result.size();

  TA_ASSERT(n == arg.size());

  device::mult_to_kernel(result.data(), arg.data(), n, stream);
  device::sync_madness_task_with(stream);
}

/// result[i] = arg1[i] * arg2[i]
template <typename T, typename Range, typename Storage>
::btas::Tensor<T, Range, Storage> mult(
    const ::btas::Tensor<T, Range, Storage> &arg1,
    const ::btas::Tensor<T, Range, Storage> &arg2) {
  std::size_t n = arg1.size();

  TA_ASSERT(arg2.size() == n);

  auto stream = stream_for(arg1.range());

  Storage result_storage;
  make_device_storage(result_storage, n, stream);
  ::btas::Tensor<T, Range, Storage> result(arg1.range(),
                                           std::move(result_storage));

  device::mult_kernel(result.data(), arg1.data(), arg2.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

// foreach(i) result += arg[i] * arg[i]
template <typename T, typename Range, typename Storage>
typename ::btas::Tensor<T, Range, Storage>::value_type squared_norm(
    const ::btas::Tensor<T, Range, Storage> &arg) {
  auto &queue = blasqueue_for(arg.range());
  const device::Stream stream(queue.device(), queue.stream());

  auto &storage = arg.storage();
  using TiledArray::math::blas::integer;
  integer size = storage.size();
  T result = 0;
  if (in_memory_space<MemorySpace::Device>(storage)) {
    blas::dot(size, device_data(storage), 1, device_data(storage), 1, &result,
              queue);
  } else {
    TA_ASSERT(false);
    //    result = TiledArray::math::dot(size, storage.data(), storage.data());
  }
  device::sync_madness_task_with(stream);
  return result;
}

// foreach(i) result += arg1[i] * arg2[i]
template <typename T, typename Range, typename Storage>
typename ::btas::Tensor<T, Range, Storage>::value_type dot(
    const ::btas::Tensor<T, Range, Storage> &arg1,
    const ::btas::Tensor<T, Range, Storage> &arg2) {
  auto &queue = blasqueue_for(arg1.range());
  const device::Stream stream(queue.device(), queue.stream());

  using TiledArray::math::blas::integer;
  integer size = arg1.storage().size();

  TA_ASSERT(size == arg2.storage().size());

  T result = 0;
  if (in_memory_space<MemorySpace::Device>(arg1.storage()) &&
      in_memory_space<MemorySpace::Device>(arg2.storage())) {
    blas::dot(size, device_data(arg1.storage()), 1, device_data(arg2.storage()),
              1, &result, queue);
  } else {
    TA_ASSERT(false);
    //    result = TiledArray::math::dot(size, storage.data(), storage.data());
  }
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T sum(const ::btas::Tensor<T, Range, Storage> &arg) {
  auto stream = device::stream_for(arg.range());

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = device::sum_kernel(arg.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T product(const ::btas::Tensor<T, Range, Storage> &arg) {
  auto stream = device::stream_for(arg.range());

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = device::product_kernel(arg.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T min(const ::btas::Tensor<T, Range, Storage> &arg) {
  auto stream = device::stream_for(arg.range());

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = device::min_kernel(arg.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T max(const ::btas::Tensor<T, Range, Storage> &arg) {
  auto stream = device::stream_for(arg.range());

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = device::max_kernel(arg.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T absmin(const ::btas::Tensor<T, Range, Storage> &arg) {
  auto stream = device::stream_for(arg.range());

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = device::absmin_kernel(arg.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Range, typename Storage>
T absmax(const ::btas::Tensor<T, Range, Storage> &arg) {
  auto stream = device::stream_for(arg.range());

  auto &storage = arg.storage();
  auto n = storage.size();

  auto result = device::absmax_kernel(arg.data(), n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

}  // namespace btas

}  // namespace device

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_BTAS_H__INCLUDED
