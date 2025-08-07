/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2025  Virginia Tech
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
 *  July 30, 2025
 *
 */

#ifndef TILEDARRAY_DEVICE_UM_TENSOR_H
#define TILEDARRAY_DEVICE_UM_TENSOR_H

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/device/blas.h>
#include <TiledArray/device/device_array_ops.h>
#include <TiledArray/device/kernel/mult_kernel.h>
#include <TiledArray/device/kernel/reduce_kernel.h>
#include <TiledArray/device/um_storage.h>
#include <TiledArray/external/device.h>
#include <TiledArray/external/librett.h>
#include <TiledArray/fwd.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/platform.h>
#include <TiledArray/range.h>


// Forward declare Tensor
namespace TiledArray {
template <typename T, typename Allocator>
class Tensor;
}

#include <concepts>

namespace TiledArray {

/// concept to identify UMTensor types
template <typename T>
concept UMTensorType =
    requires {
      typename T::value_type;
      typename T::allocator_type;
    } && std::same_as<T, TiledArray::Tensor<typename T::value_type,
                                            TiledArray::device_um_allocator<
                                                typename T::value_type>>>;

namespace detail {

template <UMTensorType T>
void to_device(const T &tensor) {
  auto stream = device::stream_for(tensor.range());
  TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
      const_cast<T &>(tensor), stream);
}

/// get device data pointer
template <UMTensorType T>
auto *device_data(const T &tensor) {
  return tensor.data();
}

/// get device data pointer (non-const)
template <UMTensorType T>
auto *device_data(T &tensor) {
  return tensor.data();
}

}  // namespace detail

///
/// gemm
///

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor gemm(const Tensor &left, const Tensor &right, Scalar factor,
            const TiledArray::math::GemmHelper &gemm_helper) {
  // Check that the arguments are not empty and have the correct ranks
  TA_ASSERT(!left.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(!right.empty());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  // TA::Tensor operations currently support only single batch
  TA_ASSERT(left.nbatch() == 1);
  TA_ASSERT(right.nbatch() == 1);

  // result range
  auto result_range = gemm_helper.make_result_range<TiledArray::Range>(
      left.range(), right.range());

  auto &queue = blasqueue_for(result_range);
  const auto stream = device::Stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  Tensor result(result_range);
  TA_ASSERT(result.nbatch() == 1);

  detail::to_device(left);
  detail::to_device(right);
  detail::to_device(result);

  // compute dimensions
  using TiledArray::math::blas::integer;
  integer m = 1, n = 1, k = 1;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  const integer lda = std::max(
      integer{1},
      (gemm_helper.left_op() == TiledArray::math::blas::Op::NoTrans ? k : m));
  const integer ldb = std::max(
      integer{1},
      (gemm_helper.right_op() == TiledArray::math::blas::Op::NoTrans ? n : k));
  const integer ldc = std::max(integer{1}, m);

  using value_type = typename Tensor::value_type;
  value_type factor_t = value_type(factor);
  value_type zero(0);

  blas::gemm(blas::Layout::ColMajor, gemm_helper.right_op(),
             gemm_helper.left_op(), n, m, k, factor_t,
             detail::device_data(right), ldb, detail::device_data(left), lda,
             zero, detail::device_data(result), ldc, queue);

  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
void gemm(Tensor &result, const Tensor &left, const Tensor &right,
          Scalar factor, const TiledArray::math::GemmHelper &gemm_helper) {
  // Check that the result is not empty and has the correct rank
  TA_ASSERT(!result.empty());
  TA_ASSERT(result.range().rank() == gemm_helper.result_rank());

  // Check that the arguments are not empty and have the correct ranks
  TA_ASSERT(!left.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(!right.empty());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  // TA::Tensor operations currently support only single batch
  TA_ASSERT(left.nbatch() == 1);
  TA_ASSERT(right.nbatch() == 1);
  TA_ASSERT(result.nbatch() == 1);

  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(left);
  detail::to_device(right);
  detail::to_device(result);

  // compute dimensions
  using TiledArray::math::blas::integer;
  integer m = 1, n = 1, k = 1;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  const integer lda = std::max(
      integer{1},
      (gemm_helper.left_op() == TiledArray::math::blas::Op::NoTrans ? k : m));
  const integer ldb = std::max(
      integer{1},
      (gemm_helper.right_op() == TiledArray::math::blas::Op::NoTrans ? n : k));
  const integer ldc = std::max(integer{1}, m);

  using value_type = typename Tensor::value_type;
  value_type factor_t = value_type(factor);
  value_type one(1);

  blas::gemm(blas::Layout::ColMajor, gemm_helper.right_op(),
             gemm_helper.left_op(), n, m, k, factor_t,
             detail::device_data(right), ldb, detail::device_data(left), lda,
             one, detail::device_data(result), ldc, queue);

  device::sync_madness_task_with(stream);
}

///
/// clone
///

template <UMTensorType Tensor>
Tensor clone(const Tensor &arg) {
  TA_ASSERT(!arg.empty());

  Tensor result(arg.range());
  auto stream = device::stream_for(result.range());

  detail::to_device(arg);
  detail::to_device(result);

  // copy data
  blas::copy(result.size(), detail::device_data(arg), 1,
             detail::device_data(result), 1, blasqueue_for(result.range()));
  device::sync_madness_task_with(stream);
  return result;
}

///
/// shift
///

template <UMTensorType Tensor, typename Index>
Tensor shift(const Tensor &arg, const Index &bound_shift) {
  TA_ASSERT(!arg.empty());

  // create a shifted range
  TiledArray::Range result_range(arg.range());
  result_range.inplace_shift(bound_shift);

  // get stream using shifted range
  auto &queue = blasqueue_for(result_range);
  const auto stream = device::Stream(queue.device(), queue.stream());

  Tensor result(result_range);

  detail::to_device(arg);
  detail::to_device(result);

  // copy data
  blas::copy(result.size(), detail::device_data(arg), 1,
             detail::device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Index>
Tensor &shift_to(Tensor &arg, const Index &bound_shift) {
  // although shift_to is currently fine on shared objects since ranges are
  // not shared, this will change in the future
#ifdef TA_TENSOR_ASSERT_NO_MUTABLE_OPS_WHILE_SHARED
  TA_ASSERT(data_.use_count() <= 1);
#endif
  const_cast<TiledArray::Range &>(arg.range()).inplace_shift(bound_shift);
  return arg;
}

///
/// permute
///

template <UMTensorType Tensor>
Tensor permute(const Tensor &arg, const TiledArray::Permutation &perm) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(perm.size() == arg.range().rank());

  // compute result range
  auto result_range = perm * arg.range();
  auto stream = device::stream_for(result_range);

  Tensor result(result_range);

  detail::to_device(arg);
  detail::to_device(result);

  // invoke permute function from librett
  using value_type = typename Tensor::value_type;
  librett_permute(const_cast<value_type *>(detail::device_data(arg)),
                  detail::device_data(result), arg.range(), perm, stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
Tensor permute(const Tensor &arg,
               const TiledArray::BipartitePermutation &perm) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(inner_size(perm) == 0);  // this must be a plain permutation
  return permute(arg, outer(perm));
}

///
/// scale
///

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor scale(const Tensor &arg, const Scalar factor) {
  Tensor result(arg.range());

  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg);
  detail::to_device(result);

  // copy and scale
  using value_type = typename Tensor::value_type;
  value_type factor_t = value_type(factor);
  blas::copy(result.size(), detail::device_data(arg), 1,
             detail::device_data(result), 1, queue);
  blas::scal(result.size(), factor_t, detail::device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor &scale_to(Tensor &arg, const Scalar factor) {
  auto &queue = blasqueue_for(arg.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg);

  // in-place scale
  using value_type = typename Tensor::value_type;
  value_type factor_t = value_type(factor);
  blas::scal(arg.size(), factor_t, detail::device_data(arg), 1, queue);
  device::sync_madness_task_with(stream);
  return arg;
}

template <UMTensorType Tensor, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
Tensor scale(const Tensor &arg, const Scalar factor, const Perm &perm) {
  auto result = scale(arg, factor);
  return permute(result, perm);
}

///
/// neg
///

template <UMTensorType Tensor>
Tensor neg(const Tensor &arg) {
  using value_type = typename Tensor::value_type;
  return scale(arg, value_type(-1.0));
}

template <UMTensorType Tensor, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm>
Tensor neg(const Tensor &arg, const Perm &perm) {
  auto result = neg(arg);
  return permute(result, perm);
}

template <UMTensorType Tensor>
Tensor &neg_to(Tensor &arg) {
  using value_type = typename Tensor::value_type;
  return scale_to(arg, value_type(-1.0));
}

///
/// add
///

template <UMTensorType Tensor>
Tensor add(const Tensor &arg1, const Tensor &arg2) {
  Tensor result(arg1.range());

  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg1);
  detail::to_device(arg2);
  detail::to_device(result);

  // result = arg1 + arg2
  using value_type = typename Tensor::value_type;
  blas::copy(result.size(), detail::device_data(arg1), 1,
             detail::device_data(result), 1, queue);
  blas::axpy(result.size(), value_type(1), detail::device_data(arg2), 1,
             detail::device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor add(const Tensor &arg1, const Tensor &arg2, const Scalar factor) {
  auto result = add(arg1, arg2);
  return scale_to(result, factor);
}

template <UMTensorType Tensor, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm>
Tensor add(const Tensor &arg1, const Tensor &arg2, const Perm &perm) {
  auto result = add(arg1, arg2);
  return permute(result, perm);
}

template <UMTensorType Tensor, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
Tensor add(const Tensor &arg1, const Tensor &arg2, const Scalar factor,
           const Perm &perm) {
  auto result = add(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// add_to
///

template <UMTensorType Tensor>
Tensor &add_to(Tensor &result, const Tensor &arg) {
  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(result);
  detail::to_device(arg);

  // result += arg
  using value_type = typename Tensor::value_type;
  blas::axpy(result.size(), value_type(1), detail::device_data(arg), 1,
             detail::device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor &add_to(Tensor &result, const Tensor &arg, const Scalar factor) {
  add_to(result, arg);
  return scale_to(result, factor);
}

///
/// subt
///

template <UMTensorType Tensor>
Tensor subt(const Tensor &arg1, const Tensor &arg2) {
  Tensor result(arg1.range());

  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg1);
  detail::to_device(arg2);
  detail::to_device(result);

  // result = arg1 - arg2
  using value_type = typename Tensor::value_type;
  blas::copy(result.size(), detail::device_data(arg1), 1,
             detail::device_data(result), 1, queue);
  blas::axpy(result.size(), value_type(-1), detail::device_data(arg2), 1,
             detail::device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor subt(const Tensor &arg1, const Tensor &arg2, const Scalar factor) {
  auto result = subt(arg1, arg2);
  return scale_to(result, factor);
}

template <UMTensorType Tensor, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm>
Tensor subt(const Tensor &arg1, const Tensor &arg2, const Perm &perm) {
  auto result = subt(arg1, arg2);
  return permute(result, perm);
}

template <UMTensorType Tensor, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
Tensor subt(const Tensor &arg1, const Tensor &arg2, const Scalar factor,
            const Perm &perm) {
  auto result = subt(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// subt_to
///

template <UMTensorType Tensor>
Tensor &subt_to(Tensor &result, const Tensor &arg) {
  auto &queue = blasqueue_for(result.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(result);
  detail::to_device(arg);

  // result -= arg
  using value_type = typename Tensor::value_type;
  blas::axpy(result.size(), value_type(-1), detail::device_data(arg), 1,
             detail::device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor &subt_to(Tensor &result, const Tensor &arg, const Scalar factor) {
  subt_to(result, arg);
  return scale_to(result, factor);
}

///
/// mult
///

template <UMTensorType Tensor>
Tensor mult(const Tensor &arg1, const Tensor &arg2) {
  std::size_t n = arg1.size();
  TA_ASSERT(arg2.size() == n);

  auto stream = device::stream_for(arg1.range());

  using value_type = typename Tensor::value_type;
  Tensor result(arg1.range());

  detail::to_device(arg1);
  detail::to_device(arg2);
  detail::to_device(result);

  // element-wise multiplication
  device::mult_kernel(detail::device_data(result), detail::device_data(arg1),
                      detail::device_data(arg2), n, stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor mult(const Tensor &arg1, const Tensor &arg2, const Scalar factor) {
  auto result = mult(arg1, arg2);
  return scale_to(result, factor);
}

template <UMTensorType Tensor, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm>
Tensor mult(const Tensor &arg1, const Tensor &arg2, const Perm &perm) {
  auto result = mult(arg1, arg2);
  return permute(result, perm);
}

template <UMTensorType Tensor, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
Tensor mult(const Tensor &arg1, const Tensor &arg2, const Scalar factor,
            const Perm &perm) {
  auto result = mult(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// mult_to
///

template <UMTensorType Tensor>
Tensor &mult_to(Tensor &result, const Tensor &arg) {
  auto stream = device::stream_for(result.range());

  std::size_t n = result.size();
  TA_ASSERT(n == arg.size());

  detail::to_device(result);
  detail::to_device(arg);

  // in-place element-wise multiplication
  device::mult_to_kernel(detail::device_data(result), detail::device_data(arg),
                         n, stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
Tensor &mult_to(Tensor &result, const Tensor &arg, const Scalar factor) {
  mult_to(result, arg);
  return scale_to(result, factor);
}

///
/// dot
///

template <UMTensorType Tensor>
typename Tensor::value_type dot(const Tensor &arg1, const Tensor &arg2) {
  auto &queue = blasqueue_for(arg1.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg1);
  detail::to_device(arg2);

  // compute dot product using device BLAS
  using value_type = typename Tensor::value_type;
  value_type result = value_type(0);
  blas::dot(arg1.size(), detail::device_data(arg1), 1,
            detail::device_data(arg2), 1, &result, queue);
  device::sync_madness_task_with(stream);
  return result;
}

///
/// Reduction
///

template <UMTensorType Tensor>
typename Tensor::value_type squared_norm(const Tensor &arg) {
  auto &queue = blasqueue_for(arg.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg);

  // compute squared norm using dot
  using value_type = typename Tensor::value_type;
  value_type result = value_type(0);
  blas::dot(arg.size(), detail::device_data(arg), 1, detail::device_data(arg),
            1, &result, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
typename Tensor::value_type norm(const Tensor &arg) {
  return std::sqrt(squared_norm(arg));
}

template <UMTensorType Tensor>
typename Tensor::value_type sum(const Tensor &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result =
      device::sum_kernel(detail::device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
typename Tensor::value_type product(const Tensor &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result =
      device::product_kernel(detail::device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
typename Tensor::value_type max(const Tensor &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result =
      device::max_kernel(detail::device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
typename Tensor::value_type min(const Tensor &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result =
      device::min_kernel(detail::device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
typename Tensor::value_type abs_max(const Tensor &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result =
      device::absmax_kernel(detail::device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <UMTensorType Tensor>
typename Tensor::value_type abs_min(const Tensor &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result =
      device::absmin_kernel(detail::device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

}  // namespace TiledArray

/// Serialization support
namespace madness {
namespace archive {

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void load(const Archive &ar, TiledArray::UMTensor<T> &t) {
    TiledArray::Range range{};
    ar & range;

    if (range.volume() > 0) {
      t = TiledArray::UMTensor<T>(std::move(range));
      ar &madness::archive::wrap(t.data(), t.size());
    } else {
      t = TiledArray::UMTensor<T>{};
    }
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void store(const Archive &ar,
                           const TiledArray::UMTensor<T> &t) {
    ar & t.range();
    if (t.range().volume() > 0) {
      ar &madness::archive::wrap(t.data(), t.size());
    }
  }
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_UM_TENSOR_H
