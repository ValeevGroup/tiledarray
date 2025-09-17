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

#include <TiledArray/fwd.h>

#include <TiledArray/conversions/to_new_tile_type.h>
#include <TiledArray/device/blas.h>
#include <TiledArray/device/device_array.h>
#include <TiledArray/device/kernel/mult_kernel.h>
#include <TiledArray/device/kernel/reduce_kernel.h>
#include <TiledArray/device/um_storage.h>
#include <TiledArray/external/device.h>
#include <TiledArray/external/librett.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/platform.h>
#include <TiledArray/range.h>

#include <concepts>

namespace TiledArray {
namespace detail {

/// is_device_tile specialization for UMTensor
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
struct is_device_tile<
    ::TiledArray::Tensor<T, TiledArray::device_um_allocator<T>>>
    : public std::true_type {};

/// pre-fetch to device
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
void to_device(const UMTensor<T> &tensor) {
  auto stream = device::stream_for(tensor.range());
  TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(tensor,
                                                                     stream);
}

/// pre-fetch to host
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
void to_host(const UMTensor<T> &tensor) {
  auto stream = device::stream_for(tensor.range());
  TiledArray::to_execution_space<TiledArray::ExecutionSpace::Host>(tensor,
                                                                   stream);
}

/// handle ComplexConjugate handling for scaling functions
/// follows the logic in device/btas.h
template <typename T, typename Scalar, typename Queue>
  requires TiledArray::detail::is_numeric_v<T>
void apply_scale_factor(T *data, std::size_t size, const Scalar &factor,
                        Queue &queue) {
  if constexpr (TiledArray::detail::is_blas_numeric_v<Scalar> ||
                std::is_arithmetic_v<Scalar>) {
    blas::scal(size, factor, data, 1, queue);
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
        blas::scal(size, T(-1), data, 1, queue);
      }
    }
  }
}

}  // namespace detail

/// Contains internal implementations of functions which take blas::Queue as an
/// argument to be used in composite functions to avoid race conditions
namespace device::impl {

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> clone(const UMTensor<T> &arg, blas::Queue &queue) {
  TA_ASSERT(!arg.empty());

  UMTensor<T> result(arg.range());
  auto stream = device::Stream(queue.device(), queue.stream());
  TiledArray::detail::to_device(arg);
  TiledArray::detail::to_device(result);

  // copy data
  blas::copy(result.size(), device_data(arg), 1, device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

/*/// make sure you pass the correct queue object to this function. ie, the
queue
/// generated from the permuted range
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> permute(const UMTensor<T> &arg, const TiledArray::Permutation &perm,
                    blas::Queue &queue) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(perm.size() == arg.range().rank());

  // computed result range
  auto result_range = perm * arg.range();
  auto stream = device::Stream(queue.device(), queue.stream());

  UMTensor<T> result(result_range);
  TiledArray::detail::to_device(arg);
  TiledArray::detail::to_device(result);

  // invoke permute from librett
  librett_permute(const_cast<T *>(device_data(arg)), device_data(result),
                  arg.range(), perm, stream);
  device::sync_madness_task_with(stream);
  return result;
}*/

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> scale(const UMTensor<T> &arg, const Scalar factor,
                  blas::Queue &queue) {
  const auto stream = device::Stream(queue.device(), queue.stream());
  auto result = device::impl::clone(arg, queue);
  TiledArray::detail::apply_scale_factor(device_data(result), result.size(),
                                         factor, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> &scale_to(UMTensor<T> &arg, const Scalar factor,
                      blas::Queue &queue) {
  const auto stream = device::Stream(queue.device(), queue.stream());

  TiledArray::detail::to_device(arg);
  // in-place scale
  TiledArray::detail::apply_scale_factor(device_data(arg), arg.size(), factor,
                                         queue);
  device::sync_madness_task_with(stream);
  return arg;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> add(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                blas::Queue &queue) {
  const auto stream = device::Stream(queue.device(), queue.stream());

  UMTensor<T> result(arg1.range());

  TiledArray::detail::to_device(arg1);
  TiledArray::detail::to_device(arg2);
  TiledArray::detail::to_device(result);

  // result = arg1 + arg2
  blas::copy(result.size(), device_data(arg1), 1, device_data(result), 1,
             queue);
  blas::axpy(result.size(), 1, device_data(arg2), 1, device_data(result), 1,
             queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &add_to(UMTensor<T> &result, const UMTensor<T> &arg,
                    blas::Queue &queue) {
  const auto stream = device::Stream(queue.device(), queue.stream());

  TiledArray::detail::to_device(result);
  TiledArray::detail::to_device(arg);

  // result += arg
  blas::axpy(result.size(), 1, device_data(arg), 1, device_data(result), 1,
             queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> subt(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 blas::Queue &queue) {
  const auto stream = device::Stream(queue.device(), queue.stream());

  UMTensor<T> result(arg1.range());

  TiledArray::detail::to_device(arg1);
  TiledArray::detail::to_device(arg2);
  TiledArray::detail::to_device(result);

  // result = arg1 - arg2
  blas::copy(result.size(), device_data(arg1), 1, device_data(result), 1,
             queue);
  blas::axpy(result.size(), T(-1), device_data(arg2), 1, device_data(result), 1,
             queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &subt_to(UMTensor<T> &result, const UMTensor<T> &arg,
                     blas::Queue &queue) {
  const auto stream = device::Stream(queue.device(), queue.stream());

  TiledArray::detail::to_device(result);
  TiledArray::detail::to_device(arg);

  // result -= arg
  blas::axpy(result.size(), T(-1), device_data(arg), 1, device_data(result), 1,
             queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> mult(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 blas::Queue &queue) {
  TA_ASSERT(arg1.size() == arg2.size());

  const auto stream = device::Stream(queue.device(), queue.stream());

  UMTensor<T> result(arg1.range());

  TiledArray::detail::to_device(arg1);
  TiledArray::detail::to_device(arg2);
  TiledArray::detail::to_device(result);

  // element-wise multiplication
  device::mult_kernel(device_data(result), device_data(arg1), device_data(arg2),
                      arg1.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &mult_to(UMTensor<T> &result, const UMTensor<T> &arg,
                     blas::Queue &queue) {
  TA_ASSERT(result.size() == arg.size());

  const auto stream = device::Stream(queue.device(), queue.stream());

  TiledArray::detail::to_device(result);
  TiledArray::detail::to_device(arg);

  // in-place element-wise multiplication
  device::mult_to_kernel(device_data(result), device_data(arg), result.size(),
                         stream);

  device::sync_madness_task_with(stream);
  return result;
}

}  // namespace device::impl

///
/// gemm
///

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> gemm(const UMTensor<T> &left, const UMTensor<T> &right,
                 Scalar factor,
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

  UMTensor<T> result(result_range);
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
  const integer ldc = std::max(integer{1}, n);

  auto factor_t = T(factor);
  T zero(0);

  blas::gemm(blas::Layout::ColMajor, gemm_helper.right_op(),
             gemm_helper.left_op(), n, m, k, factor_t, device_data(right), ldb,
             device_data(left), lda, zero, device_data(result), ldc, queue);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
void gemm(UMTensor<T> &result, const UMTensor<T> &left,
          const UMTensor<T> &right, Scalar factor,
          const TiledArray::math::GemmHelper &gemm_helper) {
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

  // Check dimension congruence
  TA_ASSERT(gemm_helper.left_result_congruent(left.range().extent_data(),
                                              result.range().extent_data()));
  TA_ASSERT(gemm_helper.right_result_congruent(right.range().extent_data(),
                                               result.range().extent_data()));
  TA_ASSERT(gemm_helper.left_right_congruent(left.range().extent_data(),
                                             right.range().extent_data()));

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
  const integer ldc = std::max(integer{1}, n);

  auto factor_t = T(factor);
  T one(1);

  blas::gemm(blas::Layout::ColMajor, gemm_helper.right_op(),
             gemm_helper.left_op(), n, m, k, factor_t, device_data(right), ldb,
             device_data(left), lda, one, device_data(result), ldc, queue);

  device::sync_madness_task_with(stream);
}

///
/// clone
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> clone(const UMTensor<T> &arg) {
  TA_ASSERT(!arg.empty());

  auto &queue = blasqueue_for(arg.range());
  return device::impl::clone(arg, queue);
}

///
/// shift
///

template <typename T, typename Index>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> shift(const UMTensor<T> &arg, const Index &bound_shift) {
  TA_ASSERT(!arg.empty());

  // create a shifted range
  TiledArray::Range result_range(arg.range());
  result_range.inplace_shift(bound_shift);

  // get stream using shifted range
  auto &queue = blasqueue_for(result_range);
  const auto stream = device::Stream(queue.device(), queue.stream());

  UMTensor<T> result(result_range);

  detail::to_device(arg);
  detail::to_device(result);

  // copy data
  blas::copy(result.size(), device_data(arg), 1, device_data(result), 1, queue);
  device::sync_madness_task_with(stream);
  return result;
}

/// this is probably not needed, range changes, but no actual data of the tensor
/// changes
template <typename T, typename Index>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &shift_to(UMTensor<T> &arg, const Index &bound_shift) {
  const_cast<TiledArray::Range &>(arg.range()).inplace_shift(bound_shift);
  return arg;
}

///
/// permute
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> permute(const UMTensor<T> &arg,
                    const TiledArray::Permutation &perm) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(perm.size() == arg.range().rank());

  // compute result range
  auto result_range = perm * arg.range();
  auto stream = device::stream_for(result_range);

  UMTensor<T> result(result_range);

  detail::to_device(arg);
  detail::to_device(result);

  // invoke permute function from librett
  librett_permute(const_cast<T *>(device_data(arg)), device_data(result),
                  arg.range(), perm, stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> permute(const UMTensor<T> &arg,
                    const TiledArray::BipartitePermutation &perm) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(inner_size(perm) == 0);  // this must be a plain permutation
  return permute(arg, outer(perm));
}

///
/// scale
///

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> scale(const UMTensor<T> &arg, const Scalar factor) {
  auto &queue = blasqueue_for(arg.range());
  return device::impl::scale(arg, factor, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> &scale_to(UMTensor<T> &arg, const Scalar factor) {
  auto &queue = blasqueue_for(arg.range());
  return device::impl::scale_to(arg, factor, queue);
}

template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
UMTensor<T> scale(const UMTensor<T> &arg, const Scalar factor,
                  const Perm &perm) {
  auto result = permute(arg, perm);
  auto &queue = blasqueue_for(result.range());
  device::impl::scale_to(result, factor, queue);
  return result;
}

///
/// neg
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> neg(const UMTensor<T> &arg) {
  return scale(arg, T(-1.0));
}

template <typename T, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> neg(const UMTensor<T> &arg, const Perm &perm) {
  auto result = permute(arg, perm);
  auto &queue = blasqueue_for(result.range());
  device::impl::scale_to(result, T(-1.0), queue);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &neg_to(UMTensor<T> &arg) {
  return scale_to(arg, T(-1.0));
}

///
/// add
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> add(const UMTensor<T> &arg1, const UMTensor<T> &arg2) {
  auto &queue = blasqueue_for(arg1.range());
  return device::impl::add(arg1, arg2, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> add(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                const Scalar factor) {
  auto &queue = blasqueue_for(arg1.range());
  auto result = device::impl::add(arg1, arg2, queue);
  return device::impl::scale_to(result, factor, queue);
}

template <typename T, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> add(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                const Perm &perm) {
  auto result = add(arg1, arg2);
  return permute(result, perm);
}

template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
UMTensor<T> add(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                const Scalar factor, const Perm &perm) {
  auto result = add(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// add_to
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &add_to(UMTensor<T> &result, const UMTensor<T> &arg) {
  auto &queue = blasqueue_for(result.range());
  return device::impl::add_to(result, arg, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> &add_to(UMTensor<T> &result, const UMTensor<T> &arg,
                    const Scalar factor) {
  auto &queue = blasqueue_for(result.range());
  device::impl::add_to(result, arg, queue);
  return device::impl::scale_to(result, factor, queue);
}

///
/// subt
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> subt(const UMTensor<T> &arg1, const UMTensor<T> &arg2) {
  UMTensor<T> result(arg1.range());

  auto &queue = blasqueue_for(result.range());
  return device::impl::subt(arg1, arg2, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> subt(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 const Scalar factor) {
  auto &queue = blasqueue_for(arg1.range());
  auto result = device::impl::subt(arg1, arg2, queue);
  return device::impl::scale_to(result, factor, queue);
}

template <typename T, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> subt(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 const Perm &perm) {
  auto result = subt(arg1, arg2);
  return permute(result, perm);
}

template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
UMTensor<T> subt(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 const Scalar factor, const Perm &perm) {
  auto result = subt(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// subt_to
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &subt_to(UMTensor<T> &result, const UMTensor<T> &arg) {
  auto &queue = blasqueue_for(result.range());
  return device::impl::subt_to(result, arg, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> &subt_to(UMTensor<T> &result, const UMTensor<T> &arg,
                     const Scalar factor) {
  auto &queue = blasqueue_for(result.range());
  device::impl::subt_to(result, arg, queue);
  return device::impl::scale_to(result, factor, queue);
}

///
/// mult
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> mult(const UMTensor<T> &arg1, const UMTensor<T> &arg2) {
  TA_ASSERT(arg1.size() == arg2.size());
  auto &queue = blasqueue_for(arg1.range());
  return device::impl::mult(arg1, arg2, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> mult(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 const Scalar factor) {
  auto &queue = blasqueue_for(arg1.range());
  auto result = device::impl::mult(arg1, arg2, queue);
  return device::impl::scale_to(result, factor, queue);
}

template <typename T, typename Perm>
  requires TiledArray::detail::is_permutation_v<Perm> &&
           TiledArray::detail::is_numeric_v<T>
UMTensor<T> mult(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 const Perm &perm) {
  auto result = mult(arg1, arg2);
  return permute(result, perm);
}

template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
UMTensor<T> mult(const UMTensor<T> &arg1, const UMTensor<T> &arg2,
                 const Scalar factor, const Perm &perm) {
  auto result = mult(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// mult_to
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
UMTensor<T> &mult_to(UMTensor<T> &result, const UMTensor<T> &arg) {
  TA_ASSERT(result.size() == arg.size());
  auto &queue = blasqueue_for(result.range());
  return device::impl::mult_to(result, arg, queue);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
UMTensor<T> &mult_to(UMTensor<T> &result, const UMTensor<T> &arg,
                     const Scalar factor) {
  auto &queue = blasqueue_for(result.range());
  device::impl::mult_to(result, arg, queue);
  return device::impl::scale_to(result, factor, queue);
}

///
/// dot
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T dot(const UMTensor<T> &arg1, const UMTensor<T> &arg2) {
  auto &queue = blasqueue_for(arg1.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg1);
  detail::to_device(arg2);

  // compute dot product using device BLAS
  auto result = T(0);
  blas::dot(arg1.size(), device_data(arg1), 1, device_data(arg2), 1, &result,
            queue);
  device::sync_madness_task_with(stream);
  return result;
}

///
/// Reduction
///

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T squared_norm(const UMTensor<T> &arg) {
  auto &queue = blasqueue_for(arg.range());
  const auto stream = device::Stream(queue.device(), queue.stream());

  detail::to_device(arg);

  // compute squared norm using dot
  auto result = T(0);
  blas::dot(arg.size(), device_data(arg), 1, device_data(arg), 1, &result,
            queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T norm(const UMTensor<T> &arg) {
  return std::sqrt(squared_norm(arg));
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T sum(const UMTensor<T> &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result = device::sum_kernel(device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T product(const UMTensor<T> &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result = device::product_kernel(device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T max(const UMTensor<T> &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result = device::max_kernel(device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T min(const UMTensor<T> &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result = device::min_kernel(device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T abs_max(const UMTensor<T> &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result = device::absmax_kernel(device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
T abs_min(const UMTensor<T> &arg) {
  detail::to_device(arg);
  auto stream = device::stream_for(arg.range());
  auto result = device::absmin_kernel(device_data(arg), arg.size(), stream);
  device::sync_madness_task_with(stream);

  return result;
}

/// convert array from UMTensor to TiledArray::Tensor
template <typename UMT, typename TATensor, typename Policy>
TiledArray::DistArray<TATensor, Policy> um_tensor_to_ta_tensor(
    const TiledArray::DistArray<UMT, Policy> &um_array) {
  if constexpr (std::is_same_v<UMT, TATensor>) {
    // No-op if UMTensor is the same type as TATensor type
    return um_array;
  } else {
    const auto convert_tile_memcpy = [](const UMT &tile) {
      TATensor result(tile.range());

      auto stream = device::stream_for(result.range());
      DeviceSafeCall(device::memcpyAsync(
          result.data(), tile.data(),
          tile.size() * sizeof(typename TATensor::value_type),
          device::MemcpyDefault, stream));
      device::sync_madness_task_with(stream);

      return result;
    };

    const auto convert_tile_um = [](const UMT &tile) {
      TATensor result(tile.range());
      using std::begin;
      const auto n = tile.size();

      auto stream = device::stream_for(tile.range());

      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Host>(tile,
                                                                       stream);

      std::copy_n(tile.data(), n, result.data());

      return result;
    };

    const char *use_legacy_conversion =
        std::getenv("TA_DEVICE_LEGACY_UM_CONVERSION");
    auto ta_array = use_legacy_conversion
                        ? to_new_tile_type(um_array, convert_tile_um)
                        : to_new_tile_type(um_array, convert_tile_memcpy);

    um_array.world().gop.fence();
    return ta_array;
  }
}

/// convert array from TiledArray::Tensor to UMTensor
template <typename UMT, typename TATensor, typename Policy>
TiledArray::DistArray<UMT, Policy> ta_tensor_to_um_tensor(
    const TiledArray::DistArray<TATensor, Policy> &array) {
  if constexpr (std::is_same_v<UMT, TATensor>) {
    // No-op if array is the same as return type
    return array;
  } else {
    using inT = typename TATensor::value_type;
    using outT = typename UMT::value_type;
    // check if element conversion is necessary
    constexpr bool T_conversion = !std::is_same_v<inT, outT>;

    // this is safe even when need to convert element types, but less efficient
    auto convert_tile_um = [](const TATensor &tile) {
      /// UMTensor must be wrapped into TA::Tile
      UMT result(tile.range());

      const auto n = tile.size();
      std::copy_n(tile.data(), n, result.data());

      auto stream = device::stream_for(result.range());

      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Device>(
          result, stream);

      // N.B. move! without it have D-to-H transfer due to calling UM
      // allocator construct() on the host
      return std::move(result);
    };

    TiledArray::DistArray<UMT, Policy> um_array;
    if constexpr (T_conversion) {
      um_array = to_new_tile_type(array, convert_tile_um);
    } else {
      // this is more efficient for copying:
      // - avoids copy on host followed by UM transfer, instead uses direct copy
      // - replaced unneeded copy (which also caused D-to-H transfer due to
      // calling UM allocator construct() on the host) by move
      // This eliminates all spurious UM traffic in (T) W3 contractions
      auto convert_tile_memcpy = [](const TATensor &tile) {
        /// UMTensor must be wrapped into TA::Tile .. Why?

        auto stream = device::stream_for(tile.range());
        UMT result(tile.range());

        DeviceSafeCall(
            device::memcpyAsync(result.data(), tile.data(),
                                tile.size() * sizeof(typename UMT::value_type),
                                device::MemcpyDefault, stream));

        device::sync_madness_task_with(stream);
        // N.B. move! without it have D-to-H transfer due to calling UM
        // allocator construct() on the host
        return std::move(result);
      };

      const char *use_legacy_conversion =
          std::getenv("TA_DEVICE_LEGACY_UM_CONVERSION");
      um_array = use_legacy_conversion
                     ? to_new_tile_type(array, convert_tile_um)
                     : to_new_tile_type(array, convert_tile_memcpy);
    }

    array.world().gop.fence();
    return um_array;
  }
}

}  // namespace TiledArray

/// Serialization support
namespace madness {
namespace archive {

template <typename Archive, typename T>
  requires TiledArray::detail::is_numeric_v<T>
struct ArchiveStoreImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void store(const Archive &ar,
                           const TiledArray::UMTensor<T> &t) {
    ar & t.range();
    ar & t.nbatch();
    if (t.range().volume() > 0) {
      auto stream = TiledArray::device::stream_for(t.range());
      TiledArray::to_execution_space<TiledArray::ExecutionSpace::Host>(t,
                                                                       stream);
      ar &madness::archive::wrap(t.data(), t.range().volume() * t.nbatch());
    }
  }
};

template <typename Archive, typename T>
  requires TiledArray::detail::is_numeric_v<T>
struct ArchiveLoadImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void load(const Archive &ar, TiledArray::UMTensor<T> &t) {
    TiledArray::Range range{};
    size_t nbatch{};
    ar & range;
    ar & nbatch;
    if (range.volume() > 0) {
      t = TiledArray::UMTensor<T>(std::move(range), nbatch);
      ar &madness::archive::wrap(t.data(), range.volume() * nbatch);
    }
  }
};

}  // namespace archive
}  // namespace madness

#ifndef TILEDARRAY_HEADER_ONLY

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

#endif  // TILEDARRAY_HEADER_ONLY

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_UM_TENSOR_H
