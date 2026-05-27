/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 */

#ifndef TILEDARRAY_DEVICE_TENSOR_H
#define TILEDARRAY_DEVICE_TENSOR_H

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/conversions/to_new_tile_type.h>
#include <TiledArray/device/blas.h>
#include <TiledArray/device/kernel/mult_kernel.h>
#include <TiledArray/external/device.h>
#include <TiledArray/external/librett.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tensor/complex.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tile.h>

#include <madness/world/archive.h>
#include <blas.hh>

namespace TiledArray {
namespace detail {

/// UMTensor lives in unified memory; it is identified as a device_tile and
/// the expression engine must route its tile ops through
/// madness::add_device_task.
template <typename T>
struct is_device_tile<TiledArray::UMTensor<T>>
    : public std::bool_constant<TiledArray::detail::is_numeric_v<T>> {};

/// Prefetch a UMTensor's storage to the device associated with its tile range.
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_device(const TiledArray::UMTensor<T>& tile) {
  if (tile.empty()) return;
  auto stream = device::stream_for(tile.range());
  if (deviceEnv::instance()->concurrent_managed_access()) {
    DeviceSafeCall(device::memPrefetchAsync(tile.data(),
                                            tile.total_size() * sizeof(T),
                                            stream.device, stream.stream));
  }
}

/// Prefetch a UMTensor's storage back to the host.
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_host(const TiledArray::UMTensor<T>& tile) {
  if (tile.empty()) return;
  auto stream = device::stream_for(tile.range());
  if (deviceEnv::instance()->concurrent_managed_access()) {
    DeviceSafeCall(
        device::memPrefetchAsync(tile.data(), tile.total_size() * sizeof(T),
                                 device::CpuDeviceId, stream.stream));
  }
}

}  // namespace detail

// clang-format off
/// Tile-op overloads for UMTensor.
///
/// Each overload sits in `namespace TiledArray` so ADL finds it from the
/// expression engine and from the tile_op layer's free-function defaults.
/// More-specialized concrete-type overloads win against the generic
/// forwarder in `tile_op/tile_interface.h`:
/// \code
/// template <typename Left, typename Right>
/// auto add(Left&& left, Right&& right) {
///   return left.add(right);
/// }
/// \endcode
/// so we never fall back to the CPU member functions for UMTensor.
///
/// All overloads follow the stream/queue contract:
///   1. Resolve a queue via `blasqueue_for(range)`. Inside a device task
///      this is the same queue everyone else in the task uses (see
///      `external/device.h:899-907`); outside one, it round-robins.
///   2. Prefetch every input + the result to the device.
///   3. Call into BLAS++ / device kernels on that queue.
///   4. `sync_madness_task_with(stream)` so the enclosing MADNESS device
///      task waits for the queue to drain before completing.
///
/// In-place ops provide both an lvalue and an rvalue overload: the lvalue
/// overload does the work, the rvalue overload forwards to it.
///
/// nbatch_ > 1 is not yet supported; the host-side tile
/// ops don't support them either.
// clang-format on

/// result[i] = arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> clone(const UMTensor<T>& arg) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(arg.nbatch() == 1);

  auto& queue = blasqueue_for(arg.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(arg.range());

  detail::to_device(arg);
  detail::to_device(result);

  blas::copy(result.size(), arg.data(), 1, result.data(), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

namespace detail {

/// Apply a scaling factor in-place on the device, replicating the
/// ComplexConjugate handling from device/btas.h::scale. Real-valued kernels
/// reduce to a single `blas::scal`; conjugation+scale on complex tiles
/// requires a custom kernel that we have not implemented yet.
template <typename T, typename Scalar>
inline void apply_scale_factor(T* data, std::size_t n, const Scalar factor,
                               ::blas::Queue& queue) {
  if constexpr (TiledArray::detail::is_blas_numeric_v<Scalar> ||
                std::is_arithmetic_v<Scalar>) {
    ::blas::scal(n, factor, data, 1, queue);
  } else if constexpr (TiledArray::detail::is_complex_v<T>) {
    TA_EXCEPTION(
        "UMTensor scale with ComplexConjugate factor on complex T is not "
        "implemented (requires a fused conjugation kernel)");
  } else if constexpr (std::is_same_v<
                           Scalar,
                           TiledArray::detail::ComplexConjugate<void>>) {
    // conjugation on a real tensor is a no-op
  } else if constexpr (std::is_same_v<Scalar,
                                      TiledArray::detail::ComplexConjugate<
                                          TiledArray::detail::ComplexNegTag>>) {
    ::blas::scal(n, static_cast<T>(-1), data, 1, queue);
  }
}

}  // namespace detail

/// result[i] = arg[i] * factor
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T> scale(const UMTensor<T>& arg, const Scalar factor) {
  auto result = clone(arg);
  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  detail::apply_scale_factor(result.data(), result.size(), factor, queue);
  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] *= factor (in-place)
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& scale_to(UMTensor<T>& result, const Scalar factor) {
  TA_ASSERT(!result.empty());
  TA_ASSERT(result.nbatch() == 1);
  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));
  detail::to_device(result);
  detail::apply_scale_factor(result.data(), result.size(), factor, queue);
  device::sync_madness_task_with(stream);
  return result;
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& scale_to(UMTensor<T>&& result, const Scalar factor) {
  return scale_to(result, factor);
}

/// result[i] = -arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> neg(const UMTensor<T>& arg) {
  return scale(arg, T(-1));
}

/// arg[i] = -arg[i] (in-place)
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& neg_to(UMTensor<T>& arg) {
  return scale_to(arg, T(-1));
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& neg_to(UMTensor<T>&& arg) {
  return neg_to(arg);
}

/// result[i] = arg1[i] + arg2[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2) {
  TA_ASSERT(!arg1.empty());
  TA_ASSERT(!arg2.empty());
  TA_ASSERT(arg1.nbatch() == 1 && arg2.nbatch() == 1);

  auto& queue = blasqueue_for(arg1.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(arg1.range());

  detail::to_device(arg1);
  detail::to_device(arg2);
  detail::to_device(result);

  ::blas::copy(result.size(), arg1.data(), 1, result.data(), 1, queue);
  ::blas::axpy(result.size(), T(1), arg2.data(), 1, result.data(), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] = (arg1[i] + arg2[i]) * factor
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                       const Scalar factor) {
  auto result = add(arg1, arg2);
  return scale_to(result, factor);
}

/// result[i] += arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& add_to(UMTensor<T>& result, const UMTensor<T>& arg) {
  TA_ASSERT(!result.empty());
  TA_ASSERT(!arg.empty());
  TA_ASSERT(result.nbatch() == 1 && arg.nbatch() == 1);

  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(result);
  detail::to_device(arg);

  ::blas::axpy(result.size(), T(1), arg.data(), 1, result.data(), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& add_to(UMTensor<T>&& result, const UMTensor<T>& arg) {
  return add_to(result, arg);
}

/// result[i] = (result[i] + arg[i]) * factor
/// Matches TA::Tensor::add_to(right, factor) semantics: `(l += r) *= factor`.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& add_to(UMTensor<T>& result, const UMTensor<T>& arg,
                           const Scalar factor) {
  add_to(result, arg);
  return scale_to(result, factor);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& add_to(UMTensor<T>&& result, const UMTensor<T>& arg,
                           const Scalar factor) {
  return add_to(result, arg, factor);
}

/// result[i] = arg1[i] - arg2[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2) {
  TA_ASSERT(!arg1.empty());
  TA_ASSERT(!arg2.empty());
  TA_ASSERT(arg1.nbatch() == 1 && arg2.nbatch() == 1);

  auto& queue = blasqueue_for(arg1.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(arg1.range());

  detail::to_device(arg1);
  detail::to_device(arg2);
  detail::to_device(result);

  ::blas::copy(result.size(), arg1.data(), 1, result.data(), 1, queue);
  ::blas::axpy(result.size(), T(-1), arg2.data(), 1, result.data(), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] = (arg1[i] - arg2[i]) * factor
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor) {
  auto result = subt(arg1, arg2);
  return scale_to(result, factor);
}

/// result[i] -= arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& subt_to(UMTensor<T>& result, const UMTensor<T>& arg) {
  TA_ASSERT(!result.empty());
  TA_ASSERT(!arg.empty());
  TA_ASSERT(result.nbatch() == 1 && arg.nbatch() == 1);

  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(result);
  detail::to_device(arg);

  ::blas::axpy(result.size(), T(-1), arg.data(), 1, result.data(), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& subt_to(UMTensor<T>&& result, const UMTensor<T>& arg) {
  return subt_to(result, arg);
}

/// result[i] = (result[i] - arg[i]) * factor
/// Matches TA::Tensor::subt_to(right, factor) semantics: `(l -= r) *= factor`.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& subt_to(UMTensor<T>& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  subt_to(result, arg);
  return scale_to(result, factor);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& subt_to(UMTensor<T>&& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  return subt_to(result, arg, factor);
}

/// dot product: scalar = sum_i arg1[i] * arg2[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline T dot(const UMTensor<T>& arg1, const UMTensor<T>& arg2) {
  TA_ASSERT(!arg1.empty());
  TA_ASSERT(!arg2.empty());
  TA_ASSERT(arg1.nbatch() == 1 && arg2.nbatch() == 1);
  TA_ASSERT(arg1.size() == arg2.size());

  auto& queue = blasqueue_for(arg1.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(arg1);
  detail::to_device(arg2);

  T result(0);
  ::blas::dot(arg1.size(), arg1.data(), 1, arg2.data(), 1, &result, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// scalar = sum_i arg[i] * arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline auto squared_norm(const UMTensor<T>& arg) {
  return dot(arg, arg);
}

/// scalar = sqrt(squared_norm(arg))
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline auto norm(const UMTensor<T>& arg) {
  using std::sqrt;
  using ResultType = TiledArray::detail::scalar_t<T>;
  return static_cast<ResultType>(sqrt(squared_norm(arg)));
}

/// result[perm(i)] = arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> permute(const UMTensor<T>& arg,
                           const TiledArray::Permutation& perm) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(arg.nbatch() == 1);
  TA_ASSERT(perm.size() == arg.range().rank());

  auto result_range = perm * arg.range();
  auto& queue = blasqueue_for(result_range);
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(result_range);

  detail::to_device(arg);
  detail::to_device(result);

  // librett operates on the original (unpermuted) range and writes into the
  // permuted layout; pointers go in as-is.
  librett_permute(const_cast<T*>(arg.data()), result.data(), arg.range(), perm,
                  stream.stream);

  device::sync_madness_task_with(stream);
  return result;
}

/// BipartitePermutation -> plain Permutation forward.
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> permute(const UMTensor<T>& arg,
                           const TiledArray::BipartitePermutation& perm) {
  TA_ASSERT(inner_size(perm) == 0);  // UMTensor is a non-nested tile
  return permute(arg, outer(perm));
}

/// result[perm(i)] = arg[i] * factor
template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> scale(const UMTensor<T>& arg, const Scalar factor,
                         const Perm& perm) {
  auto scaled = scale(arg, factor);
  return permute(scaled, perm);
}

/// result[perm(i)] = -arg[i]
template <typename T, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> neg(const UMTensor<T>& arg, const Perm& perm) {
  return permute(neg(arg), perm);
}

/// result[perm(i)] = arg1[i] + arg2[i]
template <typename T, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                       const Perm& perm) {
  return permute(add(arg1, arg2), perm);
}

/// result[perm(i)] = (arg1[i] + arg2[i]) * factor
template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                       const Scalar factor, const Perm& perm) {
  return permute(add(arg1, arg2, factor), perm);
}

/// result[perm(i)] = arg1[i] - arg2[i]
template <typename T, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Perm& perm) {
  return permute(subt(arg1, arg2), perm);
}

/// result[perm(i)] = (arg1[i] - arg2[i]) * factor
template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor, const Perm& perm) {
  return permute(subt(arg1, arg2, factor), perm);
}

/// shift: result has arg's data, range shifted by bound_shift.
template <typename T, typename Index>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> shift(const UMTensor<T>& arg, const Index& bound_shift) {
  TA_ASSERT(!arg.empty());
  TA_ASSERT(arg.nbatch() == 1);

  TiledArray::Range result_range(arg.range());
  result_range.inplace_shift(bound_shift);

  auto& queue = blasqueue_for(result_range);
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(result_range);

  detail::to_device(arg);
  detail::to_device(result);

  ::blas::copy(result.size(), arg.data(), 1, result.data(), 1, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// shift_to: in-place range shift, no data movement.
template <typename T, typename Index>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& shift_to(UMTensor<T>& arg, const Index& bound_shift) {
  return arg.shift_to(bound_shift);
}

template <typename T, typename Index>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& shift_to(UMTensor<T>&& arg, const Index& bound_shift) {
  return shift_to(arg, bound_shift);
}

/// result[i] = arg1[i] * arg2[i] (element-wise / Hadamard)
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2) {
  TA_ASSERT(!arg1.empty());
  TA_ASSERT(!arg2.empty());
  TA_ASSERT(arg1.size() == arg2.size());
  TA_ASSERT(arg1.nbatch() == 1 && arg2.nbatch() == 1);

  auto& queue = blasqueue_for(arg1.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(arg1.range());

  detail::to_device(arg1);
  detail::to_device(arg2);
  detail::to_device(result);

  device::mult_kernel(result.data(), arg1.data(), arg2.data(), arg1.size(),
                      stream);

  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] = arg1[i] * arg2[i] * factor
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor) {
  auto result = mult(arg1, arg2);
  return scale_to(result, factor);
}

/// result[perm(i)] = arg1[i] * arg2[i]
template <typename T, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Perm& perm) {
  return permute(mult(arg1, arg2), perm);
}

/// result[perm(i)] = arg1[i] * arg2[i] * factor
template <typename T, typename Scalar, typename Perm>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar> &&
           TiledArray::detail::is_permutation_v<Perm>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor, const Perm& perm) {
  return permute(mult(arg1, arg2, factor), perm);
}

/// result[i] *= arg[i]
template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& mult_to(UMTensor<T>& result, const UMTensor<T>& arg) {
  TA_ASSERT(!result.empty());
  TA_ASSERT(!arg.empty());
  TA_ASSERT(result.size() == arg.size());
  TA_ASSERT(result.nbatch() == 1 && arg.nbatch() == 1);

  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(result);
  detail::to_device(arg);

  device::mult_to_kernel(result.data(), arg.data(), result.size(), stream);

  device::sync_madness_task_with(stream);
  return result;
}

template <typename T>
  requires TiledArray::detail::is_numeric_v<T>
inline UMTensor<T>& mult_to(UMTensor<T>&& result, const UMTensor<T>& arg) {
  return mult_to(result, arg);
}

/// result[i] = (result[i] * arg[i]) * factor
/// Matches TA::Tensor::mult_to(right, factor) semantics: `(l *= r) *= factor`.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& mult_to(UMTensor<T>& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  mult_to(result, arg);
  return scale_to(result, factor);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& mult_to(UMTensor<T>&& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  return mult_to(result, arg, factor);
}

/// gemm: result = factor * left * right
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T> gemm(const UMTensor<T>& left, const UMTensor<T>& right,
                        const Scalar factor,
                        const TiledArray::math::GemmHelper& gemm_helper) {
  TA_ASSERT(!left.empty());
  TA_ASSERT(!right.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());
  TA_ASSERT(left.nbatch() == 1 && right.nbatch() == 1);

  TA_ASSERT(gemm_helper.left_right_congruent(left.range().extent_data(),
                                             right.range().extent_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.left_right_congruent(left.range().lobound_data(),
                                             right.range().lobound_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.left_right_congruent(left.range().upbound_data(),
                                             right.range().upbound_data()));

  auto result_range = gemm_helper.template make_result_range<TiledArray::Range>(
      left.range(), right.range());

  auto& queue = blasqueue_for(result_range);
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  UMTensor<T> result(result_range);
  TA_ASSERT(result.nbatch() == 1);

  detail::to_device(left);
  detail::to_device(right);
  detail::to_device(result);

  using TiledArray::math::blas::integer;
  integer m = 1, n = 1, k = 1;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  const integer lda = std::max(
      integer{1},
      gemm_helper.left_op() == TiledArray::math::blas::Op::NoTrans ? k : m);
  const integer ldb = std::max(
      integer{1},
      gemm_helper.right_op() == TiledArray::math::blas::Op::NoTrans ? n : k);
  const integer ldc = std::max(integer{1}, n);

  const T factor_t = T(factor);
  const T zero(0);

  // Match btas device gemm (device/btas.h): col-major view with right/left
  // swapped reproduces TA::Tensor's row-major layout under cublas.
  ::blas::gemm(::blas::Layout::ColMajor, gemm_helper.right_op(),
               gemm_helper.left_op(), n, m, k, factor_t, right.data(), ldb,
               left.data(), lda, zero, result.data(), ldc, queue);

  device::sync_madness_task_with(stream);
  return result;
}

/// gemm: result += factor * left * right
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<T> &&
           TiledArray::detail::is_numeric_v<Scalar>
inline void gemm(UMTensor<T>& result, const UMTensor<T>& left,
                 const UMTensor<T>& right, const Scalar factor,
                 const TiledArray::math::GemmHelper& gemm_helper) {
  TA_ASSERT(!result.empty());
  TA_ASSERT(!left.empty());
  TA_ASSERT(!right.empty());
  TA_ASSERT(result.range().rank() == gemm_helper.result_rank());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());
  TA_ASSERT(left.nbatch() == 1 && right.nbatch() == 1 && result.nbatch() == 1);

  TA_ASSERT(gemm_helper.left_result_congruent(left.range().extent_data(),
                                              result.range().extent_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.left_result_congruent(left.range().lobound_data(),
                                              result.range().lobound_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.left_result_congruent(left.range().upbound_data(),
                                              result.range().upbound_data()));
  TA_ASSERT(gemm_helper.right_result_congruent(right.range().extent_data(),
                                               result.range().extent_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.right_result_congruent(right.range().lobound_data(),
                                               result.range().lobound_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.right_result_congruent(right.range().upbound_data(),
                                               result.range().upbound_data()));
  TA_ASSERT(gemm_helper.left_right_congruent(left.range().extent_data(),
                                             right.range().extent_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.left_right_congruent(left.range().lobound_data(),
                                             right.range().lobound_data()));
  TA_ASSERT(ignore_tile_position() ||
            gemm_helper.left_right_congruent(left.range().upbound_data(),
                                             right.range().upbound_data()));

  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(device::setDevice(stream.device));

  detail::to_device(left);
  detail::to_device(right);
  detail::to_device(result);

  using TiledArray::math::blas::integer;
  integer m = 1, n = 1, k = 1;
  gemm_helper.compute_matrix_sizes(m, n, k, left.range(), right.range());

  const integer lda = std::max(
      integer{1},
      gemm_helper.left_op() == TiledArray::math::blas::Op::NoTrans ? k : m);
  const integer ldb = std::max(
      integer{1},
      gemm_helper.right_op() == TiledArray::math::blas::Op::NoTrans ? n : k);
  const integer ldc = std::max(integer{1}, n);

  const T factor_t = T(factor);
  const T one(1);

  ::blas::gemm(::blas::Layout::ColMajor, gemm_helper.right_op(),
               gemm_helper.left_op(), n, m, k, factor_t, right.data(), ldb,
               left.data(), lda, one, result.data(), ldc, queue);

  device::sync_madness_task_with(stream);
}

/// Array-level helpers: bulk to-host / to-device prefetch and conversions
/// between UMTensor-backed and host-Tensor-backed DistArrays.

/// Prefetch every local tile of `array` to the host. Fences on the
/// containing world and globally synchronizes the device on exit.
template <typename T, typename Policy>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_host(TiledArray::DistArray<UMTensor<T>, Policy>& array) {
  auto prefetch = [](UMTensor<T>& tile) {
    auto stream = device::stream_for(tile.range());
    detail::to_host(tile);
    device::sync_madness_task_with(stream);
  };
  auto& world = array.world();
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (!array.is_zero(*it)) world.taskq.add(prefetch, array.find(*it));
  }
  world.gop.fence();
  DeviceSafeCall(device::deviceSynchronize());
}

/// Prefetch every local tile of `array` to the device. Fences on the
/// containing world and globally synchronizes the device on exit.
template <typename T, typename Policy>
  requires TiledArray::detail::is_numeric_v<T>
inline void to_device(TiledArray::DistArray<UMTensor<T>, Policy>& array) {
  auto prefetch = [](UMTensor<T>& tile) {
    auto stream = device::stream_for(tile.range());
    detail::to_device(tile);
    device::sync_madness_task_with(stream);
  };
  auto& world = array.world();
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (!array.is_zero(*it)) world.taskq.add(prefetch, array.find(*it));
  }
  world.gop.fence();
  DeviceSafeCall(device::deviceSynchronize());
}

/// Convert a UMTensor-backed `DistArray` to one backed by a host tile type.
template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<!std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<HostTile, Policy>>
um_tensor_to_ta_tensor(const TiledArray::DistArray<UMTile, Policy>& um_array) {
  auto convert_tile = [](const UMTile& tile) {
    auto stream = device::stream_for(tile.range());
    detail::to_host(tile);
    device::sync_madness_task_with(stream);
    HostTile result(tile.range());
    std::copy_n(tile.data(), tile.total_size(), result.data());
    return result;
  };
  auto out = to_new_tile_type<UMTile>(um_array, convert_tile);
  um_array.world().gop.fence();
  return out;
}

template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<UMTile, Policy>>
um_tensor_to_ta_tensor(const TiledArray::DistArray<UMTile, Policy>& um_array) {
  return um_array;
}

/// Convert a host-tile-backed `DistArray` to a UMTensor-backed one.
template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<!std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<UMTile, Policy>>
ta_tensor_to_um_tensor(
    const TiledArray::DistArray<HostTile, Policy>& host_array) {
  auto convert_tile = [](const HostTile& tile) {
    UMTile result(tile.range());
    std::copy_n(tile.data(), tile.total_size(), result.data());
    detail::to_device(result);
    return result;
  };
  auto out = to_new_tile_type<HostTile>(host_array, convert_tile);
  host_array.world().gop.fence();
  return out;
}

template <typename UMTile, typename HostTile, typename Policy>
inline std::enable_if_t<std::is_same_v<UMTile, HostTile>,
                        TiledArray::DistArray<UMTile, Policy>>
ta_tensor_to_um_tensor(
    const TiledArray::DistArray<HostTile, Policy>& host_array) {
  return host_array;
}

}  // namespace TiledArray

/// MADNESS archive specializations for UMTensor.
///
/// `TA::Tensor::serialize(ar)` works on any allocator (the member just walks
/// `data() + range().volume() * nbatch()`), but UM data may be stale on the
/// host if a device kernel is in flight. The Store specialization prefetches
/// the tile back to the host before reading.
namespace madness {
namespace archive {

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::UMTensor<T>& t) {
    if constexpr (TiledArray::detail::is_numeric_v<T>) {
      if (!t.empty()) {
        auto stream = TiledArray::device::stream_for(t.range());
        TiledArray::detail::to_host(t);
        TiledArray::device::sync_madness_task_with(stream);
      }
    }
    const bool empty = t.empty();
    ar & empty;
    if (!empty) {
      ar & t.range();
      ar & t.nbatch();
      ar& madness::archive::wrap(t.data(), t.range().volume() * t.nbatch());
    }
  }
};

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void load(const Archive& ar, TiledArray::UMTensor<T>& t) {
    bool empty = false;
    ar & empty;
    if (!empty) {
      TiledArray::Range range;
      std::size_t nbatch = 1;
      ar & range;
      ar & nbatch;
      t = TiledArray::UMTensor<T>(
          std::move(range), typename TiledArray::UMTensor<T>::nbatches(nbatch));
      ar& madness::archive::wrap(t.data(), t.range().volume() * t.nbatch());
    } else {
      t = TiledArray::UMTensor<T>();
    }
  }
};

}  // namespace archive
}  // namespace madness

/// extern template declarations for the UMTensor class.
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

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_TENSOR_H
