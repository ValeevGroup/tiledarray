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

#include <blas.hh>
#include <madness/world/archive.h>

namespace TiledArray {
namespace detail {

/// `UMTensor` lives in unified memory; the expression engine must route its
/// tile ops through `madness::add_device_task`. The pass-through specs for
/// `Tile<T>` and `LazyArrayTile<T, Op>` in tensor/type_traits.h pick this up.
template <typename T>
struct is_device_tile<TiledArray::UMTensor<T>> : public std::true_type {};

/// Prefetch a UMTensor's storage to the device associated with its tile range.
/// Mirrors the pattern in device/btas_um_tensor.h but reaches the storage via
/// `.data()` + `.total_size()` since `TA::Tensor`'s buffer is a
/// `shared_ptr<T[]>` rather than a varray-like container.
template <typename T>
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
inline void to_host(const TiledArray::UMTensor<T>& tile) {
  if (tile.empty()) return;
  auto stream = device::stream_for(tile.range());
  if (deviceEnv::instance()->concurrent_managed_access()) {
    DeviceSafeCall(device::memPrefetchAsync(tile.data(),
                                            tile.total_size() * sizeof(T),
                                            device::CpuDeviceId, stream.stream));
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// In-place tile ops are tricky to dispatch correctly.
//
// `tile_op/{subt,add,mult,...}.h::Op::eval` passes the result via
// `std::move(...)` when the operand is consumable -- so the engine calls our
// `subt_to`, `add_to`, etc. with an rvalue. A plain `UMTensor<T>&` overload
// is not a viable candidate for an rvalue, so overload resolution falls
// through to the generic forwarder in `tile_op/tile_interface.h` (and
// `tile_interface/scale.h`). That forwarder delegates to TA::Tensor's CPU
// member function, which then reads UM memory while the previous device
// kernel is still in flight on the queue -- silently miscomputing.
//
// To win the dispatch we provide two concrete-type overloads per in-place
// op: one taking `UMTensor<T>&` and one taking `UMTensor<T>&&`. Concrete
// types beat the templated forwarding reference `Result&&` in partial
// ordering regardless of the SFINAE / `requires` constraint shape, so this
// is robust against compiler differences. (A single forwarding-ref overload
// constrained with a `requires UMTensorArg<...>` concept would in principle
// also win because a constrained template subsumes an unconstrained one,
// but g++ does not consistently treat tile_interface's `enable_if`-only
// templates as unconstrained for this purpose -- the result is an ambiguous
// overload error. The two-concrete-overload form sidesteps the question.)
//
// The lvalue overload forwards to the rvalue overload to keep a single
// implementation per op. Value-returning overloads (e.g.
// `add(const UMTensor&, const UMTensor&)`) don't need this because
// reference-to-const binds to both lvalues and rvalues.
//
// The `UMTensorArg` concept is kept around as documentation of intent and
// as a clean handle for any future helper that genuinely wants forwarding
// references (e.g. a `to_device` overload set).
// ---------------------------------------------------------------------------
namespace detail {
template <typename U>
struct is_um_tensor : std::false_type {};
template <typename T>
struct is_um_tensor<UMTensor<T>> : std::true_type {};
template <typename U>
inline constexpr bool is_um_tensor_v =
    is_um_tensor<std::remove_cv_t<std::remove_reference_t<U>>>::value;
}  // namespace detail

template <typename U>
concept UMTensorArg = detail::is_um_tensor_v<U>;

// ---------------------------------------------------------------------------
// Tile-op overloads for UMTensor.
//
// Each overload sits in `namespace TiledArray` so ADL finds it from the
// expression engine and from the tile_op layer's free-function defaults.
// More-specialized concrete-type overloads win against the generic
// `template<typename Left, typename Right> ... add(left, right) { return
// left.add(right); }` forwarders in `tile_op/tile_interface.h`, so we never
// fall back to the CPU member functions for UMTensor.
//
// All overloads follow the stream/queue contract:
//   1. Resolve a queue via `blasqueue_for(range)`. Inside a device task this
//      is the same queue everyone else in the task uses (see
//      `external/device.h:899-907`); outside one, it round-robins.
//   2. Prefetch every input + the result to the device.
//   3. Call into BLAS++ / device kernels on that queue.
//   4. `sync_madness_task_with(stream)` so the enclosing MADNESS device task
//      waits for the queue to drain before completing.
//
// For Phase 2 batched tiles (`nbatch_ > 1`) are not yet supported -- the
// expression engine doesn't currently feed batched UMTensor through these
// paths, and dropping the assertion now would silently miscompute.
// ---------------------------------------------------------------------------

/// result[i] = arg[i]
template <typename T>
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
  } else {
    if constexpr (TiledArray::detail::is_complex_v<T>) {
      abort();  // fused conjugation requires custom kernels, not yet supported
    } else {
      if constexpr (std::is_same_v<
                        Scalar, TiledArray::detail::ComplexConjugate<void>>) {
        // conjugation on a real tensor is a no-op
      } else if constexpr (std::is_same_v<
                               Scalar,
                               TiledArray::detail::ComplexConjugate<
                                   TiledArray::detail::ComplexNegTag>>) {
        ::blas::scal(n, static_cast<T>(-1), data, 1, queue);
      }
    }
  }
}

}  // namespace detail

/// result[i] = arg[i] * factor
template <typename T, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
inline UMTensor<T> scale(const UMTensor<T>& arg, const Scalar factor) {
  auto result = clone(arg);
  auto& queue = blasqueue_for(result.range());
  const device::Stream stream(queue.device(), queue.stream());
  detail::apply_scale_factor(result.data(), result.size(), factor, queue);
  device::sync_madness_task_with(stream);
  return result;
}

/// result[i] *= factor (in-place). Forwarding-reference form so the engine's
/// `scale_to(std::move(tile), factor)` (from `tile_op/scal.h:82`) dispatches
/// here rather than to the tile_interface forwarder that would call the CPU
/// member function on UM memory.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
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
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& scale_to(UMTensor<T>&& result, const Scalar factor) {
  return scale_to(result, factor);
}

/// result[i] = -arg[i]
template <typename T>
inline UMTensor<T> neg(const UMTensor<T>& arg) {
  return scale(arg, T(-1));
}

/// arg[i] = -arg[i] (in-place)
template <typename T>
inline UMTensor<T>& neg_to(UMTensor<T>& arg) {
  return scale_to(arg, T(-1));
}

template <typename T>
inline UMTensor<T>& neg_to(UMTensor<T>&& arg) {
  return scale_to(arg, T(-1));
}

/// result[i] = arg1[i] + arg2[i]
template <typename T>
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
template <typename T, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                       const Scalar factor) {
  auto result = add(arg1, arg2);
  return scale_to(result, factor);
}

/// result[i] += arg[i]
template <typename T>
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
inline UMTensor<T>& add_to(UMTensor<T>&& result, const UMTensor<T>& arg) {
  return add_to(result, arg);
}

/// result[i] = (result[i] + arg[i]) * factor
/// Matches TA::Tensor::add_to(right, factor) semantics: `(l += r) *= factor`.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& add_to(UMTensor<T>& result, const UMTensor<T>& arg,
                           const Scalar factor) {
  add_to(result, arg);
  return scale_to(result, factor);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& add_to(UMTensor<T>&& result, const UMTensor<T>& arg,
                           const Scalar factor) {
  return add_to(result, arg, factor);
}

/// result[i] = arg1[i] - arg2[i]
template <typename T>
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
template <typename T, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor) {
  auto result = subt(arg1, arg2);
  return scale_to(result, factor);
}

/// result[i] -= arg[i]
template <typename T>
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
inline UMTensor<T>& subt_to(UMTensor<T>&& result, const UMTensor<T>& arg) {
  return subt_to(result, arg);
}

/// result[i] = (result[i] - arg[i]) * factor
/// Matches TA::Tensor::subt_to(right, factor) semantics: `(l -= r) *= factor`.
/// This convention is load-bearing for `tile_op/subt.h::Subt::eval` -- when
/// the engine reuses the right operand's storage, it calls
/// `subt_to(std::move(second), first, -1)` and relies on the result being
/// `(second - first) * -1 = first - second`. Hence the forwarding reference
/// on `result`: lvalue-only signatures lose overload resolution to the
/// templated forwarder in tile_op/tile_interface.h, which then dispatches to
/// TA::Tensor's CPU member function and races with any in-flight device
/// kernel on UM memory.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& subt_to(UMTensor<T>& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  subt_to(result, arg);
  return scale_to(result, factor);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& subt_to(UMTensor<T>&& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  return subt_to(result, arg, factor);
}

/// dot product: scalar = sum_i arg1[i] * arg2[i]
template <typename T>
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
inline auto squared_norm(const UMTensor<T>& arg) {
  return dot(arg, arg);
}

/// scalar = sqrt(squared_norm(arg))
template <typename T>
inline auto norm(const UMTensor<T>& arg) {
  using std::sqrt;
  using ResultType = TiledArray::detail::scalar_t<T>;
  return static_cast<ResultType>(sqrt(squared_norm(arg)));
}

/// result[perm(i)] = arg[i]
template <typename T>
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
/// Required to win ADL against the generic CPU member-delegating overload;
/// see the matching warning in device/btas_um_tensor.h:193.
template <typename T>
inline UMTensor<T> permute(const UMTensor<T>& arg,
                           const TiledArray::BipartitePermutation& perm) {
  TA_ASSERT(inner_size(perm) == 0);  // UMTensor is a non-nested tile
  return permute(arg, outer(perm));
}

/// result[perm(i)] = arg[i] * factor
template <typename T, typename Scalar, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> &&
                                      TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> scale(const UMTensor<T>& arg, const Scalar factor,
                         const Perm& perm) {
  auto scaled = scale(arg, factor);
  return permute(scaled, perm);
}

/// result[perm(i)] = -arg[i]
template <typename T, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> neg(const UMTensor<T>& arg, const Perm& perm) {
  return permute(neg(arg), perm);
}

/// result[perm(i)] = arg1[i] + arg2[i]
template <typename T, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                       const Perm& perm) {
  return permute(add(arg1, arg2), perm);
}

/// result[perm(i)] = (arg1[i] + arg2[i]) * factor
template <typename T, typename Scalar, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> &&
                                      TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> add(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                       const Scalar factor, const Perm& perm) {
  return permute(add(arg1, arg2, factor), perm);
}

/// result[perm(i)] = arg1[i] - arg2[i]
template <typename T, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Perm& perm) {
  return permute(subt(arg1, arg2), perm);
}

/// result[perm(i)] = (arg1[i] - arg2[i]) * factor
template <typename T, typename Scalar, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> &&
                                      TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> subt(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor, const Perm& perm) {
  return permute(subt(arg1, arg2, factor), perm);
}

/// shift: result has arg's data, range shifted by bound_shift.
template <typename T, typename Index>
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
inline UMTensor<T>& shift_to(UMTensor<T>& arg, const Index& bound_shift) {
  const_cast<TiledArray::Range&>(arg.range()).inplace_shift(bound_shift);
  return arg;
}

template <typename T, typename Index>
inline UMTensor<T>& shift_to(UMTensor<T>&& arg, const Index& bound_shift) {
  return shift_to(arg, bound_shift);
}

/// result[i] = arg1[i] * arg2[i] (element-wise / Hadamard)
template <typename T>
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
template <typename T, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor) {
  auto result = mult(arg1, arg2);
  return scale_to(result, factor);
}

/// result[perm(i)] = arg1[i] * arg2[i]
template <typename T, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Perm& perm) {
  return permute(mult(arg1, arg2), perm);
}

/// result[perm(i)] = arg1[i] * arg2[i] * factor
template <typename T, typename Scalar, typename Perm,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> &&
                                      TiledArray::detail::is_permutation_v<Perm>>>
inline UMTensor<T> mult(const UMTensor<T>& arg1, const UMTensor<T>& arg2,
                        const Scalar factor, const Perm& perm) {
  return permute(mult(arg1, arg2, factor), perm);
}

/// result[i] *= arg[i]
template <typename T>
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
inline UMTensor<T>& mult_to(UMTensor<T>&& result, const UMTensor<T>& arg) {
  return mult_to(result, arg);
}

/// result[i] = (result[i] * arg[i]) * factor
/// Matches TA::Tensor::mult_to(right, factor) semantics: `(l *= r) *= factor`.
template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& mult_to(UMTensor<T>& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  mult_to(result, arg);
  return scale_to(result, factor);
}

template <typename T, typename Scalar>
  requires TiledArray::detail::is_numeric_v<Scalar>
inline UMTensor<T>& mult_to(UMTensor<T>&& result, const UMTensor<T>& arg,
                            const Scalar factor) {
  return mult_to(result, arg, factor);
}

/// gemm: returning form. result = factor * left * right
template <typename T, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
inline UMTensor<T> gemm(const UMTensor<T>& left, const UMTensor<T>& right,
                        const Scalar factor,
                        const TiledArray::math::GemmHelper& gemm_helper) {
  TA_ASSERT(!left.empty());
  TA_ASSERT(!right.empty());
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());
  TA_ASSERT(left.nbatch() == 1 && right.nbatch() == 1);

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

/// gemm: accumulating form. result += factor * left * right
template <typename T, typename Scalar,
          typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
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

// ---------------------------------------------------------------------------
// Array-level helpers: bulk to-host / to-device prefetch and conversions
// between UMTensor-backed and host-Tensor-backed DistArrays. Mirrors the
// btas-device helpers in btas_um_tensor.h:567-617 but for the bare
// TA::Tensor specialization -- so the tile type is `UMTensor<T>` directly,
// not wrapped in `TA::Tile<...>`.
//
// `to_host` / `to_device` are oneshot bulk-prefetch routines: they walk the
// pmap, dispatch one prefetch task per local tile, fence, then issue a
// `deviceSynchronize` to make sure every stream has drained. They're
// "stop the world" by design -- intended for explicit synchronization
// points (before a host read, after a load, etc.), not for inner loops.
// ---------------------------------------------------------------------------

/// Prefetch every local tile of `array` to the host. Fences on the
/// containing world and globally synchronizes the device on exit.
template <typename T, typename Policy>
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

/// Convert a UMTensor-backed `DistArray` to one backed by host
/// `TA::Tensor<T>`. Tile-by-tile copy through `to_new_tile_type` -- the
/// per-tile lambda allocates a host result, prefetches the source UM
/// buffer to host, and memcpys.
template <typename T, typename Policy>
inline TiledArray::DistArray<TiledArray::Tensor<T>, Policy>
um_tensor_to_ta_tensor(
    const TiledArray::DistArray<UMTensor<T>, Policy>& um_array) {
  auto convert_tile = [](const UMTensor<T>& tile) {
    detail::to_host(tile);
    TiledArray::Tensor<T> result(tile.range());
    std::copy_n(tile.data(), tile.total_size(), result.data());
    return result;
  };
  auto out = to_new_tile_type<UMTensor<T>>(um_array, convert_tile);
  um_array.world().gop.fence();
  return out;
}

/// Convert a host `TA::Tensor<T>`-backed `DistArray` to a UMTensor-backed
/// one. Tile-by-tile copy: allocate UM, memcpy, prefetch to device.
template <typename T, typename Policy>
inline TiledArray::DistArray<UMTensor<T>, Policy> ta_tensor_to_um_tensor(
    const TiledArray::DistArray<TiledArray::Tensor<T>, Policy>& host_array) {
  auto convert_tile = [](const TiledArray::Tensor<T>& tile) {
    UMTensor<T> result(tile.range());
    std::copy_n(tile.data(), tile.total_size(), result.data());
    detail::to_device(result);
    return result;
  };
  auto out = to_new_tile_type<TiledArray::Tensor<T>>(host_array, convert_tile);
  host_array.world().gop.fence();
  return out;
}

}  // namespace TiledArray

// ---------------------------------------------------------------------------
// MADNESS archive specializations for UMTensor.
//
// `TA::Tensor::serialize(ar)` works on any allocator (the member just walks
// `data() + range().volume() * nbatch()`), but UM data may be stale on the
// host if a device kernel is in flight. The Store specialization prefetches
// the tile back to the host before reading. Load goes through the default
// member -- the freshly constructed UM-allocated tile is host-writable, so
// no additional prefetch is needed (downstream code that wants the data on
// the device should call `to_device` explicitly).
// ---------------------------------------------------------------------------
namespace madness {
namespace archive {

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::UMTensor<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::UMTensor<T>& t) {
    TiledArray::detail::to_host(t);
    // Mirror TA::Tensor::serialize's store side; we cannot call the member
    // because it is non-const and we want to keep the input parameter
    // const-correct.
    const bool empty = t.empty();
    ar & empty;
    if (!empty) {
      ar & t.range();
      ar & t.nbatch();
      ar & madness::archive::wrap(t.data(),
                                  t.range().volume() * t.nbatch());
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
      ar & madness::archive::wrap(t.data(),
                                  t.range().volume() * t.nbatch());
    } else {
      t = TiledArray::UMTensor<T>();
    }
  }
};

}  // namespace archive
}  // namespace madness

// ---------------------------------------------------------------------------
// `extern template` declarations for the UMTensor class. Match the explicit
// instantiations in src/TiledArray/device/tensor.cpp so that consumers do
// not re-instantiate the full Tensor<T, device_um_allocator<T>> class body
// in each TU. (Mirrors the analogous pattern at the bottom of
// src/TiledArray/tensor/tensor.h for the host-side instantiations.)
// ---------------------------------------------------------------------------
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
