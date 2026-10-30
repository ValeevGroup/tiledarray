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

#ifndef TILEDARRAY_DEVICE_DISPATCH_H
#define TILEDARRAY_DEVICE_DISPATCH_H

#include <TiledArray/config.h>
#include <TiledArray/fwd.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/platform.h>
#include <TiledArray/tensor/type_traits.h>

#include <complex>
#include <type_traits>

/// Device kernels behind TA::Tensor's tile ops. A TA::Tensor member runs the
/// device kernel when its operands satisfy device_executable_v and
/// execution_space() is Device, and its host code otherwise.
///
/// Everything here is declared in every build, so that TA::Tensor needs no
/// preprocessor guards: without device support device_executable_v is always
/// false, and the kernels are never instantiated. With it, they are compiled
/// once, in device/tensor.cpp, for the element types admitted by
/// device_executable_v.
///
/// Each kernel queues its work on a stream picked by its Range argument (see
/// blasqueue_for) and calls device::sync_madness_task_with; inside a device
/// task every later kernel reuses the stream the first one recorded, so the
/// kernels of one task run in order. Host code that reads data a kernel of the
/// same task wrote must call device::synchronize() first.
///
/// Whether a tile op runs in a device task at all is decided earlier, by
/// detail::is_device_tile; under a Host scope, dist_eval/unary_eval.h and
/// binary_eval.h skip the device task for types satisfying
/// device_executable_v.

namespace TiledArray {
namespace detail {

template <typename T>
inline constexpr bool is_device_tensor_v = false;

#ifdef TILEDARRAY_HAS_DEVICE
template <typename T>
  requires requires {
    typename T::value_type;
    typename T::allocator_type;
  }
inline constexpr bool is_device_tensor_v<T> =
    std::is_same_v<typename T::allocator_type,
                   device_um_allocator<typename T::value_type>> &&
    (std::is_same_v<typename T::value_type, float> ||
     std::is_same_v<typename T::value_type, double> ||
     std::is_same_v<typename T::value_type, std::complex<float>> ||
     std::is_same_v<typename T::value_type, std::complex<double>>);

/// UMTensor tiles of the element types the device kernels support are device
/// tiles: the expression engine normally routes their tile ops through
/// madness::add_device_task; under a Host scope, eval_tile runs them directly
/// (see dist_eval). Defined here, next to the trait it uses, so that every
/// translation unit that uses TA::Tensor sees it.
template <typename T>
struct is_device_tile<UMTensor<T>>
    : public std::bool_constant<is_device_tensor_v<UMTensor<T>>> {};
#endif

}  // namespace detail

/// true if \c T is a unified-memory tensor of a device-supported element type
/// and \c Ts... are all \c T, i.e. an op on these operands may run on the
/// device
template <typename T, typename... Ts>
inline constexpr bool device_executable_v =
    detail::is_device_tensor_v<T> && (std::is_same_v<T, Ts> && ...);

/// Runs the statement that follows it, which must end by returning, instead
/// of the host code that follows, when the tensor types in the argument list
/// satisfy device_executable_v and execution_space() is Device.
/// The statement is the macro's last \c else, so following it with another
/// \c else does not compile.
#define TA_ON_DEVICE(...)                                          \
  if constexpr (!::TiledArray::device_executable_v<__VA_ARGS__>) { \
  } else if (::TiledArray::execution_space() !=                    \
             ::TiledArray::ExecutionSpace::Device) {               \
  } else

namespace device {

using math::blas::integer;

/// row-major C = alpha * op(A) * op(B) + beta * C, with the semantics of
/// math::blas::gemm
template <typename T>
void gemm(math::blas::Op op_a, math::blas::Op op_b, integer m, integer n,
          integer k, T alpha, const T* a, integer lda, const T* b, integer ldb,
          T beta, T* c, integer ldc, const Range& range);

/// y[i] = x[i]
template <typename T>
void copy(integer n, const T* x, T* y, const Range& range);

/// x[i] *= alpha
template <typename T>
void scal(integer n, T alpha, T* x, const Range& range);

/// y[i] += alpha * x[i]
template <typename T>
void axpy(integer n, T alpha, const T* x, T* y, const Range& range);

/// c[i] = a[i] * b[i]
template <typename T>
void mult(integer n, const T* a, const T* b, T* c, const Range& range);

/// c[i] *= a[i]
template <typename T>
void mult_to(integer n, const T* a, T* c, const Range& range);

/// out = perm ^ in, where \p in spans \p in_range
/// \throw TiledArray::Exception if \p in_range has more than INT_MAX elements
template <typename T>
void permute(const T* in, T* out, const Range& in_range,
             const Permutation& perm);

/// \return sum_i x[i] * y[i]
template <typename T>
T dot(integer n, const T* x, const T* y, const Range& range);

/// \return sum_i conj(x[i]) * y[i]
template <typename T>
T inner_product(integer n, const T* x, const T* y, const Range& range);

/// blocks until the device work queued so far by this task completes
void synchronize();

}  // namespace device
}  // namespace TiledArray

#endif  // TILEDARRAY_DEVICE_DISPATCH_H
