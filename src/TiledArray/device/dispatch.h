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

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/fwd.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/platform.h>

#include <complex>
#include <type_traits>

/// Device kernels behind TA::Tensor's tile ops. TA::Tensor members call these
/// when every operand lives in unified memory and execution_space() is
/// Device; otherwise they run their host code. The kernels are compiled once,
/// in device/tensor.cpp, for the element types device_executable_v admits.

namespace TiledArray {
namespace detail {

template <typename T>
inline constexpr bool is_device_numeric_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

template <typename T>
inline constexpr bool is_um_tensor_v =
    std::is_same_v<typename T::allocator_type,
                   device_um_allocator<typename T::value_type>>;

}  // namespace detail

/// true if tensors of types \c T, \c Ts... share a device-supported element
/// type and all live in unified memory, i.e. an op on them may run on the
/// device
template <typename T, typename... Ts>
inline constexpr bool device_executable_v =
    detail::is_device_numeric_v<typename T::value_type> &&
    (std::is_same_v<typename T::value_type, typename Ts::value_type> && ...) &&
    detail::is_um_tensor_v<T> && (detail::is_um_tensor_v<Ts> && ...);

namespace device {

using math::blas::integer;

/// row-major C = alpha * op(A) * op(B) + beta * C on the device, with the
/// semantics of math::blas::gemm; \p range picks the stream (see
/// blasqueue_for)
template <typename T>
void gemm(math::blas::Op op_a, math::blas::Op op_b, integer m, integer n,
          integer k, T alpha, const T* a, integer lda, const T* b, integer ldb,
          T beta, T* c, integer ldc, const Range& range);

extern template void gemm(math::blas::Op, math::blas::Op, integer, integer,
                          integer, float, const float*, integer, const float*,
                          integer, float, float*, integer, const Range&);
extern template void gemm(math::blas::Op, math::blas::Op, integer, integer,
                          integer, double, const double*, integer,
                          const double*, integer, double, double*, integer,
                          const Range&);
extern template void gemm(math::blas::Op, math::blas::Op, integer, integer,
                          integer, std::complex<float>,
                          const std::complex<float>*, integer,
                          const std::complex<float>*, integer,
                          std::complex<float>, std::complex<float>*, integer,
                          const Range&);
extern template void gemm(math::blas::Op, math::blas::Op, integer, integer,
                          integer, std::complex<double>,
                          const std::complex<double>*, integer,
                          const std::complex<double>*, integer,
                          std::complex<double>, std::complex<double>*, integer,
                          const Range&);

}  // namespace device
}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_DISPATCH_H
