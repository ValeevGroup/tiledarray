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

#include <TiledArray/device/tensor.h>

#include <complex>

namespace TiledArray {

// Explicit instantiations of the UMTensor class for the standard numeric
// types. Without these, every TU including device/tensor.h would instantiate
// the full TA::Tensor<T, device_um_allocator<T>> class body.
// Mirrors the host-side set in tensor/tensor.cpp; paired with the
// `extern template` declarations in device/tensor.h.
template class Tensor<double, device_um_allocator<double>>;
template class Tensor<float, device_um_allocator<float>>;
template class Tensor<std::complex<double>,
                      device_um_allocator<std::complex<double>>>;
template class Tensor<std::complex<float>,
                      device_um_allocator<std::complex<float>>>;
template class Tensor<int, device_um_allocator<int>>;
template class Tensor<long, device_um_allocator<long>>;

namespace device {

namespace {
/// prefetch \p n elements at \p p to the device of \p stream, on \p stream
template <typename T>
void prefetch(const T* p, std::size_t n, const Stream& stream) {
  if (deviceEnv::instance()->concurrent_managed_access())
    DeviceSafeCall(
        memPrefetchAsync(p, n * sizeof(T), stream.device, stream.stream));
}
}  // namespace

template <typename T>
void gemm(math::blas::Op op_a, math::blas::Op op_b, integer m, integer n,
          integer k, T alpha, const T* a, integer lda, const T* b, integer ldb,
          T beta, T* c, integer ldc, const Range& range) {
  auto& queue = blasqueue_for(range);
  const Stream stream(queue.device(), queue.stream());
  DeviceSafeCall(setDevice(stream.device));

  using math::blas::NoTranspose;
  prefetch(a, (op_a == NoTranspose ? m : k) * lda, stream);
  prefetch(b, (op_b == NoTranspose ? k : n) * ldb, stream);
  prefetch(c, m * ldc, stream);

  // a column-major gemm with the operands swapped computes the row-major one
  ::blas::gemm(::blas::Layout::ColMajor, op_b, op_a, n, m, k, alpha, b, ldb, a,
               lda, beta, c, ldc, queue);

  sync_madness_task_with(stream);
}

template void gemm(math::blas::Op, math::blas::Op, integer, integer, integer,
                   float, const float*, integer, const float*, integer, float,
                   float*, integer, const Range&);
template void gemm(math::blas::Op, math::blas::Op, integer, integer, integer,
                   double, const double*, integer, const double*, integer,
                   double, double*, integer, const Range&);
template void gemm(math::blas::Op, math::blas::Op, integer, integer, integer,
                   std::complex<float>, const std::complex<float>*, integer,
                   const std::complex<float>*, integer, std::complex<float>,
                   std::complex<float>*, integer, const Range&);
template void gemm(math::blas::Op, math::blas::Op, integer, integer, integer,
                   std::complex<double>, const std::complex<double>*, integer,
                   const std::complex<double>*, integer, std::complex<double>,
                   std::complex<double>*, integer, const Range&);

}  // namespace device

}  // namespace TiledArray
