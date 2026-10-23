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

#include <TiledArray/device/blas.h>
#include <TiledArray/device/kernel/mult_kernel.h>
#include <TiledArray/external/librett.h>

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

/// the BLAS queue, and its stream, that device work on a Range goes to
struct Launch {
  ::blas::Queue& queue;
  Stream stream;

  explicit Launch(const Range& range)
      : queue(blasqueue_for(range)), stream(queue.device(), queue.stream()) {
    DeviceSafeCall(setDevice(stream.device));
  }

  /// prefetches \p n elements at \p p to this stream's device
  template <typename T>
  void prefetch(const T* p, std::size_t n) const {
    if (deviceEnv::instance()->concurrent_managed_access())
      DeviceSafeCall(
          memPrefetchAsync(p, n * sizeof(T), stream.device, stream.stream));
  }

  /// makes the enclosing task wait for (or, outside a task, blocks until)
  /// the work queued on this stream
  void finish() const { sync_madness_task_with(stream); }
};

}  // namespace

template <typename T>
void gemm(math::blas::Op op_a, math::blas::Op op_b, integer m, integer n,
          integer k, T alpha, const T* a, integer lda, const T* b, integer ldb,
          T beta, T* c, integer ldc, const Range& range) {
  const Launch launch(range);
  using math::blas::NoTranspose;
  launch.prefetch(a, (op_a == NoTranspose ? m : k) * lda);
  launch.prefetch(b, (op_b == NoTranspose ? k : n) * ldb);
  launch.prefetch(c, m * ldc);
  // a column-major gemm with the operands swapped computes the row-major one
  ::blas::gemm(::blas::Layout::ColMajor, op_b, op_a, n, m, k, alpha, b, ldb, a,
               lda, beta, c, ldc, launch.queue);
  launch.finish();
}

template <typename T>
void copy(integer n, const T* x, T* y, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  launch.prefetch(y, n);
  ::blas::copy(n, x, 1, y, 1, launch.queue);
  launch.finish();
}

template <typename T>
void scal(integer n, T alpha, T* x, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  ::blas::scal(n, alpha, x, 1, launch.queue);
  launch.finish();
}

template <typename T>
void axpy(integer n, T alpha, const T* x, T* y, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  launch.prefetch(y, n);
  ::blas::axpy(n, alpha, x, 1, y, 1, launch.queue);
  launch.finish();
}

template <typename T>
void mult(integer n, const T* a, const T* b, T* c, const Range& range) {
  const Launch launch(range);
  launch.prefetch(a, n);
  launch.prefetch(b, n);
  launch.prefetch(c, n);
  mult_kernel(c, a, b, n, launch.stream);
  launch.finish();
}

template <typename T>
void mult_to(integer n, const T* a, T* c, const Range& range) {
  const Launch launch(range);
  launch.prefetch(a, n);
  launch.prefetch(c, n);
  mult_to_kernel(c, a, n, launch.stream);
  launch.finish();
}

template <typename T>
void permute(const T* in, T* out, const Range& in_range,
             const Permutation& perm) {
  // ponytail: librett builds and destroys a plan per call; cache plans by
  // (extents, perm) if permutes show up in profiles
  const Launch launch(in_range);
  const auto n = in_range.volume();
  launch.prefetch(in, n);
  launch.prefetch(out, n);
  librett_permute(const_cast<T*>(in), out, in_range, perm,
                  launch.stream.stream);
  launch.finish();
}

template <typename T>
T dot(integer n, const T* x, const T* y, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  launch.prefetch(y, n);
  T result(0);
  ::blas::dotu(n, x, 1, y, 1, &result, launch.queue);
  launch.queue.sync();  // result is written on the host by the stream
  launch.finish();
  return result;
}

template <typename T>
T inner_product(integer n, const T* x, const T* y, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  launch.prefetch(y, n);
  T result(0);
  ::blas::dot(n, x, 1, y, 1, &result, launch.queue);
  launch.queue.sync();  // result is written on the host by the stream
  launch.finish();
  return result;
}

void synchronize() {
  if (auto stream = madness_task_current_stream())
    DeviceSafeCall(streamSynchronize(stream->stream));
}

#define TA_INSTANTIATE_DEVICE_KERNELS(T)                                      \
  template void gemm(math::blas::Op, math::blas::Op, integer, integer,        \
                     integer, T, const T*, integer, const T*, integer, T, T*, \
                     integer, const Range&);                                  \
  template void copy(integer, const T*, T*, const Range&);                    \
  template void scal(integer, T, T*, const Range&);                           \
  template void axpy(integer, T, const T*, T*, const Range&);                 \
  template void mult(integer, const T*, const T*, T*, const Range&);          \
  template void mult_to(integer, const T*, T*, const Range&);                 \
  template void permute(const T*, T*, const Range&, const Permutation&);      \
  template T dot(integer, const T*, const T*, const Range&);                  \
  template T inner_product(integer, const T*, const T*, const Range&);

TA_INSTANTIATE_DEVICE_KERNELS(float)
TA_INSTANTIATE_DEVICE_KERNELS(double)
TA_INSTANTIATE_DEVICE_KERNELS(std::complex<float>)
TA_INSTANTIATE_DEVICE_KERNELS(std::complex<double>)

#undef TA_INSTANTIATE_DEVICE_KERNELS

}  // namespace device

}  // namespace TiledArray
