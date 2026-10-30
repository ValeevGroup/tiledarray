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

#include <algorithm>
#include <complex>
#include <limits>
#include <numeric>
#include <vector>

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
    TiledArray::detail::prefetch(p, n, stream.device, stream.stream);
  }

  /// makes the enclosing task wait for (or, outside a task, blocks until)
  /// the work queued on this stream
  void finish() const { sync_madness_task_with(stream); }
};

/// blaspp's device calls take int lengths
constexpr integer max_blas_length = std::numeric_limits<int>::max();

/// calls \p f(offset, length) for consecutive pieces of [0, n) that are
/// short enough for blaspp
template <typename F>
void for_each_blas_chunk(integer n, F&& f) {
  for (integer i = 0; i < n; i += max_blas_length)
    f(i, std::min(max_blas_length, n - i));
}

/// the number of pieces for_each_blas_chunk splits [0, n) into
integer blas_chunk_count(integer n) {
  return (n + max_blas_length - 1) / max_blas_length;
}

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
  for_each_blas_chunk(n, [&](integer i, integer m) {
    ::blas::copy(m, x + i, 1, y + i, 1, launch.queue);
  });
  launch.finish();
}

template <typename T>
void scal(integer n, T alpha, T* x, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  for_each_blas_chunk(n, [&](integer i, integer m) {
    ::blas::scal(m, alpha, x + i, 1, launch.queue);
  });
  launch.finish();
}

template <typename T>
void axpy(integer n, T alpha, const T* x, T* y, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  launch.prefetch(y, n);
  for_each_blas_chunk(n, [&](integer i, integer m) {
    ::blas::axpy(m, alpha, x + i, 1, y + i, 1, launch.queue);
  });
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
  // librett indexes with 32-bit integers
  if (in_range.volume() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    TA_EXCEPTION(
        "device::permute: librett cannot permute more than INT_MAX elements");
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
  // one partial result per chunk, written on the host by the stream
  std::vector<T> partial(blas_chunk_count(n));
  for_each_blas_chunk(n, [&](integer i, integer m) {
    ::blas::dotu(m, x + i, 1, y + i, 1, &partial[i / max_blas_length],
                 launch.queue);
  });
  launch.queue.sync();
  launch.finish();
  return std::accumulate(partial.begin(), partial.end(), T(0));
}

template <typename T>
T inner_product(integer n, const T* x, const T* y, const Range& range) {
  const Launch launch(range);
  launch.prefetch(x, n);
  launch.prefetch(y, n);
  // one partial result per chunk, written on the host by the stream
  std::vector<T> partial(blas_chunk_count(n));
  for_each_blas_chunk(n, [&](integer i, integer m) {
    ::blas::dot(m, x + i, 1, y + i, 1, &partial[i / max_blas_length],
                launch.queue);
  });
  launch.queue.sync();
  launch.finish();
  return std::accumulate(partial.begin(), partial.end(), T(0));
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
