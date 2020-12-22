/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  parallel_gemm.h
 *  Apr 29, 2015
 *
 */

#ifndef TILEDARRAY_PARALLEL_GEMM_H__INCLUDED
#define TILEDARRAY_PARALLEL_GEMM_H__INCLUDED

#include <TiledArray/blas.h>
#include <TiledArray/external/madness.h>
#include <TiledArray/vector_op.h>

#define TILEDARRAY_DYNAMIC_BLOCK_SIZE std::numeric_limits<std::size_t>::max();

namespace TiledArray {
namespace math {

//#ifdef HAVE_INTEL_TBB

template <typename T, integer Size>
class MatrixBlockTask : public tbb::task {
  const integer rows_;
  const integer cols_;
  T* data_;
  const integer ld_;
  std::shared_ptr<T> result_;

  /// Copy a \c Size^2 block from \c data to \c result

  /// \param[out] result A pointer to the first element of the result block
  /// \param[in] data A pointer to the first element of the block to be copied
  /// \param[in] ld The leading dimension stride for the \c data block
  void copy_block(T* result, const T* data, const integer ld) {
    const T* const block_end = result + (TILEDARRAY_LOOP_UNWIND * Size);
    for (; result < block_end; result += Size, data += ld)
      TiledArray::math::copy_block(result, data);
  }

  /// Copy a rectangular \c m*n block from \c data to \c result

  /// \param[in] m The number of rows to copy
  /// \param[in] n The number of columns to copy
  /// \param[out] result A pointer to the first element of the result block
  /// \param[in] data A pointer to the first element of the block to be copied
  /// \param[in] ld The leading dimension stride for the \c data block
  void copy_block(const integer m, const integer n, T* result, const T* data,
                  const integer ld) {
    const T* const block_end = result + (m * Size);
    for (; result < block_end; result += Size, data += ld)
      TiledArray::math::copy_block(n, result, data);
  }

 public:
  MatrixBlockTask(const integer rows, const integer cols, const T* const data,
                  const integer ld)
      : rows_(rows), cols_(cols), data_(data), ld_(ld) {}

  /// Task body
  virtual tbb::task* execut() {
    // Compute block iteration limit
    constexpr integer index_mask = ~integer(TILEDARRAY_LOOP_UNWIND - 1ul);
    const integer mx =
        rows_ & index_mask;  // = rows - rows % TILEDARRAY_LOOP_UNWIND
    const integer nx =
        cols_ & index_mask;  // = cols - cols % TILEDARRAY_LOOP_UNWIND
    const integer m_tail = rows_ - mx;
    const integer n_tail = cols_ - nx;

    // Copy data into block_
    integer i = 0ul;
    T* result_i = result_.get();
    const T* data_i = data_;
    for (; i < mx;
         i += TILEDARRAY_LOOP_UNWIND, result_i += Size, data_i += ld_) {
      integer j = 0ul;
      for (; j < nx; j += TILEDARRAY_LOOP_UNWIND)
        copy_block(result_i + j, data_i + j);

      if (n_tail)
        copy_block(TILEDARRAY_LOOP_UNWIND, n_tail, result_i + j, data_i + j);
    }

    if (m_tail) {
      integer j = 0ul;
      for (; j < nx; j += TILEDARRAY_LOOP_UNWIND)
        copy_block(m_tail, TILEDARRAY_LOOP_UNWIND, result_i + j, data_i + j);

      if (n_tail) copy_block(m_tail, n_tail, result_i + j, data_i + j);
    }

    return nullptr;
  }

  std::shared_ptr<T> result() {
    constexpr integer size = Size * Size;
    constexpr integer bytes = size * sizeof(T);

    T* result_ptr = nullptr;
    if (!posix_memalign(result_ptr, TILEARRAY_ALIGNMENT, bytes))
      throw std::bad_alloc();

    result_.reset(result_ptr);

    return result_;
  }

};  // class MatrixBlockTask

template <integer Size, typename C, typename A = C, typename B = C,
          typename Alpha = C, typename Beta = C>
class GemmTask : public tbb::task {
  const blas::Op op_a_, op_b_;
  const integer m_, n_, k_;
  const Alpha alpha_;
  std::shared_ptr<A> a_;
  constexpr integer lda_ = Size;
  std::shared_ptr<B> b_;
  const Beta beta_;
  std::shared_ptr<C> c_;
  const integer ldc_;

 public:
  GemmTask(blas::Op op_a, blas::Op op_b, const integer m, const integer n,
           const integer k, const Alpha alpha, const std::shared_ptr<A>& a,
           const std::shared_ptr<B>& b, const Beta beta,
           const std::shared_ptr<C>& c, const integer ldc)
      : op_a_(op_a),
        op_b_(op_b),
        m_(m),
        n_(n),
        k_(k),
        alpha_(alpha),
        a_(a),
        b_(b),
        beta_(beta),
        c_(c),
        ldc_(c) {}

  virtual tbb::task execute() {
    gemm(op_a_, op_b_, m_, n_, k_, alpha_, a_.get(), Size, b_.get(), Size, c_,
         ldc_);
  }

};  // class GemmTask

//#endif // HAVE_INTEL_TBB

}  // namespace math
}  // namespace TiledArray

#endif  // TILEDARRAY_PARALLEL_GEMM_H__INCLUDED
