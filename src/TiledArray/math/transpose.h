/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  transpose.h
 *  Jun 9, 2014
 *
 */

#ifndef TILEDARRAY_MATH_TRANSPOSE_H__INCLUDED
#define TILEDARRAY_MATH_TRANSPOSE_H__INCLUDED

#include <TiledArray/math/vector_op.h>

namespace TiledArray {
  namespace math {

    /// Partial transpose algorithm automatic loop unwinding

    /// \tparam N The number of steps to unwind
    template <std::size_t N> class TransposeUnwind;

    template <>
    class TransposeUnwind<0> {
    public:

      static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - 1;

      template <typename Arg, typename Result>
      static TILEDARRAY_FORCE_INLINE void
      gather_trans(const Arg* restrict const arg, const std::size_t arg_stride,
          Result* restrict const result)
      {
        TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);
        VecOpUnwindN::scatter(arg_block, result, TILEDARRAY_LOOP_UNWIND);
      }

      template <typename Arg, typename Result>
      static TILEDARRAY_FORCE_INLINE void
      block_scatter(const Arg* restrict const arg, Result* restrict const result,
          const std::size_t result_stride)
      {
        VecOpUnwindN::copy(arg, result);
      }

    }; // class TransposeUnwind<0>

    template <std::size_t N>
    class TransposeUnwind : public TransposeUnwind<N - 1> {
    public:

      typedef TransposeUnwind<N - 1> TransposeUnwindN1;

      static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - N - 1;

      template <typename Arg, typename Result>
      static TILEDARRAY_FORCE_INLINE void
      gather_trans(const Arg* restrict const arg, const std::size_t arg_stride,
          Result* restrict const result)
      {
        TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);
        VecOpUnwindN::scatter(arg_block, result, TILEDARRAY_LOOP_UNWIND);

        TransposeUnwindN1::gather_trans(arg + arg_stride, arg_stride, result + 1);
      }

      template <typename Arg, typename Result>
      static TILEDARRAY_FORCE_INLINE void
      block_scatter(const Arg* restrict const arg, Result* restrict const result,
          const std::size_t result_stride)
      {
        VecOpUnwindN::copy(arg, result);

        TransposeUnwindN1::block_scatter(arg + TILEDARRAY_LOOP_UNWIND,
            result + result_stride, result_stride);
      }

    }; // class TransposeUnwind

    // Convenience typedef
    typedef TransposeUnwind<TILEDARRAY_LOOP_UNWIND - 1> TransposeUnwindN;
    typedef std::integral_constant<std::size_t, TILEDARRAY_LOOP_UNWIND * TILEDARRAY_LOOP_UNWIND> BlockLoopUnwind;

    template <typename Arg, typename Result>
    TILEDARRAY_FORCE_INLINE void
    transpose_block(const Arg* restrict const arg, const std::size_t arg_stride,
        Result* restrict const result, const std::size_t result_stride)
    {
      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Transpose block
      TransposeUnwindN::gather_trans(arg, arg_stride, temp);
      TransposeUnwindN::block_scatter(temp, result, result_stride);
    }

    template <typename Arg, typename Result>
    TILEDARRAY_FORCE_INLINE void
    transpose_block(const std::size_t m, const std::size_t n,
        const Arg* restrict const arg, const std::size_t arg_stride,
        Result* restrict const result, const std::size_t result_stride)
    {
      TA_ASSERT(m <= TILEDARRAY_LOOP_UNWIND);
      TA_ASSERT(n <= TILEDARRAY_LOOP_UNWIND);

      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Copy and transpose arg data into temp block
      for(std::size_t i = 0ul; i < m; ++i) {
        const Arg* restrict const arg_i = arg + (i * arg_stride);
        for(std::size_t j = 0ul, x = i; j < n; ++j, x += TILEDARRAY_LOOP_UNWIND)
          temp[x] = arg_i[j];
      }

      // Copy the temp block into result
      for(std::size_t j = 0ul; j < n; ++j) {
        Result* restrict const result_j = result + (j * result_stride);
        const Result* restrict const temp_j = temp + (j * TILEDARRAY_LOOP_UNWIND);
        for(std::size_t i = 0ul; i < m; ++i)
          result_j[i] = temp_j[i];
      }
    }

    /// Matrix transpose

    /// This function will store a transposed copy of arg in result.
    /// \tparam Arg The argument element type
    /// \tparam Result The result element type
    /// \param[in] m The number of rows in the argument matrix
    /// \param[in] n The number of columns in the argument matrix
    /// \param[in] arg A pointer to the first element of the argument matrix
    /// \param[in] arg_stride The stride between argument rows
    /// \param[out] result A pointer to the first element of the result matrix
    /// \param[in] result_stride THe stride between result rows
    template <typename Arg, typename Result>
    void transpose(const std::size_t m, const std::size_t n,
        const Arg* const arg, const std::size_t arg_stride,
        Result* const result, const std::size_t result_stride)
    {
      // Compute block iteration control variables
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t m_tail = m - mx;
      const std::size_t n_tail = n - nx;
      const std::size_t result_block_step = result_stride * TILEDARRAY_LOOP_UNWIND;

      // Iterate over block rows
      std::size_t i = 0ul;
      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND) {
        const Arg* const arg_i = arg + (i * arg_stride);
        Result* const result_i = result + i;
        std::size_t j = 0ul, j_stride = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND, j_stride += result_block_step)
          transpose_block(arg_i + j, arg_stride, result_i + j_stride, result_stride);

        if(n_tail)
          transpose_block(TILEDARRAY_LOOP_UNWIND, n_tail,
              arg_i + j, arg_stride, result_i + j_stride, result_stride);
      }

      if(m_tail) {
        const Arg* const arg_i = arg + (i * arg_stride);
        Result* const result_i = result + i;
        std::size_t j = 0ul, j_stride = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND, j_stride += result_block_step)
          transpose_block(m_tail, TILEDARRAY_LOOP_UNWIND,
              arg_i + j, arg_stride, result_i + j_stride, result_stride);

        if(n_tail)
          transpose_block(m_tail, n_tail,
              arg_i + j, arg_stride, result_i + j_stride, result_stride);
      }
    }

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_TRANSPOSE_H__INCLUDED
