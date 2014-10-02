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
      gather_trans(const Arg* restrict const arg, const std::size_t /*arg_stride*/,
          Result* restrict const result)
      {
        // Load arg block
        TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);

        // Transpose arg_block
        VecOpUnwindN::scatter(arg_block, result, TILEDARRAY_LOOP_UNWIND);
      }

      template <typename Arg, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      unary_gather_trans(const Arg* restrict const arg, const std::size_t /*arg_stride*/,
          Result* restrict const result, const Op& op)
      {
        // Load arg block
        TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);

        // Compute result block
        TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::unary(arg_block, result_block, op);

        // Transpose result block
        VecOpUnwindN::scatter(result_block, result, TILEDARRAY_LOOP_UNWIND);
      }


      template <typename Left, typename Right, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      binary_gather_trans(const Left* restrict const left,
          const Right* restrict const right, const std::size_t /*arg_stride*/,
          Result* restrict const result, const Op& op)
      {
        // Load left block
        TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(left, left_block);

        // Load right block
        TILEDARRAY_ALIGNED_STORAGE Right right_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(right, right_block);

        // Compute result block
        TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::binary(left_block, right_block, result_block, op);

        // Transpose result block
        VecOpUnwindN::scatter(result_block, result, TILEDARRAY_LOOP_UNWIND);
      }

      template <typename Arg, typename Result>
      static TILEDARRAY_FORCE_INLINE void
      block_scatter(const Arg* restrict const arg, Result* restrict const result,
          const std::size_t /*result_stride*/)
      {
        VecOpUnwindN::uninitialized_copy(arg, result);
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
        // Load arg block
        TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);

        // Transpose arg_block
        VecOpUnwindN::scatter(arg_block, result, TILEDARRAY_LOOP_UNWIND);

        TransposeUnwindN1::gather_trans(arg + arg_stride, arg_stride, result + 1);
      }

      template <typename Arg, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      unary_gather_trans(const Arg* restrict const arg, const std::size_t arg_stride,
          Result* restrict const result, const Op& op)
      {
        // Load arg block
        TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);

        // Compute result block
        TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::unary(arg_block, result_block, op);

        // Transpose result block
        VecOpUnwindN::scatter(result_block, result, TILEDARRAY_LOOP_UNWIND);

        TransposeUnwindN1::unary_gather_trans(arg + arg_stride, arg_stride, result + 1, op);
      }


      template <typename Left, typename Right, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      binary_gather_trans(const Left* restrict const left,
          const Right* restrict const right, const std::size_t arg_stride,
          Result* restrict const result, const Op& op)
      {
        // Load left block
        TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(left, left_block);

        // Load right block
        TILEDARRAY_ALIGNED_STORAGE Right right_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(right, right_block);

        // Compute result block
        TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::binary(left_block, right_block, result_block, op);

        // Transpose result block
        VecOpUnwindN::scatter(result_block, result, TILEDARRAY_LOOP_UNWIND);

        TransposeUnwindN1::binary_gather_trans(left + arg_stride, right + arg_stride,
            arg_stride, result + 1, op);
      }

      template <typename Arg, typename Result>
      static TILEDARRAY_FORCE_INLINE void
      block_scatter(const Arg* restrict const arg, Result* restrict const result,
          const std::size_t result_stride)
      {
        VecOpUnwindN::uninitialized_copy(arg, result);

        TransposeUnwindN1::block_scatter(arg + TILEDARRAY_LOOP_UNWIND,
            result + result_stride, result_stride);
      }

    }; // class TransposeUnwind

    // Convenience typedef
    typedef TransposeUnwind<TILEDARRAY_LOOP_UNWIND - 1> TransposeUnwindN;
    typedef std::integral_constant<std::size_t, TILEDARRAY_LOOP_UNWIND * TILEDARRAY_LOOP_UNWIND> BlockLoopUnwind;

    template <typename Arg, typename Result>
    TILEDARRAY_FORCE_INLINE void
    transpose_block(const Arg* const arg, const std::size_t arg_stride,
        Result* const result, const std::size_t result_stride)
    {
      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Transpose block
      TransposeUnwindN::gather_trans(arg, arg_stride, temp);
      TransposeUnwindN::block_scatter(temp, result, result_stride);
    }

    template <typename Arg, typename Result, typename Op>
    TILEDARRAY_FORCE_INLINE void
    unary_transpose_block(const Arg* const arg, const std::size_t arg_stride,
        Result* const result, const std::size_t result_stride, const Op& op)
    {
      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Transpose block
      TransposeUnwindN::unary_gather_trans(arg, arg_stride, temp, op);
      TransposeUnwindN::block_scatter(temp, result, result_stride);
    }

    template <typename Left, typename Right, typename Result, typename Op>
    TILEDARRAY_FORCE_INLINE void
    binary_transpose_block(const Left* const left, const Right* const right,
        const std::size_t arg_stride, Result* const result,
        const std::size_t result_stride, const Op& op)
    {
      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Transpose block
      TransposeUnwindN::binary_gather_trans(left, right, arg_stride, temp, op);
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

    template <typename Arg, typename Result, typename Op>
    TILEDARRAY_FORCE_INLINE void
    unary_transpose_block(const std::size_t m, const std::size_t n,
        const Arg* restrict const arg, const std::size_t arg_stride,
        Result* restrict const result, const std::size_t result_stride, const Op& op)
    {
      TA_ASSERT(m <= TILEDARRAY_LOOP_UNWIND);
      TA_ASSERT(n <= TILEDARRAY_LOOP_UNWIND);

      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Copy and transpose arg data into temp block
      for(std::size_t i = 0ul; i < m; ++i) {
        const Arg* restrict const arg_i = arg + (i * arg_stride);
        for(std::size_t j = 0ul, x = i; j < n; ++j, x += TILEDARRAY_LOOP_UNWIND)
          temp[x] = op(arg_i[j]);
      }

      // Copy the temp block into result
      for(std::size_t j = 0ul; j < n; ++j) {
        Result* restrict const result_j = result + (j * result_stride);
        const Result* restrict const temp_j = temp + (j * TILEDARRAY_LOOP_UNWIND);
        for(std::size_t i = 0ul; i < m; ++i)
          result_j[i] = temp_j[i];
      }
    }


    template <typename Left, typename Right, typename Result, typename Op>
    TILEDARRAY_FORCE_INLINE void
    binary_transpose_block(const std::size_t m, const std::size_t n,
        const Left* restrict const left, const Right* restrict const right,
        const std::size_t arg_stride, Result* restrict const result,
        const std::size_t result_stride, const Op& op)
    {
      TA_ASSERT(m <= TILEDARRAY_LOOP_UNWIND);
      TA_ASSERT(n <= TILEDARRAY_LOOP_UNWIND);

      TILEDARRAY_ALIGNED_STORAGE Result temp[BlockLoopUnwind::value];

      // Copy and transpose arg data into temp block
      for(std::size_t i = 0ul; i < m; ++i) {
        const std::size_t offset = i * arg_stride;
        const Left* restrict const left_i = left + offset;
        const Right* restrict const right_i = right + offset;
        for(std::size_t j = 0ul, x = i; j < n; ++j, x += TILEDARRAY_LOOP_UNWIND)
          temp[x] = op(left_i[j], right_i[j]);
      }

      // Copy the temp block into result
      for(std::size_t j = 0ul; j < n; ++j) {
        Result* restrict const result_j = result + (j * result_stride);
        const Result* restrict const temp_j = temp + (j * TILEDARRAY_LOOP_UNWIND);
        for(std::size_t i = 0ul; i < m; ++i)
          result_j[i] = temp_j[i];
      }
    }

    /// Matrix copy transpose

    /// This function will copy the transposed of a matrix.
    /// \tparam Arg The argument element type
    /// \tparam Result The result element type
    /// \param[in] m The number of rows in the argument matrix
    /// \param[in] n The number of columns in the argument matrix
    /// \param[in] arg A pointer to the first element of the argument matrix
    /// \param[in] arg_stride The stride between argument rows
    /// \param[out] result A pointer to the first element of the result matrix
    /// \param[in] result_stride THe stride between result rows
    /// \note The data layout is expected to be row-major.
    template <typename Arg, typename Result>
    void uninitialized_copy_transpose(const std::size_t m, const std::size_t n,
        const Arg* arg, const std::size_t arg_stride,
        Result* result, const std::size_t result_stride)
    {
      // Compute block iteration control variables
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t m_tail = m - mx;
      const std::size_t n_tail = n - nx;
      const std::size_t result_block_step = result_stride * TILEDARRAY_LOOP_UNWIND;
      const std::size_t arg_block_step = arg_stride * TILEDARRAY_LOOP_UNWIND;
      const Arg* const arg_end = arg + (mx * arg_stride);
      const Result* result_end = result + (nx * result_stride);

      // Iterate over block rows
      for(; arg < arg_end; arg += arg_block_step, result += TILEDARRAY_LOOP_UNWIND) {
        const Arg* arg_ij = arg;
        Result* result_ij = result;
        for(; result_ij < result_end; result_ij += result_block_step,
            arg_ij += TILEDARRAY_LOOP_UNWIND)
          transpose_block(arg_ij, arg_stride, result_ij, result_stride);

        if(n_tail)
          transpose_block(TILEDARRAY_LOOP_UNWIND, n_tail,
              arg_ij, arg_stride, result_ij, result_stride);
      }

      if(m_tail) {
        const Arg* arg_ij = arg;
        Result* result_ij = result;
        for(; result_ij < result_end; result_ij += result_block_step,
            arg_ij += TILEDARRAY_LOOP_UNWIND)
          transpose_block(m_tail, TILEDARRAY_LOOP_UNWIND,
              arg_ij, arg_stride, result_ij, result_stride);

        if(n_tail)
          transpose_block(m_tail, n_tail,
              arg_ij, arg_stride, result_ij, result_stride);
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
    /// \note The data layout is expected to be row-major.
    template <typename Arg, typename Result, typename Op>
    void uninitialized_unary_transpose(const std::size_t m, const std::size_t n,
        const Arg* arg, const std::size_t arg_stride,
        Result* result, const std::size_t result_stride, const Op& op)
    {
      // Compute block iteration control variables
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t m_tail = m - mx;
      const std::size_t n_tail = n - nx;
      const std::size_t result_block_step = result_stride * TILEDARRAY_LOOP_UNWIND;
      const std::size_t arg_block_step = arg_stride * TILEDARRAY_LOOP_UNWIND;
      const Arg* const arg_end = arg + (mx * arg_stride);
      const Result* result_end = result + (nx * result_stride);

      // Iterate over block rows
      for(; arg < arg_end; arg += arg_block_step, result += TILEDARRAY_LOOP_UNWIND) {
        const Arg* arg_ij = arg;
        Result* result_ij = result;
        for(; result_ij < result_end; result_ij += result_block_step,
            arg_ij += TILEDARRAY_LOOP_UNWIND)
          unary_transpose_block(arg_ij, arg_stride, result_ij, result_stride, op);

        if(n_tail)
          unary_transpose_block(TILEDARRAY_LOOP_UNWIND, n_tail,
              arg_ij, arg_stride, result_ij, result_stride, op);
      }

      if(m_tail) {
        const Arg* arg_ij = arg;
        Result* result_ij = result;
        for(; result_ij < result_end; result_ij += result_block_step,
            arg_ij += TILEDARRAY_LOOP_UNWIND)
          unary_transpose_block(m_tail, TILEDARRAY_LOOP_UNWIND,
              arg_ij, arg_stride, result_ij, result_stride, op);

        if(n_tail)
          unary_transpose_block(m_tail, n_tail,
              arg_ij, arg_stride, result_ij, result_stride, op);
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
    /// \note The data layout is expected to be row-major.
    template <typename Left, typename Right, typename Result, typename Op>
    void uninitialized_binary_transpose(const std::size_t m, const std::size_t n,
        const Left* left, const Right* right, const std::size_t arg_stride,
        Result* result, const std::size_t result_stride, const Op& op)
    {
      // Compute block iteration control variables
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND
      const std::size_t m_tail = m - mx;
      const std::size_t n_tail = n - nx;
      const std::size_t result_block_step = result_stride * TILEDARRAY_LOOP_UNWIND;
      const std::size_t arg_block_step = arg_stride * TILEDARRAY_LOOP_UNWIND;
      const Left* const left_end = left + (mx * arg_stride);
      const Result* result_end = result + (nx * result_stride);

      // Iterate over block rows
      for(; left < left_end; left += arg_block_step, right += arg_block_step,
          result += TILEDARRAY_LOOP_UNWIND) {
        const Left* left_ij = left;
        const Right* right_ij = right;
        Result* result_ij = result;
        for(; result_ij < result_end; result_ij += result_block_step,
            left_ij += TILEDARRAY_LOOP_UNWIND, right_ij += TILEDARRAY_LOOP_UNWIND)
          binary_transpose_block(left_ij, right_ij, arg_stride, result_ij, result_stride, op);

        if(n_tail)
          binary_transpose_block(TILEDARRAY_LOOP_UNWIND, n_tail,
              left_ij, right_ij, arg_stride, result_ij, result_stride, op);
      }

      if(m_tail) {
        const Left* left_ij = left;
        const Right* right_ij = right;
        Result* result_ij = result;
        for(; result_ij < result_end; result_ij += result_block_step,
            left_ij += TILEDARRAY_LOOP_UNWIND, right_ij += TILEDARRAY_LOOP_UNWIND)
          binary_transpose_block(m_tail, TILEDARRAY_LOOP_UNWIND,
              left_ij, right_ij, arg_stride, result_ij, result_stride, op);

        if(n_tail)
          binary_transpose_block(m_tail, n_tail,
              left_ij, right_ij, arg_stride, result_ij, result_stride, op);
      }
    }

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_TRANSPOSE_H__INCLUDED
