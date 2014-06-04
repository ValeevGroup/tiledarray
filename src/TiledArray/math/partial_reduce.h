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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  partial_reduce.h
 *  Feb 27, 2014
 *
 */

#ifndef TILEDARRAY_MATH_PARTIAL_REDUCE_H__INCLUDED
#define TILEDARRAY_MATH_PARTIAL_REDUCE_H__INCLUDED

#include <TiledArray/math/vector_op.h>
#include <TiledArray/utility.h>

namespace TiledArray {
  namespace math {

    /// Outer algorithm automatic loop unwinding

    /// \tparam N The number of steps to unwind
    template <std::size_t N>
    class PartialReduceUnwind;

    template <>
    class PartialReduceUnwind<0> {
    public:

      static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - 1;

      template <typename Left, typename Right, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      row_reduce(const Left* restrict const left, const std::size_t,
          const Right* restrict const right, Result* restrict const result,
          const Op& op)
      {
        // Load the left block
        Left left_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(left, left_block);

        VecOpUnwindN::reduce(left_block, right, result[offset], op);
      }

      template <typename Arg, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      row_reduce(const Arg* restrict const arg, const std::size_t,
          Result* restrict const result, const Op& op)
      {
        // Load the left block
        Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);

        VecOpUnwindN::reduce(arg_block, result[offset], op);
      }

      template <typename Left, typename Right, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      col_reduce(const Left* restrict const left, const std::size_t stride,
          const Right* restrict const right, Result* restrict const result,
          const Op& op)
      {
        // Load right block
        const Right right_block = right[offset];

        // Load left block
        Left left_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(left, left_block);

        VecOpUnwindN::reduce(left_block, result,
            TiledArray::detail::bind_second(right_block, op));
      }


      template <typename Arg, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      col_reduce(const Arg* restrict const arg, const std::size_t stride,
          Result* restrict const result, const Op& op)
      {
        // Load the arg block
        Arg arg_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(arg, arg_block);

        VecOpUnwindN::reduce(arg_block, result, op);
      }

    }; // class PartialReduceUnwind<0>


    template <std::size_t N>
    class PartialReduceUnwind : public PartialReduceUnwind<N - 1> {
    public:

      typedef PartialReduceUnwind<N - 1> PartialReduceUnwindN1;

      static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - N - 1;

      template <typename Left, typename Right, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      row_reduce(const Left* restrict const left, const std::size_t stride,
          const Right* restrict const right, Result* restrict const result,
          const Op& op)
      {
        {
          // Load the left block
          Left left_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(left, left_block);

          VecOpUnwindN::reduce(left_block, right, result[offset], op);
        }

        PartialReduceUnwindN1::row_reduce(left + stride, stride, right, result, op);
      }


      template <typename Arg, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      row_reduce(const Arg* restrict const arg, const std::size_t stride,
          Result* restrict const result, const Op& op)
      {
        {
          // Load the left block
          Arg arg_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(arg, arg_block);

          VecOpUnwindN::reduce(arg_block, result[offset], op);
        }

        PartialReduceUnwindN1::row_reduce(arg + stride, stride, result, op);
      }


      template <typename Left, typename Right, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      col_reduce(const Left* restrict const left, const std::size_t stride,
          const Right* restrict const right, Result* restrict const result,
          const Op& op)
      {
        {
          // Load right block
          const Right right_block = right[offset];

          // Load left block
          Left left_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(left, left_block);

          VecOpUnwindN::reduce(left_block, result,
              TiledArray::detail::bind_second(right_block, op));
        }

        PartialReduceUnwindN1::col_reduce(left + stride, stride, right, result, op);
      }

      template <typename Arg, typename Result, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      col_reduce(const Arg* restrict const arg, const std::size_t stride,
          Result* restrict const result, const Op& op)
      {
        {
          // Load the left block
          Arg arg_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(arg, arg_block);

          VecOpUnwindN::reduce(arg_block, result, op);
        }

        PartialReduceUnwindN1::col_reduce(arg + stride, stride, result, op);
      }
    }; // class OuterVectorOpUnwind

    // Convenience typedef
    typedef PartialReduceUnwind<TILEDARRAY_LOOP_UNWIND - 1> PartialReduceUnwindN;


    /// Reduce the rows of a matrix

    /// <tt>op(result[i], left[i][j], right[j])</tt>.
    /// \tparam Left The left-hand matrix element type
    /// \tparam Right The right-hand vector element type
    /// \tparam Result The result vector element type
    /// \param[in] m The number of rows in left
    /// \param[in] n The size of the right-hand vector
    /// \param[in] left An m*n matrix
    /// \param[in] right A vector of size n
    /// \param[out] result The result vector of size m
    /// \param[in] op The operation that will reduce the rows of left
    template <typename Left, typename Right, typename Result, typename Op>
    void row_reduce(const std::size_t m, const std::size_t n,
        const Left* restrict const left, const Right* restrict const right,
        Result* restrict const result, const Op& op)
    {
      std::size_t i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND) {

        // Load result block
        Result result_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(result + i, result_block);

        // Compute left pointer offset
        const Left* restrict const left_i = left + (i * n);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load right block
          Right right_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(right + j, right_block);

          // Compute and store a block
          PartialReduceUnwindN::row_reduce(left_i + j, n, right_block, result_block, op);

        }

        for(; j < n; ++j) {

          // Load right block
          const Right right_j = right[j];

          // Compute a block
          Left left_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::gather(left_i + j, left_block, n);
          VecOpUnwindN::reduce(left_block, result_block,
              TiledArray::detail::bind_second(right_j, op));

        }

        // Post store result
        VecOpUnwindN::copy(result_block, result + i);
      }

#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < m; ++i) {

        // Load result block
        Result result_block = result[i];
        reduce_vector_op(n, left + (i * n), right, result_block, op);
        result[i] = result_block;
      }
    }


    /// Reduce the rows of a matrix

    /// <tt>op(result[i], arg[i][j])</tt>.
    /// \tparam Arg The left-hand vector element type
    /// \tparam Right The right-hand vector element type
    /// \tparam Result The a matrix element type
    /// \param[in] m The number of rows in left
    /// \param[in] n The size of the right-hand vector
    /// \param[in] left An m*n matrix
    /// \param[in] right A vector of size n
    /// \param[out] result The result vector of size m
    /// \param[in] op The operation that will reduce the rows of left
    template <typename Arg, typename Result, typename Op>
    void row_reduce(const std::size_t m, const std::size_t n,
        const Arg* restrict const arg,  Result* restrict const result, const Op& op)
    {
      std::size_t i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND) {

        // Load result block
        Result result_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(result + i, result_block);

        // Compute left pointer offset
        const Arg* restrict const arg_i = arg + (i * n);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Compute and store a block
          PartialReduceUnwindN::row_reduce(arg_i + j, n, result_block, op);

        }

        for(; j < n; ++j) {

          // Compute a block
          Arg arg_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::gather(arg_i + j, arg_block, n);
          VecOpUnwindN::reduce(arg_block, result_block, op);

        }

        // Post process and store result
        VecOpUnwindN::copy(result_block, result + i);
      }

#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < m; ++i) {

        // Load result block
        Result result_block = result[i];
        reduce_vector_op(n, arg + (i * n), result_block, op);
        result[i] = result_block;
      }
    }

    /// Reduce the columns of a matrix

    /// <tt>op(result[j], left[i][j], right[i])</tt>.
    /// \tparam Left The left-hand vector element type
    /// \tparam Right The right-hand vector element type
    /// \tparam Result The a matrix element type
    /// \param[in] m The number of rows in left
    /// \param[in] n The size of the right-hand vector
    /// \param[in] left An m*n matrix
    /// \param[in] right A vector of size m
    /// \param[out] result The result vector of size n
    /// \param[in] op The operation that will reduce the columns of left
    template <typename Left, typename Right, typename Result, typename Op>
    void col_reduce(const std::size_t m, const std::size_t n,
        const Left* restrict const left, const Right* restrict const right,
        Result* restrict const result, const Op& op)
    {
      std::size_t i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND) {

        // Load right block
        Right right_block[TILEDARRAY_LOOP_UNWIND];
        VecOpUnwindN::copy(right + i, right_block);

        // Compute left pointer offset
        const Left* restrict const left_i = left + (i * n);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load result block
          Result result_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(result + j, result_block);

          // Compute and store a block
          PartialReduceUnwindN::col_reduce(left_i + j, n, right_block, result_block, op);

          // Store the result
          VecOpUnwindN::copy(result_block, result + j);
        }

        for(; j < n; ++j) {

          // Load result block
          Result result_block = result[j];

          // Compute a block
          Left left_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::gather(left_i + j, left_block, n);
          VecOpUnwindN::reduce(left_block, right_block, result_block, op);

          result[j] = result_block;

        }

      }

#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < m; ++i) {

        // Reduce row i to result
        reduce_vector_op(n, left + (i * n), result, TiledArray::detail::bind_second(right[i], op));
      }
    }

    /// Reduce the columns of a matrix

    /// <tt>op(result[j], arg[i][j])</tt>.
    /// \tparam Left The left-hand vector element type
    /// \tparam Right The right-hand vector element type
    /// \tparam Result The a matrix element type
    /// \param[in] m The number of rows in left
    /// \param[in] n The size of the right-hand vector
    /// \param[in] left An m*n matrix
    /// \param[in] right A vector of size m
    /// \param[out] result The result vector of size n
    /// \param[in] op The operation that will reduce the columns of left
    template <typename Arg, typename Result, typename Op>
    void col_reduce(const std::size_t m, const std::size_t n,
        const Arg* restrict const arg, Result* restrict const result, const Op& op)
    {
      std::size_t i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1

      // Compute block iteration limit
      const std::size_t mx = m & index_mask::value; // = m - m % TILEDARRAY_LOOP_UNWIND
      const std::size_t nx = n & index_mask::value; // = n - n % TILEDARRAY_LOOP_UNWIND

      for(; i < mx; i += TILEDARRAY_LOOP_UNWIND) {

        // Compute left pointer offset
        const Arg* restrict const arg_i = arg + (i * n);

        std::size_t j = 0ul;
        for(; j < nx; j += TILEDARRAY_LOOP_UNWIND) {

          // Load result block
          Result result_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::copy(result + j, result_block);

          // Compute and store a block
          PartialReduceUnwindN::col_reduce(arg_i + j, n, result_block, op);

          // Store the result
          VecOpUnwindN::copy(result_block, result + j);
        }

        for(; j < n; ++j) {

          // Load result block
          Result result_block = result[j];

          // Compute a block
          Arg arg_block[TILEDARRAY_LOOP_UNWIND];
          VecOpUnwindN::gather(arg_i + j, arg_block, n);
          VecOpUnwindN::reduce(arg_block, result_block, op);

          result[j] = result_block;

        }

      }

#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < m; ++i) {

        // Reduce row i to result
        reduce_vector_op(n, arg + (i * n), result, op);
      }
    }

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_PARTIAL_REDUCE_H__INCLUDED
