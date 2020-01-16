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

/// Partial reduce algorithm automatic loop unwinding

/// \tparam N The number of steps to unwind
template <std::size_t N>
class PartialReduceUnwind;

template <>
class PartialReduceUnwind<0> {
 public:
  static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - 1;

  template <typename Left, typename Right, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void row_reduce(
      const Left* MADNESS_RESTRICT const left, const std::size_t,
      const Right* MADNESS_RESTRICT const right,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    // Load the left block
    TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(left_block, left);

    reduce_block(op, result[offset], left_block, right);
  }

  template <typename Arg, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void row_reduce(
      const Arg* MADNESS_RESTRICT const arg, const std::size_t,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    // Load the left block
    TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(arg_block, arg);

    reduce_block(op, result[offset], arg_block);
  }

  template <typename Left, typename Right, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void col_reduce(
      const Left* MADNESS_RESTRICT const left, const std::size_t /*stride*/,
      const Right* MADNESS_RESTRICT const right,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    // Load right block
    const Right right_j = right[offset];

    // Load left block
    TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(left_block, left);

    for_each_block(
        [right_j, &op](Result& result_ij, const Left left_i) {
          op(result_ij, left_i, right_j);
        },
        result, left_block);
  }

  template <typename Arg, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void col_reduce(
      const Arg* MADNESS_RESTRICT const arg, const std::size_t /*stride*/,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    // Load the arg block
    TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(arg_block, arg);

    for_each_block(op, result, arg_block);
  }

};  // class PartialReduceUnwind<0>

template <std::size_t N>
class PartialReduceUnwind : public PartialReduceUnwind<N - 1> {
 public:
  typedef PartialReduceUnwind<N - 1> PartialReduceUnwindN1;

  static const std::size_t offset = TILEDARRAY_LOOP_UNWIND - N - 1;

  template <typename Left, typename Right, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void row_reduce(
      const Left* MADNESS_RESTRICT const left, const std::size_t stride,
      const Right* MADNESS_RESTRICT const right,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    {
      // Load the left block
      TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(left_block, left);

      reduce_block(op, result[offset], left_block, right);
    }

    PartialReduceUnwindN1::row_reduce(left + stride, stride, right, result, op);
  }

  template <typename Arg, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void row_reduce(
      const Arg* MADNESS_RESTRICT const arg, const std::size_t stride,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    {
      // Load the left block
      TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(arg_block, arg);

      reduce_block(op, result[offset], arg_block);
    }

    PartialReduceUnwindN1::row_reduce(arg + stride, stride, result, op);
  }

  template <typename Left, typename Right, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void col_reduce(
      const Left* MADNESS_RESTRICT const left, const std::size_t stride,
      const Right* MADNESS_RESTRICT const right,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    {
      // Load right block
      const Right right_j = right[offset];

      // Load left block
      TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(left_block, left);

      for_each_block(
          [right_j, &op](Result& result_ij, const Left left_i) {
            op(result_ij, left_i, right_j);
          },
          result, left_block);
    }

    PartialReduceUnwindN1::col_reduce(left + stride, stride, right, result, op);
  }

  template <typename Arg, typename Result, typename Op>
  static TILEDARRAY_FORCE_INLINE void col_reduce(
      const Arg* MADNESS_RESTRICT const arg, const std::size_t stride,
      Result* MADNESS_RESTRICT const result, const Op& op) {
    {
      // Load the left block
      TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(arg_block, arg);

      for_each_block(op, result, arg_block);
    }

    PartialReduceUnwindN1::col_reduce(arg + stride, stride, result, op);
  }
};  // class OuterVectorOpUnwind

// Convenience typedef
typedef PartialReduceUnwind<TILEDARRAY_LOOP_UNWIND - 1> PartialReduceUnwindN;

// TODO reduce_op
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
                const Left* MADNESS_RESTRICT const left,
                const Right* MADNESS_RESTRICT const right,
                Result* MADNESS_RESTRICT const result, const Op& op) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  const std::size_t mx =
      m & index_mask::value;  // = m - m % TILEDARRAY_LOOP_UNWIND
  const std::size_t nx =
      n & index_mask::value;  // = n - n % TILEDARRAY_LOOP_UNWIND

  for (; i < mx; i += TILEDARRAY_LOOP_UNWIND) {
    // Load result block
    TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(result_block, result + i);

    // Compute left pointer offset
    const Left* MADNESS_RESTRICT const left_i = left + (i * n);

    std::size_t j = 0ul;
    for (; j < nx; j += TILEDARRAY_LOOP_UNWIND) {
      // Load right block
      TILEDARRAY_ALIGNED_STORAGE Right right_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(right_block, right + j);

      // Compute and store a block
      PartialReduceUnwindN::row_reduce(left_i + j, n, right_block, result_block,
                                       op);
    }

    for (; j < n; ++j) {
      // Load right block
      const Right right_j = right[j];

      // Compute a block
      TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
      gather_block(left_block, left_i + j, n);
      for_each_block(
          [right_j, &op](Result& result_ij, const Left left_i) {
            op(result_ij, left_i, right_j);
          },
          result_block, left_block);
    }

    // Post store result
    copy_block(result + i, result_block);
  }

  for (; i < m; ++i) {
    // Load result block
    Result result_block = result[i];
    reduce_op_serial(op, n, result_block, left + (i * n), right);
    result[i] = result_block;
  }
}

/// Reduce the rows of a matrix

/// <tt>op(result[i], arg[i][j])</tt>.
/// \tparam Arg The left-hand vector element type
/// \tparam Result The a matrix element type
/// \tparam Op The operator type
/// \param[in] m The number of rows in left
/// \param[in] n The size of the right-hand vector
/// \param[in] arg An m*n matrix
/// \param[out] result The result vector of size m
/// \param[in] op The operation that will reduce the rows of left
template <typename Arg, typename Result, typename Op>
void row_reduce(const std::size_t m, const std::size_t n,
                const Arg* MADNESS_RESTRICT const arg,
                Result* MADNESS_RESTRICT const result, const Op& op) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  const std::size_t mx =
      m & index_mask::value;  // = m - m % TILEDARRAY_LOOP_UNWIND
  const std::size_t nx =
      n & index_mask::value;  // = n - n % TILEDARRAY_LOOP_UNWIND

  for (; i < mx; i += TILEDARRAY_LOOP_UNWIND) {
    // Load result block
    TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(result_block, result + i);

    // Compute left pointer offset
    const Arg* MADNESS_RESTRICT const arg_i = arg + (i * n);

    std::size_t j = 0ul;
    for (; j < nx; j += TILEDARRAY_LOOP_UNWIND) {
      // Compute and store a block
      PartialReduceUnwindN::row_reduce(arg_i + j, n, result_block, op);
    }

    for (; j < n; ++j) {
      // Compute a block
      TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
      gather_block(arg_block, arg_i + j, n);
      for_each_block(op, result_block, arg_block);
    }

    // Post process and store result
    copy_block(result + i, result_block);
  }

  for (; i < m; ++i) {
    // Load result block
    Result result_block = result[i];
    reduce_op_serial(op, n, result_block, arg + (i * n));
    result[i] = result_block;
  }
}

/// Reduce the columns of a matrix

/// <tt>op(result[j], left[i][j], right[i])</tt>.
/// \tparam Left The left-hand vector element type
/// \tparam Right The right-hand vector element type
/// \tparam Result The a matrix element type
/// \tparam Op The operator type
/// \param[in] m The number of rows in left
/// \param[in] n The size of the right-hand vector
/// \param[in] left An m*n matrix
/// \param[in] right A vector of size m
/// \param[out] result The result vector of size n
/// \param[in] op The operation that will reduce the columns of left
template <typename Left, typename Right, typename Result, typename Op>
void col_reduce(const std::size_t m, const std::size_t n,
                const Left* MADNESS_RESTRICT const left,
                const Right* MADNESS_RESTRICT const right,
                Result* MADNESS_RESTRICT const result, const Op& op) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  const std::size_t mx =
      m & index_mask::value;  // = m - m % TILEDARRAY_LOOP_UNWIND
  const std::size_t nx =
      n & index_mask::value;  // = n - n % TILEDARRAY_LOOP_UNWIND

  for (; i < mx; i += TILEDARRAY_LOOP_UNWIND) {
    // Load right block
    TILEDARRAY_ALIGNED_STORAGE Right right_block[TILEDARRAY_LOOP_UNWIND];
    copy_block(right_block, right + i);

    // Compute left pointer offset
    const Left* MADNESS_RESTRICT const left_i = left + (i * n);

    std::size_t j = 0ul;
    for (; j < nx; j += TILEDARRAY_LOOP_UNWIND) {
      // Load result block
      TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(result_block, result + j);

      // Compute and store a block
      PartialReduceUnwindN::col_reduce(left_i + j, n, right_block, result_block,
                                       op);

      // Store the result
      copy_block(result + j, result_block);
    }

    for (; j < n; ++j) {
      // Load result block
      Result result_block = result[j];

      // Compute a block
      TILEDARRAY_ALIGNED_STORAGE Left left_block[TILEDARRAY_LOOP_UNWIND];
      gather_block(left_block, left_i + j, n);
      reduce_block(op, result_block, left_block, right_block);

      result[j] = result_block;
    }
  }

  for (; i < m; ++i) {
    const Right right_i = right[i];

    // Reduce row i to result
    inplace_vector_op(
        [&op, right_i](Result& result_j, const Left left_ij) {
          op(result_j, left_ij, right_i);
        },
        n, result, left + (i * n));
  }
}

/// Reduce the columns of a matrix

/// <tt>op(result[j], arg[i][j])</tt>.
/// \tparam Arg The argument vector element type
/// \tparam Result The a matrix element type
/// \tparam Op The operator type
/// \param[in] m The number of rows in left
/// \param[in] n The size of the right-hand vector
/// \param[in] arg An m*n matrix
/// \param[out] result The result vector of size n
/// \param[in] op The operation that will reduce the columns of left
template <typename Arg, typename Result, typename Op>
void col_reduce(const std::size_t m, const std::size_t n,
                const Arg* MADNESS_RESTRICT const arg,
                Result* MADNESS_RESTRICT const result, const Op& op) {
  std::size_t i = 0ul;

  // Compute block iteration limit
  const std::size_t mx =
      m & index_mask::value;  // = m - m % TILEDARRAY_LOOP_UNWIND
  const std::size_t nx =
      n & index_mask::value;  // = n - n % TILEDARRAY_LOOP_UNWIND

  for (; i < mx; i += TILEDARRAY_LOOP_UNWIND) {
    // Compute left pointer offset
    const Arg* MADNESS_RESTRICT const arg_i = arg + (i * n);

    std::size_t j = 0ul;
    for (; j < nx; j += TILEDARRAY_LOOP_UNWIND) {
      // Load result block
      TILEDARRAY_ALIGNED_STORAGE Result result_block[TILEDARRAY_LOOP_UNWIND];
      copy_block(result_block, result + j);

      // Compute and store a block
      PartialReduceUnwindN::col_reduce(arg_i + j, n, result_block, op);

      // Store the result
      copy_block(result + j, result_block);
    }

    for (; j < n; ++j) {
      // Load result block
      Result result_block = result[j];

      // Compute a block
      TILEDARRAY_ALIGNED_STORAGE Arg arg_block[TILEDARRAY_LOOP_UNWIND];
      gather_block(arg_block, arg_i + j, n);
      reduce_block(op, result_block, arg_block);

      result[j] = result_block;
    }
  }

  for (; i < m; ++i) {
    // Reduce row i to result
    inplace_vector_op(op, n, result, arg + (i * n));
  }
}

}  // namespace math
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_PARTIAL_REDUCE_H__INCLUDED
