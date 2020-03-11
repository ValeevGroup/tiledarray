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

#include <TiledArray/error.h>
#include <TiledArray/math/vector_op.h>

namespace TiledArray {
namespace math {

/// Partial transpose algorithm automatic loop unwinding

/// \tparam N The number of steps to unwind
template <std::size_t N>
class TransposeUnwind;

template <>
class TransposeUnwind<0> {
 public:
  static constexpr std::size_t offset = TILEDARRAY_LOOP_UNWIND - 1;

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void gather_trans(
      Op&& op, Result* MADNESS_RESTRICT const result,
      const std::size_t arg_stride,
      const Args* MADNESS_RESTRICT const... args) {
    // Load arg block
    Block<Result> result_block;
    for_each_block(std::forward<Op>(op), result_block, Block<Args>(args)...);

    // Transpose arg_block
    result_block.scatter_to(result, TILEDARRAY_LOOP_UNWIND);
  }

  template <typename Op, typename Result>
  static TILEDARRAY_FORCE_INLINE void block_scatter(
      Op&& op, Result* const result, const Result* const arg,
      const std::size_t /*result_stride*/) {
    for_each_block_ptr(std::forward<Op>(op), result, arg);
  }

};  // class TransposeUnwind<0>

template <std::size_t N>
class TransposeUnwind : public TransposeUnwind<N - 1> {
 public:
  typedef TransposeUnwind<N - 1> TransposeUnwindN1;

  static constexpr std::size_t offset = TILEDARRAY_LOOP_UNWIND - N - 1;

  template <typename Op, typename Result, typename... Args>
  static TILEDARRAY_FORCE_INLINE void gather_trans(
      Op&& op, Result* MADNESS_RESTRICT const result,
      const std::size_t arg_stride,
      const Args* MADNESS_RESTRICT const... args) {
    {
      // Load arg block
      Block<Result> result_block;
      for_each_block(op, result_block, Block<Args>(args)...);

      // Transpose arg_block
      result_block.scatter_to(result, TILEDARRAY_LOOP_UNWIND);
    }

    TransposeUnwindN1::gather_trans(std::forward<Op>(op), result + 1,
                                    arg_stride, (args + arg_stride)...);
  }

  template <typename Op, typename Result>
  static TILEDARRAY_FORCE_INLINE void block_scatter(
      Op&& op, Result* const result, const Result* const arg,
      const std::size_t result_stride) {
    for_each_block_ptr(op, result, arg);
    TransposeUnwindN1::block_scatter(
        std::forward<Op>(op), result + result_stride,
        arg + TILEDARRAY_LOOP_UNWIND, result_stride);
  }

};  // class TransposeUnwind

// Convenience typedef
typedef TransposeUnwind<TILEDARRAY_LOOP_UNWIND - 1> TransposeUnwindN;

template <typename InputOp, typename OutputOp, typename Result,
          typename... Args>
TILEDARRAY_FORCE_INLINE void transpose_block(InputOp&& input_op,
                                             OutputOp&& output_op,
                                             const std::size_t result_stride,
                                             Result* const result,
                                             const std::size_t arg_stride,
                                             const Args* const... args) {
  constexpr std::size_t block_size =
      TILEDARRAY_LOOP_UNWIND * TILEDARRAY_LOOP_UNWIND;
  TILEDARRAY_ALIGNED_STORAGE Result temp[block_size];

  // Transpose block
  TransposeUnwindN::gather_trans(std::forward<InputOp>(input_op), temp,
                                 arg_stride, args...);

  TransposeUnwindN::block_scatter(std::forward<OutputOp>(output_op), result,
                                  temp, result_stride);
}

template <typename InputOp, typename OutputOp, typename Result,
          typename... Args>
TILEDARRAY_FORCE_INLINE void transpose_block(
    InputOp&& input_op, OutputOp&& output_op, const std::size_t m,
    const std::size_t n, const std::size_t result_stride,
    Result* MADNESS_RESTRICT const result, const std::size_t arg_stride,
    const Args* MADNESS_RESTRICT const... args) {
  TA_ASSERT(m <= TILEDARRAY_LOOP_UNWIND);
  TA_ASSERT(n <= TILEDARRAY_LOOP_UNWIND);

  constexpr std::size_t block_size =
      TILEDARRAY_LOOP_UNWIND * TILEDARRAY_LOOP_UNWIND;
  TILEDARRAY_ALIGNED_STORAGE Result temp[block_size];

  // Copy and transpose arg data into temp block
  for (std::size_t i = 0ul; i < m; ++i) {
    std::size_t offset = i * arg_stride;
    for (std::size_t j = 0ul, x = i; j < n;
         ++j, x += TILEDARRAY_LOOP_UNWIND, ++offset)
      input_op(temp[x], args[offset]...);
  }

  // Copy the temp block into result
  for (std::size_t j = 0ul; j < n; ++j) {
    Result* MADNESS_RESTRICT const result_j = result + (j * result_stride);
    const Result* MADNESS_RESTRICT const temp_j =
        temp + (j * TILEDARRAY_LOOP_UNWIND);
    for (std::size_t i = 0ul; i < m; ++i) output_op(result_j + i, temp_j[i]);
  }
}

/// Matrix transpose and initialization

/// This function will transpose and transform argument matrices into an
/// uninitialized block of memory
/// \tparam InputOp The input transform operation type
/// \tparam OutputOp The output transform operation type
/// \tparam Result The result element type
/// \tparam Args The argument element type
/// \param[in] input_op The transformation operation applied to input arguments
/// \param[in] output_op The transformation operation used to set the result
/// \param[in] m The number of rows in the argument matrix
/// \param[in] n The number of columns in the argument matrix
/// \param[in] result_stride THe stride between result rows
/// \param[out] result A pointer to the first element of the result matrix
/// \param[in] arg_stride The stride between argument rows
/// \param[in] args A pointer to the first element of the argument matrix
/// \note The data layout is expected to be row-major.
template <typename InputOp, typename OutputOp, typename Result,
          typename... Args>
void transpose(InputOp&& input_op, OutputOp&& output_op, const std::size_t m,
               const std::size_t n, const std::size_t result_stride,
               Result* result, const std::size_t arg_stride,
               const Args* const... args) {
  // Compute block iteration control variables
  constexpr std::size_t index_mask = ~std::size_t(TILEDARRAY_LOOP_UNWIND - 1ul);
  const std::size_t mx = m & index_mask;  // = m - m % TILEDARRAY_LOOP_UNWIND
  const std::size_t nx = n & index_mask;  // = n - n % TILEDARRAY_LOOP_UNWIND
  const std::size_t m_tail = m - mx;
  const std::size_t n_tail = n - nx;
  const std::size_t result_block_step = result_stride * TILEDARRAY_LOOP_UNWIND;
  const std::size_t arg_block_step = arg_stride * TILEDARRAY_LOOP_UNWIND;
  const std::size_t arg_end = mx * arg_stride;
  const Result* result_end = result + (nx * result_stride);

  const auto wrapper_input_op = [&](Result& res, param_type<Args>... a) {
    res = input_op(a...);
  };

  // Iterate over block rows
  std::size_t arg_start = 0;
  for (; arg_start < arg_end;
       arg_start += arg_block_step, result += TILEDARRAY_LOOP_UNWIND) {
    std::size_t arg_offset = arg_start;
    Result* result_ij = result;
    for (; result_ij < result_end;
         result_ij += result_block_step, arg_offset += TILEDARRAY_LOOP_UNWIND)
      transpose_block(wrapper_input_op, output_op, result_stride, result_ij,
                      arg_stride, (args + arg_offset)...);

    if (n_tail)
      transpose_block(wrapper_input_op, output_op, TILEDARRAY_LOOP_UNWIND,
                      n_tail, result_stride, result_ij, arg_stride,
                      (args + arg_offset)...);
  }

  if (m_tail) {
    std::size_t arg_offset = arg_start;
    Result* result_ij = result;
    for (; result_ij < result_end;
         result_ij += result_block_step, arg_offset += TILEDARRAY_LOOP_UNWIND)
      transpose_block(wrapper_input_op, output_op, m_tail,
                      TILEDARRAY_LOOP_UNWIND, result_stride, result_ij,
                      arg_stride, (args + arg_offset)...);

    if (n_tail)
      transpose_block(wrapper_input_op, output_op, m_tail, n_tail,
                      result_stride, result_ij, arg_stride,
                      (args + arg_offset)...);
  }
}

}  // namespace math
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_TRANSPOSE_H__INCLUDED
