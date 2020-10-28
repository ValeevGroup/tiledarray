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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  permute.h
 *  May 31, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_PERMUTE_H__INCLUDED
#define TILEDARRAY_TENSOR_PERMUTE_H__INCLUDED

#include <TiledArray/math/transpose.h>
#include <TiledArray/perm_index.h>
#include <TiledArray/tensor/type_traits.h>

namespace TiledArray {
namespace detail {

/// Compute the fused dimensions for permutation

/// This function will compute the fused dimensions of a tensor for use in
/// permutation algorithms. The idea is to partition the stride 1 dimensions
/// in both the input and output tensor, which yields a forth-order tensor
/// (second- and third-order tensors have size of 1 and stride of 0 in the
/// unused dimensions).
/// \tparam SizeType An unsigned integral type
/// \param[out] fused_size An array for the fused size output
/// \param[out] fused_weight An array for the fused weight output
/// \param[in] size An array that holds the unfused size information of the
/// argument tensor
/// \param[in] perm The permutation that will be applied to the argument
/// tensor(s).
template <typename SizeType, typename ExtentType>
inline void fuse_dimensions(SizeType* MADNESS_RESTRICT const fused_size,
                            SizeType* MADNESS_RESTRICT const fused_weight,
                            const ExtentType* MADNESS_RESTRICT const size,
                            const Permutation& perm) {
  const unsigned int ndim1 = perm.size() - 1u;

  int i = ndim1;
  fused_size[3] = size[i--];
  while ((i >= 0) && (perm[i + 1u] == (perm[i] + 1u)))
    fused_size[3] *= size[i--];
  fused_weight[3] = 1u;

  if ((i >= 0) && (perm[i] != ndim1)) {
    fused_size[2] = size[i--];
    while ((i >= 0) && (perm[i] != ndim1)) fused_size[2] *= size[i--];

    fused_weight[2] = fused_size[3];

    fused_size[1] = size[i--];
    while ((i >= 0) && (perm[i + 1] == (perm[i] + 1u)))
      fused_size[1] *= size[i--];

    fused_weight[1] = fused_size[2] * fused_weight[2];
  } else {
    fused_size[2] = 1ul;
    fused_weight[2] = 0ul;

    fused_size[1] = size[i--];
    while ((i >= 0) && (perm[i + 1] == (perm[i] + 1u)))
      fused_size[1] *= size[i--];

    fused_weight[1] = fused_size[3];
  }

  if (i >= 0) {
    fused_size[0] = size[i--];
    while (i >= 0) fused_size[0] *= size[i--];

    fused_weight[0] = fused_size[1] * fused_weight[1];
  } else {
    fused_size[0] = 1ul;
    fused_weight[0] = 0ul;
  }
}

/// Construct a permuted tensor copy

/// The expected signature of the input operations is:
/// \code
/// Result::value_type input_op(const Arg0::value_type, const
/// Args::value_type...) \endcode The expected signature of the output
/// operations is: \code void output_op(Result::value_type*, const
/// Result::value_type) \endcode \tparam InputOp The input operation type
/// \tparam OutputOp The output operation type
/// \tparam Result The result tensor type
/// \tparam Arg0 The first tensor argument type
/// \tparam Args The remaining tensor argument types
/// \param input_op The operation that is used to generate the output value
/// from the input arguments
/// \param output_op The operation that is used to set the value of the
/// result tensor given the element pointer and the result value
/// \param args The data pointers of the tensors to be permuted
/// \param perm The permutation that will be applied to the copy
template <typename InputOp, typename OutputOp, typename Result, typename Perm,
          typename Arg0, typename... Args,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
inline void permute(InputOp&& input_op, OutputOp&& output_op, Result& result,
                    const Perm& perm, const Arg0& arg0, const Args&... args) {
  detail::PermIndex perm_index_op(arg0.range(), outer(perm));

  // Cache constants
  const unsigned int ndim = arg0.range().rank();
  const unsigned int ndim1 = ndim - 1;
  const auto volume = arg0.range().volume();

  // Get pointer to arg extent
  const auto* MADNESS_RESTRICT const arg0_extent = arg0.range().extent_data();

  if (perm[ndim1] == ndim1) {
    // This is the simple case where the last dimension is not permuted.
    // Therefore, it can be shuffled in chunks.

    // Determine which dimensions can be permuted with the least significant
    // dimension.
    typename Result::ordinal_type block_size = arg0_extent[ndim1];
    for (int i = int(ndim1) - 1; i >= 0; --i) {
      if (int(perm[i]) != i) break;
      block_size *= arg0_extent[i];
    }

    // Combine the input and output operations
    auto op = [=](typename Result::pointer result,
                  typename Arg0::const_reference a0,
                  typename Args::const_reference... as) {
      output_op(result, input_op(a0, as...));
    };

    // Permute the data
    for (typename Result::ordinal_type index = 0ul; index < volume;
         index += block_size) {
      const typename Result::ordinal_type perm_index = perm_index_op(index);

      // Copy the block
      math::vector_ptr_op(op, block_size, result.data() + perm_index,
                          arg0.data() + index, (args.data() + index)...);
    }

  } else {
    // This is the more complicated case. Here we permute in terms of matrix
    // transposes. The data layout of the input and output matrices are
    // chosen such that they both contain stride one dimensions.

    // Here we partition the n dimensional index space, I, of the permute
    // tensor with up to four parts
    // {I_1, ..., I_i, I_i+1, ..., I_j, I_j+1, ..., I_k, I_k+1, ..., I_n}
    // where the subrange {I_k+1, ..., I_n} is the (fused) inner dimension
    // in the input tensor, and the subrange {I_i+1, ..., I_j} is the
    // (fused) inner dimension in the output tensor that has been mapped to
    // the input tensor. These ranges are used to form a set of matrices in
    // the input tensor that are transposed and copied to the output tensor.
    // The remaining (fused) index ranges {I_1, ..., I_i} and
    // {I_j+1, ..., I_k} are used to form the outer loop around the matrix
    // transpose operations. These outer ranges may or may not be zero size.
    typename Result::ordinal_type other_fused_size[4];
    typename Result::ordinal_type other_fused_weight[4];
    fuse_dimensions(other_fused_size, other_fused_weight,
                    arg0.range().extent_data(), perm);

    // Compute the fused stride for the result matrix transpose.
    const auto* MADNESS_RESTRICT const result_extent =
        result.range().extent_data();
    typename Result::ordinal_type result_outer_stride = 1ul;
    for (unsigned int i = perm[ndim1] + 1u; i < ndim; ++i)
      result_outer_stride *= result_extent[i];

    // Copy data from the input to the output matrix via a series of matrix
    // transposes.
    for (typename Result::ordinal_type i = 0ul; i < other_fused_size[0]; ++i) {
      typename Result::ordinal_type index = i * other_fused_weight[0];
      for (typename Result::ordinal_type j = 0ul; j < other_fused_size[2];
           ++j, index += other_fused_weight[2]) {
        // Compute the ordinal index of the input and output matrices.
        typename Result::ordinal_type perm_index = perm_index_op(index);

        math::transpose(input_op, output_op, other_fused_size[1],
                        other_fused_size[3], result_outer_stride,
                        result.data() + perm_index, other_fused_weight[1],
                        arg0.data() + index, (args.data() + index)...);
      }
    }
  }
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_PERMUTE_H__INCLUDED
