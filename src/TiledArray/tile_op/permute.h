/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  May 7, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED
#define TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED

#include <TiledArray/permutation.h>

namespace TiledArray {

  class Permutation;
  template <typename, typename> class Tensor;

  namespace math {

    /// Permute a tensor

    /// Permute \c tensor by \c perm and place the permuted result in \c result .
    /// \tparam ResT The result tensor element type
    /// \tparam ResA The result tensor allocator type
    /// \tparam ArgT The argument tensor element type
    /// \tparam ArgA The argument tensor allocator type
    /// \param[out] result The tensor that will hold the result
    /// \param[in] perm The permutation to be applied to \c arg
    /// \param[in] arg The tensor to be permuted
    template <typename ResT, typename ResA, typename ArgT, typename ArgA>
    inline void permute(Tensor<ResT,ResA>& result, const Permutation& perm,
        const Tensor<ArgT, ArgA>& arg)
    {
      // Define convenience types
      typedef typename Tensor<ArgT, ArgA>::size_type size_type;

      // Check input and allocate result if necessary.
      TA_ASSERT(perm);
      TA_ASSERT(! arg.empty());
      TA_ASSERT(perm.dim() == arg.range().dim());
      if(result.empty()) {
        // Create tensor to hold the result
        result = perm ^ arg.range();
      } else {
        TA_ASSERT(result.range() == (perm ^ arg.range()));
      }

      // Construct the inverse permuted result weight
      const std::vector<typename Tensor<ArgT, ArgA>::range_type::size_type> inv_weight =
          -perm ^ result.range().weight();

      // Cache constants
      const unsigned int ndim = arg.range().dim();
      const size_type volume = arg.range().volume();
      const size_type block_size = arg.range().size()[ndim - 1u];
      const size_type stride = inv_weight[ndim - 1u];

      {
        // Get pointers to weight arrays
        const size_type* restrict const weight = arg.range().weight().data();
        const size_type* restrict const inv_result_weight = & inv_weight.front();

        // Get pointers to tensor data arrays
        typename Tensor<ArgT, ArgA>::const_pointer restrict const arg_data = arg.data();
        typename Tensor<ResT,ResA>::pointer restrict const result_data = result.data();

        size_type index = 0ul;
        while(index < volume) {
          // Compute the permuted index for the current block
          size_type i = index;
          size_type perm_index = 0ul;
          for(unsigned int dim = 0u; dim < ndim; ++dim) {
            const size_type weight_dim = weight[dim];
            const size_type inv_result_weight_dim = inv_result_weight[dim];
            perm_index += i / weight_dim * inv_result_weight_dim;
            i %= weight_dim;
          }

          // Permute a block of arg
          const size_type end = index + block_size;
          for(; index < end; ++index, perm_index += stride)
            result_data[perm_index] = arg_data[index];
        }
      }
    }

    /// Apply an operation to a tensor and permute the result

    /// Permute \c tensor by \c perm and place the permuted result in \c result .
    /// \c Op is applied to each element of the result.
    /// \tparam ResT The result tensor element type
    /// \tparam ResA The result tensor allocator type
    /// \tparam ArgT The argument tensor element type
    /// \tparam ArgA The argument tensor allocator type
    /// \tparam Op The element operation type
    /// \param[out] result The tensor that will hold the permuted result
    /// \param[in] perm The permutation to be applied to \c tensor
    /// \param[in] tensor The tensor to be permuted by \c perm
    /// \param[in] op The operation to be applied to each element of \c result
    template <typename ResT, typename ResA, typename ArgT, typename ArgA, typename Op>
    inline void permute(Tensor<ResT,ResA>& result, const Permutation& perm,
        const Tensor<ArgT, ArgA>& arg, const Op& op)
    {
      // Define convenience types
      typedef typename Tensor<ArgT, ArgA>::size_type size_type;

      // Check input and allocate result if necessary.
      TA_ASSERT(perm);
      TA_ASSERT(! arg.empty());
      TA_ASSERT(perm.dim() == arg.range().dim());
      if(result.empty()) {
        // Create tensor to hold the result
        result = perm ^ arg.range();
      } else {
        TA_ASSERT(result.range() == (perm ^ arg.range()));
      }
      TA_ASSERT(result.data() != arg.data());

      // Construct the inverse permuted result weight
      const std::vector<typename Tensor<ArgT, ArgA>::range_type::size_type> inv_weight =
          -perm ^ result.range().weight();

      // Cache constants
      const unsigned int ndim = arg.range().dim();
      const size_type volume = arg.range().volume();
      const size_type block_size = arg.range().size()[ndim - 1u];
      const size_type stride = inv_weight[ndim - 1u];

      {
        // Get pointers to weight arrays
        const size_type* restrict const weight = arg.range().weight().data();
        const size_type* restrict const inv_result_weight = & inv_weight.front();

        // Get pointers to tensor data arrays
        typename Tensor<ArgT, ArgA>::const_pointer restrict const arg_data = arg.data();
        typename Tensor<ResT,ResA>::pointer restrict const result_data = result.data();

        size_type index = 0ul;
        while(index < volume) {
          // Compute the permuted index for the current block
          size_type i = index;
          size_type perm_index = 0ul;
          for(unsigned int dim = 0u; dim < ndim; ++dim) {
            const size_type weight_dim = weight[dim];
            const size_type inv_result_weight_dim = inv_result_weight[dim];
            perm_index += i / weight_dim * inv_result_weight_dim;
            i %= weight_dim;
          }

          // Permute a block of arg
          const size_type end = index + block_size;
          for(; index < end; ++index, perm_index += stride)
            result_data[perm_index] = op(arg_data[index]);
        }
      }
    }

    /// Apply an operation to a pair of tensors and permute the result

    /// Permute \c tensor by \c perm and place the permuted result in \c result .
    /// \c Op is applied to each element of the result.
    /// \tparam ResT The result tensor element type
    /// \tparam ResA The result tensor allocator type
    /// \tparam LeftT The left-hand tensor element type
    /// \tparam LeftA The left-tensor allocator type
    /// \tparam RightT The right-tensor element type
    /// \tparam RightA The right-tensor allocator type
    /// \tparam Op The element operation type
    /// \param[out] result The tensor that will hold the permuted result
    /// \param[in] perm The permutation to be applied to \c tensor
    /// \param[in] left The left-hand tensor argument
    /// \param[in] right The right-hand tensor argument
    /// \param[in] op The operation to be applied to each element of \c result
    template <typename ResT, typename ResA, typename LeftT, typename LeftA,
        typename RightT, typename RightA, typename Op>
    inline void permute(Tensor<ResT,ResA>& result, const Permutation& perm,
        const Tensor<LeftT, LeftA>& left, const Tensor<RightT, RightA>& right, const Op& op)
    {

      // Define convenience types
      typedef typename Tensor<LeftT, LeftA>::size_type size_type;

      // Check input and allocate result if necessary.
      TA_ASSERT(perm);
      TA_ASSERT(! left.empty());
      TA_ASSERT(! right.empty());
      TA_ASSERT(perm.dim() == left.range().dim());
      TA_ASSERT(left.range() == right.range());
      if(result.empty()) {
        // Create tensor to hold the result
        result = perm ^ left.range();
      } else {
        TA_ASSERT(result.range() == (perm ^ left.range()));
      }

      // Construct the inverse permuted result weight
      const std::vector<typename Tensor<LeftT, LeftA>::range_type::size_type> inv_weight =
          -perm ^ result.range().weight();

      // Cache constants
      const unsigned int ndim = left.range().dim();
      const size_type volume = left.range().volume();
      const size_type block_size = left.range().size()[ndim - 1u];
      const size_type stride = inv_weight[ndim - 1u];

      {
        // Get pointers to weight arrays
        const size_type* restrict const weight = left.range().weight().data();
        const size_type* restrict const inv_result_weight = & inv_weight.front();

        // Get pointers to tensor data arrays
        typename Tensor<LeftT, LeftA>::const_pointer restrict const left_data = left.data();
        typename Tensor<RightT, RightA>::const_pointer restrict const right_data = right.data();
        typename Tensor<ResT,ResA>::pointer restrict const result_data = result.data();

        size_type index = 0ul;
        while(index < volume) {
          // Compute the permuted index for the current block
          size_type i = index;
          size_type perm_index = 0ul;
          for(unsigned int dim = 0u; dim < ndim; ++dim) {
            const size_type weight_dim = weight[dim];
            const size_type inv_result_weight_dim = inv_result_weight[dim];
            perm_index += i / weight_dim * inv_result_weight_dim;
            i %= weight_dim;
          }

          // Permute a block of arg
          const size_type end = index + block_size;
          for(; index < end; ++index, perm_index += stride)
            result_data[perm_index] = op(left_data[index], right_data[index]);
        }
      }
    }

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED
