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

    template <typename Perm, typename Size, typename Weight, typename Result,
        typename Arg>
    inline void permute(const unsigned int ndim, const Perm* restrict const perm,
        const Size* restrict const size, const Weight* restrict const weight,
        Result* restrict const result, const Arg* restrict const arg)
    {
      // Compute the volume and inverse-permuted weight of the result
      std::size_t volume = 1ul;
      Weight* restrict const result_weight = new Weight[ndim];

      for(int dim = int(ndim) - 1; dim >= 0; --dim) {
        const Perm inv_perm_dim = perm[perm[dim]];
        result_weight[inv_perm_dim] = volume;
        volume *= size[inv_perm_dim];
      }

      std::size_t index = 0ul;
      const std::size_t block_size = size[ndim - 1u];
      const std::size_t stride = result_weight[ndim - 1u];
      while(index < volume) {
        // Compute the permuted index for the current block
        std::size_t i = index;
        std::size_t perm_index = 0ul;
        for(unsigned int dim = 0u; dim < ndim; ++dim) {
          perm_index += (i / weight[dim]) * result_weight[dim];
          i %= weight[dim];
        }

        // Permute a block of arg
        const std::size_t end = index + block_size;
        for(; index < end; ++index, perm_index += stride)
          result[perm_index] = arg[index];
      }

      delete [] result_weight;
    }

    template <typename Perm, typename Size, typename Weight, typename Result,
        typename Arg, typename Op>
    inline void permute(const unsigned int ndim, const Perm* restrict const perm,
        const Size* restrict const size, const Weight* restrict const weight,
        Result* restrict const result, const Arg* restrict const arg, const Op& op)
    {
      // Compute the volume and inverse-permuted weight of the result
      std::size_t volume = 1ul;
      Weight* restrict const result_weight = new Weight[ndim];

      for(int dim = int(ndim) - 1; dim >= 0; --dim) {
        const Perm inv_perm_dim = perm[perm[dim]];
        result_weight[inv_perm_dim] = volume;
        volume *= size[inv_perm_dim];
      }

      std::size_t index = 0ul;
      const std::size_t block_size = size[ndim - 1u];
      const std::size_t stride = result_weight[ndim - 1u];
      while(index < volume) {
        // Compute the permuted index for the current block
        std::size_t i = index;
        std::size_t perm_index = 0ul;
        for(unsigned int dim = 0u; dim < ndim; ++dim) {
          perm_index += (i / weight[dim]) * result_weight[dim];
          i %= weight[dim];
        }

        // Permute a block of arg
        const std::size_t end = index + block_size;
        for(; index < end; ++index, perm_index += stride)
          result[perm_index] = op(arg[index]);
      }

      delete [] result_weight;
    }


    template <typename Perm, typename Size, typename Weight, typename Result,
        typename Left, typename Right, typename Op>
    inline void permute(const unsigned int ndim, const Perm* restrict const perm,
        const Size* restrict const size, const Weight* restrict const weight,
        Result* restrict const result, const Left* restrict const left,
        const Right* restrict const right, const Op& op)
    {
      // Compute the volume and inverse-permuted weight of the result
      std::size_t volume = 1ul;
      Weight* restrict const result_weight = new Weight[ndim];

      for(int dim = int(ndim) - 1; dim >= 0; --dim) {
        const Perm inv_perm_dim = perm[perm[dim]];
        result_weight[inv_perm_dim] = volume;
        volume *= size[inv_perm_dim];
      }

      std::size_t index = 0ul;
      const std::size_t block_size = size[ndim - 1u];
      const std::size_t stride = result_weight[ndim - 1u];
      while(index < volume) {
        // Compute the permuted index for the current block
        std::size_t i = index;
        std::size_t perm_index = 0ul;
        for(unsigned int dim = 0u; dim < ndim; ++dim) {
          perm_index += (i / weight[dim]) * result_weight[dim];
          i %= weight[dim];
        }

        // Permute a block of arg
        const std::size_t end = index + block_size;
        for(; index < end; ++index, perm_index += stride)
          result[perm_index] = op(left[index], right[index]);
      }

      delete [] result_weight;
    }

    /// Permute a tensor

    /// Permute \c tensor by \c perm and place the permuted result in \c result .
    /// \tparam ResT The result tensor element type
    /// \tparam ResA The result tensor allocator type
    /// \tparam ArgT The argument tensor element type
    /// \tparam ArgA The argument tensor allocator type
    /// \param[out] result The tensor that will hold the permuted result
    /// \param[in] perm The permutation to be applied to \c tensor
    /// \param[in] tensor The tensor to be permuted by \c perm
    template <typename ResT, typename ResA, typename ArgT, typename ArgA>
    inline void permute(Tensor<ResT,ResA>& result, const Permutation& perm,
        const Tensor<ArgT, ArgA>& tensor)
    {
      TA_ASSERT(perm.dim() == tensor.range().dim());

      // Create tensor to hold the result
      result = perm ^ tensor.range();

      permute(tensor.range().dim(), & perm.data().front(), tensor.range().size().data(),
          tensor.range().weight().data(), result.data(), tensor.data());
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
        const Tensor<ArgT, ArgA>& tensor, const Op& op)
    {
      TA_ASSERT(perm.dim() == tensor.range().dim());

      // Create tensor to hold the result
      result = perm ^ tensor.range();

      permute(tensor.range().dim(), & perm.data().front(), tensor.range().size().data(),
          tensor.range().weight().data(), result.data(), tensor.data(), op);
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
      TA_ASSERT(perm.dim() == left.range().dim());
      TA_ASSERT(left.range() == right.range());

      // Create tensor to hold the result
      result = perm ^ left.range();

      permute(left.range().dim(), & perm.data().front(), left.range().size().data(),
          left.range().weight().data(), result.data(), left.data(), right.data(), op);
    }

  }  // namespace math

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED
