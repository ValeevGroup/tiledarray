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
 */

#ifndef TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED
#define TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED

#include <TiledArray/tensor.h>
#include <TiledArray/permutation.h>

namespace TiledArray {
  namespace math {

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

      // Construct the inverse permuted weight and size for this tensor
      std::vector<std::size_t> ip_weight = (-perm) ^ result.range().weight();
      const typename Tensor<ArgT,ArgA>::range_type::size_array& start =
          tensor.range().start();

      // Coordinated iterator for the value range
      typename Tensor<ArgT,ArgA>::range_type::const_iterator it =
          tensor.range().begin();

      // permute the data
      const std::size_t end = result.size();
      for(std::size_t i = 0ul; i != end; ++i, ++it)
        result[TiledArray::detail::calc_ordinal(*it, ip_weight, start)] = tensor[i];
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

      // Construct the inverse permuted weight and size for this tensor
      std::vector<std::size_t> ip_weight = (-perm) ^ result.range().weight();
      const typename Tensor<ArgT,ArgA>::range_type::size_array& start = tensor.range().start();

      // Coordinated iterator for the value range
      typename Tensor<ArgT,ArgA>::range_type::const_iterator it =
          tensor.range().begin();

      // permute the data
      const std::size_t end = result.size();
      for(std::size_t i = 0ul; i != end; ++i, ++it)
        result[TiledArray::detail::calc_ordinal(*it, ip_weight, start)] = op(tensor[i]);
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

      // Construct the inverse permuted weight and size for this tensor
      std::vector<std::size_t> ip_weight = (-perm) ^ result.range().weight();
      const typename Tensor<LeftT, LeftA>::range_type::size_array& start =
          left.range().start();

      // Coordinated iterator for the value range
      typename Tensor<LeftT, LeftA>::range_type::const_iterator it =
          left.range().begin();

      // permute the data
      const std::size_t end = result.size();
      for(std::size_t i = 0ul; i != end; ++i, ++it)
        result[TiledArray::detail::calc_ordinal(*it, ip_weight, start)] = op(left[i], right[i]);
    }

  }  // namespace math

  /// Permute a tensor

  /// Permute \c tensor by \c perm and place the permuted result in \c result .
  /// \tparam T The tensor element type
  /// \tparam A The tensor allocator type
  /// \param perm The permutation to be applied to \c tensor
  /// \param tensor The tensor to be permuted by \c perm
  template <typename T, typename A>
  inline Tensor<T,A> operator^(const Permutation& perm, const Tensor<T, A>& tensor) {
    // Create tensor to hold the result
    Tensor<T,A> result;
    math::permute(result, perm, tensor);
    return result;
  }

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_PERMUTE_H__INCLUDED
