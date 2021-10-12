/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_EXPRESSIONS_DETAIL_EINSUM_TRAITS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_DETAIL_EINSUM_TRAITS_H__INCLUDED

#include <type_traits>                      // For std::conditional_t
#include "TiledArray/tensor/type_traits.h"  // For is_tensor_of_tensors_v

namespace TiledArray::expressions::detail {

/** @brief Struct used to analyze the types of the tensors input to `einsum`.
 *
 *  The EinsumTraits struct encapsulates some minor template meta-programming
 *  which needs to occur in order for the main `einsum` API to dispatch
 *  correctly.
 *
 *  @tparam LHSArray The type of the array in the first tensor provided to
 *                   `einsum`. Expected to satisfy the concept of a tile.
 *  @tparam RHSArray The type of the array in the second tensor provided to
 *                   `einsum`. Expected to satisfy the concept of a tile.
 */
template <typename LHSArray, typename RHSArray>
struct EinsumTraits {
 private:
  /// Alias to shorten typedefs so they fit on one line
  template <typename T>
  static constexpr bool tot = TiledArray::detail::is_tensor_of_tensor_v<T>;

 public:
  /// True if @p LHSArray is a tile in a tensor of tensors. False otherwise.
  static constexpr bool lhs_is_tot = tot<LHSArray>;

  /// True if @p RHSArray is a tile in a tensor of tensors. False otherwise
  static constexpr bool rhs_is_tot = tot<RHSArray>;

  /// True @p LHSArray and @p RHSArray are types of ToTs
  static constexpr bool both_are_tots = lhs_is_tot && rhs_is_tot;

  /// Type of the tensor returned by einsum
  using return_type = std::conditional_t<lhs_is_tot, LHSArray, RHSArray>;
};

/** @brief Partial specialization of EinsumTraits so that it "does the right
 *         thing" when the templater parameters are DistArray instances instead
 *         of tiles.
 *
 *  Throughout TiledArray it is common to deal with DistArray instances, but it
 *  is the tile types that determine whether a particular tensor is
 *  non-hierarchical or a tensor-of-tensors. This partial specialization is
 *  instantiated when the template parameters to EinsumTraits are DistArray
 *  instances instead of tiles. This specialization unwraps the tile types and
 *  forwards them to base class, which is also the primary template. The
 *  result is instances of this partial specialization have the same TMP values
 *  as the primary template, but are populated from DistArray instances instead
 *  of tiles.
 *
 *  @tparam LHSArray The tile type in the first tensor passed to `einsum`.
 *                   Expected to satisfy the concept of tile.
 *  @tparam RHSArray The tile type in the second tensor passed to `einsum`.
 *                   Expected to satisfy the concept of tile.
 *  @tparam LHSPolicy The policy of the first tensor passed to `einsum`.
 *                    Expected to satisfy the concept of policy.
 *  @tparam RHSPolicy The policy of the second tensor passed to `einsum`.
 *                    Expected to satisfy the concept of policy.
 */
template <typename LHSArray, typename RHSArray, typename LHSPolicy,
          typename RHSPolicy>
struct EinsumTraits<DistArray<LHSArray, LHSPolicy>,
                    DistArray<RHSArray, RHSPolicy>>
    : EinsumTraits<LHSArray, RHSArray> {};

}  // namespace TiledArray::expressions::detail

#endif
