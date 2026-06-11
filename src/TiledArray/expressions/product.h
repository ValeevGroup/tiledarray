/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  permopt.h
 *  Nov 2, 2020
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_PRODUCT_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_PRODUCT_H__INCLUDED

#include <TiledArray/expressions/index_list.h>

namespace TiledArray {
namespace expressions {

/// types of binary tensor products known to TiledArray
enum class TensorProduct {
  /// fused indices only
  Hadamard,
  /// (at least 1) free and (at least 1) contracted indices
  Contraction,
  /// free, fused, and contracted indices
  General,
  /// no indices on one, free indices on the other; only used for inner index
  /// products in mixed nested products (ToT x T)
  Scale,
  /// invalid
  Invalid = -1
};

/// computes the tensor product type corresponding to the left and right
/// argument indices \param left_indices the left argument index list \param
/// right_indices the right argument index list \return TensorProduct::Invalid
/// is either argument is empty, TensorProduct::Hadamard if the arguments are
/// related by a permutation, else TensorProduct::Contraction
inline TensorProduct compute_product_type(const IndexList& left_indices,
                                          const IndexList& right_indices) {
  TensorProduct result = TensorProduct::Invalid;
  if (left_indices && right_indices) {
    if (left_indices.size() == right_indices.size() &&
        left_indices.is_permutation(right_indices))
      result = TensorProduct::Hadamard;
    else
      result = TensorProduct::Contraction;
  } else if ((left_indices && !right_indices) ||
             (!left_indices && right_indices)) {  // used for ToT*T or T*ToT
    result = TensorProduct::Scale;
  }
  return result;
}

/// computes the tensor product type corresponding to the left and right
/// argument indices, given the target indices
///
/// Unlike the 2-argument overload, this can detect TensorProduct::General:
/// the target determines the role of each index, so an index shared by both
/// arguments is *fused* (Hadamard) if it survives into the target and
/// *contracted* if it does not. The 2-argument overload, lacking a target,
/// follows the bottom-up convention that every shared index is contracted.
/// \return
///   - TensorProduct::Hadamard if left, right, and target are all related by
///     permutations (fused indices only),
///   - TensorProduct::General if at least one shared index is fused (appears
///     in left, right, AND target) alongside contracted and/or free indices,
///   - TensorProduct::Contraction if no shared index is fused,
///   - else as the 2-argument overload.
inline TensorProduct compute_product_type(const IndexList& left_indices,
                                          const IndexList& right_indices,
                                          const IndexList& target_indices) {
  auto result = compute_product_type(left_indices, right_indices);
  if (result == TensorProduct::Hadamard) {
    // left ≅ right; pure Hadamard requires the target to keep every index.
    // A target that omits some shared indices implies they are contracted
    // (a Hadamard-reduction, e.g. "i,j" * "i,j" -> "i"): fused + contracted
    // coexist => General.
    if (!left_indices.is_permutation(target_indices))
      result = TensorProduct::General;
  } else if (result == TensorProduct::Contraction) {
    // an index of the target that appears in both arguments is fused, not
    // contracted: fused + (free and/or contracted) => General.
    for (auto&& idx : target_indices) {
      if (left_indices.count(idx) && right_indices.count(idx)) {
        result = TensorProduct::General;
        break;
      }
    }
  }
  return result;
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_PRODUCT_H__INCLUDED
