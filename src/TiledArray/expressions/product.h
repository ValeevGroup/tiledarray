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
  }
  return result;
}

/// computes the tensor product type corresponding to the left and right
/// argument indices, and validates against the target indices
inline TensorProduct compute_product_type(const IndexList& left_indices,
                                          const IndexList& right_indices,
                                          const IndexList& target_indices) {
  auto result = compute_product_type(left_indices, right_indices);
  if (result == TensorProduct::Hadamard)
    TA_ASSERT(left_indices.is_permutation(target_indices));
  return result;
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_PRODUCT_H__INCLUDED
