/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2023  Virginia Tech
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
 *  Karl Pierce
 *  Department of Chemistry, Virginia Tech
 *
 *  cp.h
 *  July 12, 2022
 *
 */

#ifndef TILEDARRAY_MATH_SOLVERS_CP_CP_RECONSTRUCT_H
#define TILEDARRAY_MATH_SOLVERS_CP_CP_RECONSTRUCT_H

#include <TiledArray/conversions/btas.h>
#include <TiledArray/expressions/einsum.h>
#include <btas/btas.h>
#include <tiledarray.h>

namespace TiledArray::math::cp {

/// This is a function which reconstructs the target tensor T from
/// the CP factor matrices
/// \param[in] cp_factors The set of order-2 factor matrices in the the
/// first mode of each tensor being the rank mode.
/// \param[in] lambda If lambda is initialized, the CP rank scaling factors
/// (similar to singular values) else these factors are assumed to be folded
/// into @c cp_factors
template <typename Tile, typename Policy>
auto cp_reconstruct(
    const std::vector<DistArray<Tile, Policy>> cp_factors,
    DistArray<Tile, Policy> lambda = DistArray<Tile, Policy>()) {
  using Array = DistArray<Tile, Policy>;
  TA_ASSERT(!cp_factors.empty(), "CP factor matrices have not been computed");
  std::string lhs("r,0"), rhs("r,"), final("r,0");
  Array krp = cp_factors[0];
  if (lambda.is_initialized()) {
    krp("r,0") = lambda("r") * krp("r,0");
  }
  auto ndim = cp_factors.size();
  for (size_t i = 1; i < ndim - 1; ++i) {
    rhs += std::to_string(i);
    final += "," + std::to_string(i);
    krp = expressions::einsum(krp(lhs), cp_factors[i](rhs), final);
    lhs = final;
    rhs.pop_back();
  }
  rhs += std::to_string(ndim - 1);
  final.erase(final.begin(), final.begin() + 2);
  final += "," + std::to_string(ndim - 1);
  krp(final) = krp(lhs) * cp_factors[ndim - 1](rhs);
  return krp;
}

}  // namespace TiledArray::math::cp

#endif  // TILEDARRAY_MATH_SOLVERS_CP_CP_RECONSTRUCT_H
