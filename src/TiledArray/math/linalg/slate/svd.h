/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2023 Virginia Tech
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
 *  David Williams-Young
 *  Applied Mathematics and Computational Research Division,
 *  Lawrence Berkeley National Laboratory
 *
 *  cholesky.h
 *  Created:    24 July, 2023
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SLATE_SVD_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_SVD_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/util.h>
#include <TiledArray/math/linalg/forward.h>

namespace TiledArray::math::linalg::slate {

template <SVD::Vectors Vectors, typename Array>
auto svd(const Array& A, TA::TiledRange , TA::TiledRange ) {

  constexpr bool need_uv = (Vectors == SVD::AllVectors);
  constexpr bool need_u = (Vectors == SVD::LeftVectors) or need_uv;
  constexpr bool need_vt = (Vectors == SVD::RightVectors) or need_uv;
  constexpr bool vals_only = not need_u and not need_vt;

  static_assert(vals_only, "SLATE + SVD Vectors NYI");
  std::cout << "IN SLATE SVD" << std::endl;

  using element_type   = typename std::remove_cv_t<Array>::element_type;
  auto& world = A.world();

  // Convert to SLATE
  auto matrix = array_to_slate(A);

  // Allocate space for singular values
  const auto M = matrix.m();
  const auto N = matrix.n();
  const auto SVD_SIZE = std::min(M,N);
  std::vector<::blas::real_type<element_type>> S(SVD_SIZE);

  // Perform GESVD 
  world.gop.fence();  // stage SLATE execution
  if constexpr (vals_only)  {
    ::slate::svd_vals(matrix, S); 
    return S;
  }

}

} // namespace TiledArray::math::linalg::slate

#endif // TILEDARRAY_HAS_SLATE

#endif // TILEDARRAY_MATH_LINALG_SLATE_SVD_H__INCLUDED
