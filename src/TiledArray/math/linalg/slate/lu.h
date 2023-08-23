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
#ifndef TILEDARRAY_MATH_LINALG_SLATE_LU_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_LU_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/util.h>
namespace TiledArray::math::linalg::slate {

template <typename ArrayA, typename ArrayB>
auto lu_solve(const ArrayA& A, const ArrayB& B) {

  using element_type   = typename std::remove_cv_t<ArrayA>::element_type;
  auto& world = A.world();
  /*
  if( world != B.world() ) {
    TA_EXCEPTION("A and B must be distributed on same MADWorld context");
  }
  */

  // Convert to SLATE
  world.gop.fence();  // stage SLATE execution
  auto A_slate = array_to_slate(A);
  auto B_slate = array_to_slate(B);
  world.gop.fence();  // stage SLATE execution

  // Solve Linear System
  ::slate::lu_solve( A_slate, B_slate );

  // Convert solution to TA
  auto X = slate_to_array<ArrayB>(B_slate, world);
  world.gop.fence();  // stage SLATE execution

  return X;
}

template <typename Array>
auto lu_inv(const Array& A) {

  using element_type   = typename std::remove_cv_t<Array>::element_type;
  auto& world = A.world();

  // Convert to SLATE
  world.gop.fence();  // stage SLATE execution
  auto A_slate = array_to_slate(A);
  world.gop.fence();  // stage SLATE execution

  // Perform LU Factorization 
  ::slate::Pivots pivots;
  ::slate::lu_factor(A_slate, pivots);

  // Invert from factors
  ::slate::lu_inverse_using_factor(A_slate, pivots);

  // Convert inverse to TA
  auto X = slate_to_array<Array>(A_slate, world);
  world.gop.fence();  // stage SLATE execution

  return X;
}

}  // namespace TiledArray::math::linalg::scalapack

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_MATH_LINALG_SCALAPACK_LU_H__INCLUDED
