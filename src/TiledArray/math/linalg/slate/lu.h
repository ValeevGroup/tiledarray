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
  auto A_slate = array_to_slate(A);
  auto B_slate = array_to_slate(B);

  //for(auto it = 0; it < A_slate.mt(); ++it)
  //for(auto jt = 0; jt < A_slate.nt(); ++jt) {
  //  auto T = B_slate(it,jt);
  //  std::cout << "TILE(" << it << "," << jt << "): ";
  //  for( auto i = 0; i < T.mb()*T.nb(); ++i )
  //     printf("%.10f ", T.data()[i]);
  //  std::cout << std::endl;
  //}

  // Solve Linear System
  world.gop.fence();  // stage SLATE execution
  ::slate::lu_solve( A_slate, B_slate );
  world.gop.fence();  // stage SLATE execution

  // Convert solution to TA
  return slate_to_array<ArrayB>(B_slate, world);
}

}  // namespace TiledArray::math::linalg::scalapack

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_MATH_LINALG_SCALAPACK_LU_H__INCLUDED
