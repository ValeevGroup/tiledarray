/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020 Virginia Tech
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
 *
 *  lu.h
 *  Created:  16 October,  2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_LU_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_LU_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/math/linalg/scalapack/lu.h>
#endif
#include <TiledArray/math/linalg/basic.h>
#include <TiledArray/math/linalg/non-distributed/lu.h>
#include <TiledArray/util/threads.h>

namespace TiledArray::math::linalg {

template <typename ArrayA, typename ArrayB>
auto lu_solve(const ArrayA& A, const ArrayB& B,
              TiledRange x_trange = TiledRange()) {
  TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(lu_solve(A, B, x_trange), A);
}

template <typename Array>
auto lu_inv(const Array& A, TiledRange ainv_trange = TiledRange()) {
  TILEDARRAY_MATH_LINALG_DISPATCH_WO_TTG(lu_inv(A, ainv_trange), A);
}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
using TiledArray::math::linalg::lu_inv;
using TiledArray::math::linalg::lu_solve;
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_LINALG_LU_H__INCLUDED
