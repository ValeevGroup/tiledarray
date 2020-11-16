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
 *  chol.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_CHOL_H__INCLUDED
#define TILEDARRAY_ALGEBRA_CHOL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/algebra/scalapack/chol.h>
#endif
#include <TiledArray/algebra/lapack/chol.h>

namespace TiledArray {

template <typename Array>
auto cholesky(const Array& A, TiledRange l_trange = TiledRange()) {
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.range().volume() > 10000000)
    return scalapack::cholesky<Array>(A, l_trange);
  else
#endif
    return lapack::cholesky<Array>(A, l_trange);
}

template <typename Array, bool RetL = false>
auto cholesky_linv(const Array& A, TiledRange l_trange = TiledRange()) {
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.range().volume() > 10000000)
    return scalapack::cholesky_linv<Array, RetL>(A, l_trange);
  else
#endif
    return lapack::cholesky_linv<Array, RetL>(A, l_trange);
}

template <typename Array>
auto cholesky_solve(const Array& A, const Array& B,
                    TiledRange x_trange = TiledRange()) {
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.range().volume() > 10000000)
    return scalapack::cholesky_solve<Array>(A, B, x_trange);
  else
#endif
    return lapack::cholesky_solve<Array>(A, B, x_trange);
}

template <typename Array>
auto cholesky_lsolve(TransposeFlag transpose, const Array& A, const Array& B,
                     TiledRange l_trange = TiledRange(),
                     TiledRange x_trange = TiledRange()) {
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.range().volume() > 10000000)
    return scalapack::cholesky_lsolve<Array>(transpose, A, B, l_trange,
                                             x_trange);
  else
#endif
    return lapack::cholesky_lsolve<Array>(transpose, A, B, l_trange, x_trange);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_CHOL_H__INCLUDED
