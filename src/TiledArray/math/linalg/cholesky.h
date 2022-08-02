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
 *  cholesky.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_CHOL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/math/linalg/scalapack/cholesky.h>
#endif
#if TILEDARRAY_HAS_TTG
#include <TiledArray/math/linalg/ttg/cholesky.h>
#endif
#include <TiledArray/math/linalg/non-distributed/cholesky.h>
#include <TiledArray/util/threads.h>

#define TILEDARRAY_ENABLE_USE_OF_TTG 0

namespace TiledArray::math::linalg {

template <typename Array>
auto cholesky(const Array& A, TiledRange l_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_ENABLE_USE_OF_TTG && TILEDARRAY_HAS_TTG
  return TiledArray::math::linalg::ttg::cholesky<Array>(A, l_trange);
#elif TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.elements_range().volume() > 10000000)
    return scalapack::cholesky<Array>(A, l_trange);
#endif
  return non_distributed::cholesky<Array>(A, l_trange);
}

template <bool Both = false, typename Array>
auto cholesky_linv(const Array& A, TiledRange l_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_ENABLE_USE_OF_TTG && TILEDARRAY_HAS_TTG
  return TiledArray::math::linalg::ttg::cholesky_linv<Both>(A, l_trange);
#elif TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.elements_range().volume() > 10000000)
    return scalapack::cholesky_linv<Both>(A, l_trange);
#endif
  return non_distributed::cholesky_linv<Both>(A, l_trange);
}

template <typename Array>
auto cholesky_solve(const Array& A, const Array& B,
                    TiledRange x_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.elements_range().volume() > 10000000)
    return scalapack::cholesky_solve<Array>(A, B, x_trange);
#endif
  return non_distributed::cholesky_solve(A, B, x_trange);
}

template <typename Array>
auto cholesky_lsolve(Op transpose, const Array& A, const Array& B,
                     TiledRange l_trange = TiledRange(),
                     TiledRange x_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.elements_range().volume() > 10000000)
    return scalapack::cholesky_lsolve<Array>(transpose, A, B, l_trange,
                                             x_trange);
#endif
  return non_distributed::cholesky_lsolve(transpose, A, B, l_trange, x_trange);
}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
using TiledArray::math::linalg::cholesky;
using TiledArray::math::linalg::cholesky_linv;
using TiledArray::math::linalg::cholesky_lsolve;
using TiledArray::math::linalg::cholesky_solve;
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_LINALG_CHOL_H__INCLUDED
