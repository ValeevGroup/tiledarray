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
 *  svd.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SVD_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SVD_H__INCLUDED

#include <TiledArray/config.h>
#ifdef TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/math/linalg/scalapack/svd.h>
#endif  // TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/math/linalg/non-distributed/svd.h>
#include <TiledArray/util/threads.h>

namespace TiledArray::math::linalg {

template <SVD::Vectors Vectors, typename Array>
auto svd(const Array& A, TiledRange u_trange = TiledRange(), TiledRange vt_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.elements_range().volume() > 10000000) {
    return scalapack::svd<Vectors>(A, u_trange, vt_trange);
  }
#endif
  return non_distributed::svd<Vectors>(A, u_trange, vt_trange);
}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
  using TiledArray::math::linalg::svd;
}

#endif  // TILEDARRAY_MATH_LINALG_SVD_H__INCLUDED
