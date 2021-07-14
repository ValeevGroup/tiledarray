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
 *  heig.h
 *  Created:  16 October,  2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_HEIG_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/math/linalg/scalapack/heig.h>
#endif
#include <TiledArray/math/linalg/non-distributed/heig.h>
#include <TiledArray/util/threads.h>

namespace TiledArray::math::linalg {

template <typename Array>
auto heig(const Array& A, TiledRange evec_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.range().volume() > 10000000) {
    return scalapack::heig(A, evec_trange);
  }
#endif
  return non_distributed::heig(A, evec_trange);
}

template <typename ArrayA, typename ArrayB, typename EVecType = ArrayA>
auto heig(const ArrayA& A, const ArrayB& B, TiledRange evec_trange = TiledRange()) {
  TA_MAX_THREADS;
#if TILEDARRAY_HAS_SCALAPACK
  if (A.world().size() > 1 && A.range().volume() > 10000000) {
    return scalapack::heig(A, B, evec_trange);
  }
#endif
  return non_distributed::heig(A, B, evec_trange);
}

}  // namespace TiledArray::math::linalg

namespace TiledArray {
  using TiledArray::math::linalg::heig;
}

#endif  // TILEDARRAY_MATH_LINALG_HEIG_H__INCLUDED
