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
 *  qr.h
 *  Created:    2 August, 2023
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SLATE_QR_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_QR_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/util.h>

namespace TiledArray::math::linalg::slate {

template <bool QOnly, typename ArrayV>
auto householder_qr( const ArrayV& V ) {

  // SLATE does not yet have ORGQR/UNGQR
  // https://github.com/icl-utk-edu/slate/issues/80
  TA_EXCEPTION("SLATE + QR NYI");

}

} // namespace TiledArray::math::linalg::slate

#endif // TILEDARRAY_HAS_SLATE

#endif // TILEDARRAY_MATH_LINALG_SLATE_QR_H__INCLUDED
