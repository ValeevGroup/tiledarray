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
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_LU_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LU_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/algebra/scalapack/lu.h>
#endif
#include <TiledArray/algebra/non-distributed/lu.h>

namespace TiledArray {

#if TILEDARRAY_HAS_SCALAPACK
using scalapack::lu_inv;
using scalapack::lu_solve;
#else
#endif

}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_LU_H__INCLUDED
