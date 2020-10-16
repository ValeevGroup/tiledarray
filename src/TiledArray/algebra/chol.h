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
#else
#include <TiledArray/algebra/lapack/chol.h>
#endif

namespace TiledArray {
#if TILEDARRAY_HAS_SCALAPACK
using scalapack::cholesky;
using scalapack::cholesky_linv;
using scalapack::cholesky_lsolve;
using scalapack::cholesky_solve;
else using lapack::cholesky;
using lapack::cholesky_linv;
using lapack::cholesky_lsolve;
using lapack::cholesky_solve;
#endif

}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_CHOL_H__INCLUDED
