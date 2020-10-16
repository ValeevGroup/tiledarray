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
#ifndef TILEDARRAY_ALGEBRA_SVD_H__INCLUDED
#define TILEDARRAY_ALGEBRA_SVD_H__INCLUDED

#include <TiledArray/config.h>
#ifdef TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/algebra/scalapack/svd.h>
#else
// include eigen
#endif  // TILEDARRAY_HAS_SCALAPACK

namespace TiledArray {

#ifdef TILEDARRAY_HAS_SCALAPACK
using scalapack::svd;
#else
#endif

}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_SVD_H__INCLUDED
