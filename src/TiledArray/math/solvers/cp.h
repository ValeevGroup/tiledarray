/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2023  Virginia Tech
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
 *  Department of Chemistry, Virginia Tech
 *
 *  cp.h
 *  March 10, 2023
 *
 */

#ifndef TILEDARRAY_MATH_SOLVERS_CP_H__INCLUDED
#define TILEDARRAY_MATH_SOLVERS_CP_H__INCLUDED

#include <TiledArray/math/solvers/cp/cp.h>
#include <TiledArray/math/solvers/cp/cp_als.h>
#include <TiledArray/math/solvers/cp/cp_reconstruct.h>

namespace TiledArray {
using TiledArray::math::cp::CP_ALS;
using TiledArray::math::cp::cp_reconstruct;
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_SOLVERS_CP_H__INCLUDED
