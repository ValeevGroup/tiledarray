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
 *  David Williams-Young
 *  Computational Research Division, Lawrence Berkeley National Laboratory
 *
 *  svd.h
 *  Created:    12 June, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_FORWARD_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_FORWARD_H__INCLUDED

#include <TiledArray/config.h>
#include <type_traits>

namespace TiledArray::math::linalg {

enum TransposeFlag { NoTranspose, Transpose, ConjTranspose };

struct SVD {
  enum Vectors {
    ValuesOnly,
    LeftVectors,
    RightVectors,
    AllVectors
  };
};

}  // namespace TiledArray::math::linalg

namespace TiledArray {
  using TiledArray::math::linalg::TransposeFlag;
  using TiledArray::math::linalg::SVD;
}

#endif  // TILEDARRAY_MATH_LINALG_FORWARD_H__INCLUDED
