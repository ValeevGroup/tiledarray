/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  util.h
 *  May 20, 2013
 *
 */

#ifndef TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED

#include "TiledArray/dist_array.h"

namespace TiledArray::math::linalg {

template <typename Tile, typename Policy>
inline void vec_multiply(DistArray<Tile, Policy>& a1,
                         const DistArray<Tile, Policy>& a2) {
  auto vars = TiledArray::detail::dummy_annotation(rank(a1));
  a1(vars) = a1(vars) * a2(vars);
}

template <typename Tile, typename Policy, typename S>
inline void scale(DistArray<Tile, Policy>& a, S scaling_factor) {
  using numeric_type = typename DistArray<Tile, Policy>::numeric_type;
  auto vars = TiledArray::detail::dummy_annotation(rank(a));
  a(vars) = numeric_type(scaling_factor) * a(vars);
}

template <typename Tile, typename Policy>
inline void zero(DistArray<Tile, Policy>& a) {
  scale(a, 0);
}

template <typename Tile, typename Policy, typename S>
inline void axpy(DistArray<Tile, Policy>& y, S alpha,
                 const DistArray<Tile, Policy>& x) {
  using numeric_type = typename DistArray<Tile, Policy>::numeric_type;
  auto vars = TiledArray::detail::dummy_annotation(rank(y));
  y(vars) = y(vars) + numeric_type(alpha) * x(vars);
}

}  // namespace TiledArray::math::linalg

#endif  // TILEDARRAY_MATH_LINALG_BASIC_H__INCLUDED
