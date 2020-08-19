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
 *  utils.h
 *  May 20, 2013
 *
 */

#ifndef TILEDARRAY_ALGEBRA_UTILS_H__INCLUDED
#define TILEDARRAY_ALGEBRA_UTILS_H__INCLUDED

#include <sstream>

#include "../dist_array.h"
#include "../expressions/expr.h"
#include "TiledArray/util/annotation.h"

namespace TiledArray {

template <typename Tile, typename Policy>
inline size_t size(const DistArray<Tile, Policy>& a) {
  // this is the number of tiles
  if (a.size() > 0)  // assuming dense shape
    return a.trange().elements_range().volume();
  else
    return 0;
}

template <typename Tile, typename Policy>
inline DistArray<Tile, Policy> copy(const DistArray<Tile, Policy>& a) {
  return a;
}

template <typename Tile, typename Policy>
inline void zero(DistArray<Tile, Policy>& a) {
  const std::string vars =
      detail::dummy_annotation(a.trange().tiles_range().rank());
  a(vars) = typename DistArray<Tile, Policy>::element_type(0) * a(vars);
}

template <typename Tile, typename Policy>
typename DistArray<Tile, Policy>::element_type inline minabs_value(
    const DistArray<Tile, Policy>& a) {
  return a(detail::dummy_annotation(a.trange().tiles_range().rank())).abs_min();
}

template <typename Tile, typename Policy>
inline typename DistArray<Tile, Policy>::element_type maxabs_value(
    const DistArray<Tile, Policy>& a) {
  return a(detail::dummy_annotation(a.trange().tiles_range().rank())).abs_max();
}

template <typename Tile, typename Policy>
inline void vec_multiply(DistArray<Tile, Policy>& a1,
                         const DistArray<Tile, Policy>& a2) {
  const std::string vars =
      detail::dummy_annotation(a1.trange().tiles_range().rank());
  a1(vars) = a1(vars) * a2(vars);
}

template <typename Tile, typename Policy>
inline typename DistArray<Tile, Policy>::element_type dot_product(
    const DistArray<Tile, Policy>& a1, const DistArray<Tile, Policy>& a2) {
  const std::string vars =
      detail::dummy_annotation(a1.trange().tiles_range().rank());
  return a1(vars).dot(a2(vars)).get();
}

template <typename Left, typename Right>
inline typename TiledArray::expressions::ExprTrait<Left>::scalar_type dot(
    const TiledArray::expressions::Expr<Left>& a1,
    const TiledArray::expressions::Expr<Right>& a2) {
  static_assert(
      TiledArray::expressions::is_aliased<Left>::value,
      "no_alias() expressions are not allowed on the right-hand side of the "
      "assignment operator.");
  static_assert(
      TiledArray::expressions::is_aliased<Right>::value,
      "no_alias() expressions are not allowed on the right-hand side of the "
      "assignment operator.");
  return a1.dot(a2).get();
}

template <typename Tile, typename Policy>
inline void scale(
    DistArray<Tile, Policy>& a,
    typename DistArray<Tile, Policy>::element_type scaling_factor) {
  const std::string vars =
      detail::dummy_annotation(a.trange().tiles_range().rank());
  a(vars) = scaling_factor * a(vars);
}

template <typename Tile, typename Policy>
inline void axpy(DistArray<Tile, Policy>& y,
                 typename DistArray<Tile, Policy>::element_type a,
                 const DistArray<Tile, Policy>& x) {
  const std::string vars =
      detail::dummy_annotation(y.trange().tiles_range().rank());
  y(vars) = y(vars) + a * x(vars);
}

template <typename Tile, typename Policy>
inline void assign(DistArray<Tile, Policy>& m1,
                   const DistArray<Tile, Policy>& m2) {
  m1 = m2;
}

template <typename Tile, typename Policy>
inline typename DistArray<Tile, Policy>::scalar_type norm2(
    const DistArray<Tile, Policy>& a) {
  return std::sqrt(a(detail::dummy_annotation(a.trange().tiles_range().rank()))
                       .squared_norm());
}

template <typename Tile, typename Policy>
inline void print(const DistArray<Tile, Policy>& a, const char* label) {
  std::cout << label << ":\n" << a << "\n";
}

}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_UTILS_H__INCLUDED
