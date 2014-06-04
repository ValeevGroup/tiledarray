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
#include <TiledArray/array.h>

namespace TiledArray {

  namespace detail {

    inline std::string dummy_annotation(unsigned int DIM) {
      std::ostringstream oss;
      if (DIM > 0) oss << "i0";
      for(unsigned int d=1; d<DIM; ++d)
        oss << ",i" << d;
      return oss.str();
    }

  } // namespace detail

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline size_t size(const TiledArray::Array<T, DIM, Tile, Policy>& a) {
    // this is the number of tiles
    if (a.size() > 0) // assuming dense shape
      return a.trange().elements().volume();
    else
      return 0;
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline TiledArray::Array<T,DIM,Tile,Policy> clone(const TiledArray::Array<T,DIM,Tile,Policy>& a) {
    return a;
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline TiledArray::Array<T,DIM,Tile,Policy> copy(const TiledArray::Array<T,DIM,Tile,Policy>& a) {
    return a;
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline void zero(TiledArray::Array<T,DIM,Tile,Policy>& a) {
    const std::string vars = detail::dummy_annotation(DIM);
    a(vars) = typename TiledArray::Array<T,DIM,Tile,Policy>::element_type(0) * a(vars);
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  typename TiledArray::Array<T,DIM,Tile,Policy>::element_type
  inline minabs_value(const TiledArray::Array<T,DIM,Tile,Policy>& a) {
    return a(detail::dummy_annotation(DIM)).abs_min();
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline typename TiledArray::Array<T,DIM,Tile,Policy>::element_type
  maxabs_value(const TiledArray::Array<T,DIM,Tile,Policy>& a) {
    return a(detail::dummy_annotation(DIM)).abs_max();
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline void vec_multiply(TiledArray::Array<T,DIM,Tile,Policy>& a1,
                           const TiledArray::Array<T,DIM,Tile,Policy>& a2) {
    const std::string vars = detail::dummy_annotation(DIM);
    a1(vars) = a1(vars) * a2(vars);
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline typename TiledArray::Array<T,DIM,Tile,Policy>::element_type
  dot_product(const TiledArray::Array<T,DIM,Tile,Policy>& a1,
              const TiledArray::Array<T,DIM,Tile,Policy>& a2) {
    const std::string vars = detail::dummy_annotation(DIM);
    return a1(vars).dot(a2(vars)).get();
  }

  template <typename Left, typename Right>
  inline typename TiledArray::expressions::ExprTrait<Left>::scalar_type
  dot(const TiledArray::expressions::Expr<Left>& a1,
      const TiledArray::expressions::Expr<Right>& a2) {
    return a1.dot(a2).get();
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline void scale(TiledArray::Array<T,DIM,Tile,Policy>& a,
                    typename TiledArray::Array<T,DIM,Tile,Policy>::element_type scaling_factor) {
    const std::string vars = detail::dummy_annotation(DIM);
    a(vars) = scaling_factor * a(vars);
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline void axpy(TiledArray::Array<T,DIM,Tile,Policy>& y,
                   typename TiledArray::Array<T,DIM,Tile,Policy>::element_type a,
                   const TiledArray::Array<T,DIM,Tile,Policy>& x) {
    const std::string vars = detail::dummy_annotation(DIM);
    y(vars) = y(vars) + a * x(vars);
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline void assign(TiledArray::Array<T,DIM,Tile,Policy>& m1,
                     const TiledArray::Array<T,DIM,Tile,Policy>& m2) {
    m1 = m2;
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline typename TiledArray::Array<T,DIM,Tile,Policy>::element_type
  norm2(const TiledArray::Array<T,DIM,Tile,Policy>& a) {
    return std::sqrt(a(detail::dummy_annotation(DIM)).squared_norm());
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline void print(const TiledArray::Array<T,DIM,Tile,Policy>& a, const char* label) {
    std::cout << label << ":\n" << a << "\n";
  }

} // namespace TiledArray

#endif // TILEDARRAY_ALGEBRA_UTILS_H__INCLUDED
