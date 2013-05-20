/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
      for(auto d=1; d<DIM; ++d)
        oss << ",i" << d;
      return oss.str();
    }

  } // namespace TiledArray::detail

  template <typename T, unsigned int DIM, typename Tile>
  inline size_t size(const TiledArray::Array<T, DIM, Tile>& a) {
    // this is the number of tiles
    if (a.size() > 0) // assuming dense shape
      return a.trange().elements().volume();
    else
      return 0;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline TiledArray::Array<T,DIM,Tile> clone(const TiledArray::Array<T,DIM,Tile>& a) {
    return a;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline TiledArray::Array<T,DIM,Tile> copy(const TiledArray::Array<T,DIM,Tile>& a) {
    return a;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void zero(TiledArray::Array<T,DIM,Tile>& a) {
    a = typename TiledArray::Array<T,DIM,Tile>::element_type(0) * a(detail::dummy_annotation(DIM));
  }

  template <typename T, unsigned int DIM, typename Tile>
  typename TiledArray::Array<T,DIM,Tile>::element_type
  inline minabs_value(const TiledArray::Array<T,DIM,Tile>& a) {
    return minabs(a(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline typename TiledArray::Array<T,DIM,Tile>::element_type
  maxabs_value(const TiledArray::Array<T,DIM,Tile>& a) {
    return norminf(a(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void vec_multiply(TiledArray::Array<T,DIM,Tile>& a1, const TiledArray::Array<T,DIM,Tile>& a2) {
    a1 = multiply(a1(detail::dummy_annotation(DIM)), a2(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline typename TiledArray::Array<T,DIM,Tile>::element_type
  dot_product(const TiledArray::Array<T,DIM,Tile>& a1, const TiledArray::Array<T,DIM,Tile>& a2) {
    return dot(a1(detail::dummy_annotation(DIM)), a2(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void scale(TiledArray::Array<T,DIM,Tile>& a,
                    typename TiledArray::Array<T,DIM,Tile>::element_type scaling_factor) {
    a = scaling_factor * a(detail::dummy_annotation(DIM));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void axpy(TiledArray::Array<T,DIM,Tile>& y,
                   typename TiledArray::Array<T,DIM,Tile>::element_type a,
                   const TiledArray::Array<T,DIM,Tile>& x) {
    y = y(detail::dummy_annotation(DIM)) + a * x(detail::dummy_annotation(DIM));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void assign(TiledArray::Array<T,DIM,Tile>& m1, const TiledArray::Array<T,DIM,Tile>& m2) {
    m1 = m2;
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline typename TiledArray::Array<T,DIM,Tile>::element_type
  norm2(const TiledArray::Array<T,DIM,Tile>& a) {
    return norm2(a(detail::dummy_annotation(DIM)));
  }

  template <typename T, unsigned int DIM, typename Tile>
  inline void print(const TiledArray::Array<T,DIM,Tile>& a, const char* label) {
    std::cout << label << ":" << std::endl
              << a << std::endl;
  }

};

#endif // TILEDARRAY_ALGEBRA_UTILS_H__INCLUDED
