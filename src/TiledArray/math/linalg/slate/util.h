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
 *  util.h
 *  Created:    25 July, 2023
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SLATE_UTIL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_UTIL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <slate/slate.hh>
namespace TiledArray::math::linalg::slate {

template <typename SlateMatrixType>
void zero_triangle(::slate::Uplo tri, SlateMatrixType& A, bool zero_diag = false ) {

  const auto nt = A.nt(); // Number of column tiles
  const auto mt = A.mt(); // Number of row tiles

  auto zero_block = [&](auto it, auto jt) {
    if( A.tileIsLocal(it,jt) ) {
      auto tile = A(it,jt);
      const auto stride = tile.stride();
      const auto mb = tile.mb();
      const auto nb = tile.nb();
      auto* data = tile.data();
      for(int j = 0; j < nb; ++j)
      for(int i = 0; i < mb; ++i) {
        data[i + j*stride] = 0.0;
      }
    }
  };

  auto zero_tri = [&](auto it, auto jt) {
    if( A.tileIsLocal(it,jt) ) {
      auto tile = A(it,jt);
      const auto stride = tile.stride();
      const auto mb = tile.mb();
      const auto nb = tile.nb();
      auto* data = tile.data();
      if( tri == ::slate::Uplo::Lower ) {
        for(int j = 0;   j < nb; ++j)
        for(int i = j+1; i < mb; ++i) {
          data[i + j*stride] = 0.0;
        }
      } else {
        for(int j = 0; j < nb; ++j)
        for(int i = 0; i < j;  ++i) {
          data[i + j*stride] = 0.0;
        }
      }
    }
  };

  // TODO: Should be done in parallel
  for(auto jt = 0; jt < nt; ++jt) {
    zero_tri(jt, jt); // Handle diagonal block 
    for(auto it = jt + 1; it < mt; ++it) {
      if( tri == ::slate::Uplo::Lower ) zero_block(it,jt);
      else                              zero_block(jt,it);
    }
  }

}

} // namespace TiledArray::math::linalg::slate

#endif // TILEDARRAY_HAS_SLATE

#endif // TILEDARRAY_MATH_LINALG_SLATE_UTIL_H__INCLUDED
