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
 *  util.h
 *  Created:    19 June, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_SCALAPACK_UTIL_H__INCLUDED
#define TILEDARRAY_ALGEBRA_SCALAPACK_UTIL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/algebra/types.h>
#include <TiledArray/conversions/block_cyclic.h>

namespace TiledArray {

namespace scalapack {

inline scalapackpp::TransposeFlag to_scalapackpp_transposeflag(
    TransposeFlag t) {
  switch (t) {
    case TransposeFlag::NoTranspose:
      return scalapackpp::TransposeFlag::NoTranspose;
    case TransposeFlag::Transpose:
      return scalapackpp::TransposeFlag::Transpose;
    case TransposeFlag::ConjTranspose:
      return scalapackpp::TransposeFlag::ConjTranspose;
    default:
      abort();
  }
}

template <typename T>
void zero_triangle(blacspp::Triangle tri, scalapack::BlockCyclicMatrix<T>& A,
                   bool zero_diag = false) {
  auto zero_el = [&](size_t I, size_t J) {
    if (A.dist().i_own(I, J)) {
      auto [i, j] = A.dist().local_indx(I, J);
      A.local_mat()(i, j) = 0.;
    }
  };

  auto [M, N] = A.dims();

  // Zero the lower triangle
  if (tri == blacspp::Triangle::Lower) {
    if (zero_diag)
      for (size_t j = 0; j < N; ++j)
        for (size_t i = j; i < M; ++i) zero_el(i, j);
    else
      for (size_t j = 0; j < N; ++j)
        for (size_t i = j + 1; i < M; ++i) zero_el(i, j);

    // Zero the upper triangle
  } else {
    if (zero_diag)
      for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i <= std::min(j, M); ++i) zero_el(i, j);
    else
      for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < std::min(j, M); ++i) zero_el(i, j);
  }
}

namespace detail {
inline std::size_t& default_block_size_accessor() {
  static std::size_t block_size = 128;
  return block_size;
}
}  // namespace detail

inline std::size_t default_block_size() {
  return detail::default_block_size_accessor();
}

inline void set_default_block_size(std::size_t NB) {
  TA_ASSERT(NB > 0);
  detail::default_block_size_accessor() = NB;
}

}  // namespace scalapack
}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_ALGEBRA_SCALAPACK_UTIL_H__INCLUDED
