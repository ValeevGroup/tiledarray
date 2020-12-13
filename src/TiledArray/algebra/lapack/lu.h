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
 *  lu.h
 *  Created:    19 June, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_LAPACK_LU_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LAPACK_LU_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/algebra/lapack/lapack.h>

namespace TiledArray::lapack {

/**
 *  @brief Solve a linear system via LU factorization
 */
template <typename ArrayA, typename ArrayB>
auto lu_solve(const ArrayA& A, const ArrayB& B, TiledRange x_trange = TiledRange()) {
  (void)lapack::array_traits<ArrayA>{};
  (void)lapack::array_traits<ArrayB>{};
  auto& world = A.world();
  auto A_eig = detail::to_eigen(A);
  auto B_eig = detail::to_eigen(B);
  if (world.rank() == 0) {
    lapack::lu_solve(A_eig, B_eig);
  }
  world.gop.broadcast_serializable(B_eig, 0);
  if (x_trange.rank() == 0) x_trange = B.trange();
  return eigen_to_array<ArrayB>(world, x_trange, B_eig);
}

/**
 *  @brief Invert a matrix via LU
 */
template <typename Array>
auto lu_inv(const Array& A, TiledRange ainv_trange = TiledRange()) {
  (void)lapack::array_traits<Array>{};
  auto& world = A.world();
  auto A_eig = detail::to_eigen(A);
  if (world.rank() == 0) {
    lapack::lu_inv(A_eig);
  }
  world.gop.broadcast_serializable(A_eig, 0);
  if (ainv_trange.rank() == 0) ainv_trange = A.trange();
  return eigen_to_array<Array>(A.world(), ainv_trange, A_eig);
}

}  // namespace TiledArray::lapack

#endif  // TILEDARRAY_ALGEBRA_LAPACK_LU_H__INCLUDED
