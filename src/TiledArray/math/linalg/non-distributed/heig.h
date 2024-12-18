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
 *  heig.h
 *  Created:  19 October,  2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_HEIG_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_HEIG_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/conversions/eigen.h>
#include <TiledArray/math/linalg/rank-local.h>
#include <TiledArray/math/linalg/util.h>

namespace TiledArray::math::linalg::non_distributed {

/**
 *  @brief Solve the standard eigenvalue problem with LAPACK
 *
 *  A(i,k) X(k,j) = X(i,j) E(j)
 *
 *  Example Usage:
 *
 *  auto [E, X] = heig(A, ...)
 *
 *  @tparam Array Input array type
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] evec_trange TiledRange for resulting eigenvectors. If left empty,
 *                         will default to array.trange()
 *
 *  @returns A tuple containing the eigenvalues and eigenvectors of input array
 *  as std::vector and in TA format, respectively.
 */
template <typename Array>
auto heig(const Array& A, TiledRange evec_trange = TiledRange()) {
  using scalar_type = typename detail::array_traits<Array>::scalar_type;
  World& world = A.world();
  auto A_eig = detail::make_matrix(A);
  std::vector<scalar_type> evals;
  TA_LAPACK_ON_RANK_ZERO(heig, world, A_eig, evals);
  world.gop.broadcast_serializable(A_eig, 0);
  world.gop.broadcast_serializable(evals, 0);
  if (evec_trange.rank() == 0) evec_trange = A.trange();
  return std::tuple(evals, eigen_to_array<Array>(world, evec_trange, A_eig));
}

/**
 *  @brief Solve the generalized eigenvalue problem with LAPACK
 *
 *  A(i,k) X(k,j) = B(i,k) X(k,j) E(j)
 *
 *  with
 *
 *  X(k,i) B(k,l) X(l,j) = I(i,j)
 *
 *  Example Usage:
 *
 *  auto [E, X] = heig(A, B, ...)
 *
 *  @tparam ArrayA the type of @p A, i.e., an array type
 *  @tparam ArrayB the type of @p B, i.e., an array type
 *  @tparam EVecType an array type to use for returning the eigenvectors
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] B           Positive-definite matrix
 *  @param[in] evec_trange TiledRange for resulting eigenvectors. If left empty,
 *                         will default to array.trange()
 *
 *  @returns A tuple containing the eigenvalues and eigenvectors of input array
 *  as std::vector and in TA format, respectively.
 */
template <typename ArrayA, typename ArrayB, typename EVecType = ArrayA>
auto heig(const ArrayA& A, const ArrayB& B,
          TiledRange evec_trange = TiledRange()) {
  using scalar_type = typename detail::array_traits<ArrayA>::scalar_type;
  (void)detail::array_traits<ArrayB>{};
  World& world = A.world();
  auto A_eig = detail::make_matrix(A);
  auto B_eig = detail::make_matrix(B);
  std::vector<scalar_type> evals;
  TA_LAPACK_ON_RANK_ZERO(heig, world, A_eig, B_eig, evals);
  world.gop.broadcast_serializable(A_eig, 0);
  world.gop.broadcast_serializable(evals, 0);
  if (evec_trange.rank() == 0) evec_trange = A.trange();
  return std::tuple(evals,
                    eigen_to_array<ArrayA>(A.world(), evec_trange, A_eig));
}

}  // namespace TiledArray::math::linalg::non_distributed

#endif  // TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_HEIG_H__INCLUDED
