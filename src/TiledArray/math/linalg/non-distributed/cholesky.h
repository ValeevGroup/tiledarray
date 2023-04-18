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
 *  cholesky.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_CHOL_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/conversions/eigen.h>
#include <TiledArray/math/linalg/rank-local.h>
#include <TiledArray/math/linalg/util.h>

namespace TiledArray::math::linalg::non_distributed {

template <typename Tile, typename Policy>
auto rank_local_cholesky(const DistArray<Tile, Policy>& A) {
  using Array = DistArray<Tile, Policy>;
  using numeric_type = typename Array::numeric_type;
  static_assert(std::is_same_v<numeric_type, typename Array::element_type>,
                "TA::lapack::{cholesky*} are only usable with a DistArray of "
                "scalar types");

  World& world = A.world();
  auto A_eig = detail::make_matrix(A);
  TA_LAPACK_ON_RANK_ZERO(cholesky, world, A_eig);
  world.gop.broadcast_serializable(A_eig, 0);
  return A_eig;
}

/**
 *  @brief Compute the Cholesky factorization of a HPD rank-2 tensor
 *
 *  A(i,j) = L(i,k) * conj(L(j,k))
 *
 *  Example Usage:
 *
 *  auto L = cholesky(A, ...)
 *
 *  @tparam Array a DistArray type (i.e., @c is_array_v<Array> is true)
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] l_trange    TiledRange for resulting Cholesky factor. If left
 * empty, will default to array.trange()
 *
 *  @returns The lower triangular Cholesky factor L in TA format
 *  @note this is a collective operation with respect to the world of @p A
 *  @throw TiledArray::Exception if A is not HPD
 */
template <typename Array,
          typename = std::enable_if_t<TiledArray::detail::is_array_v<Array>>>
auto cholesky(const Array& A, TiledRange l_trange = TiledRange()) {
  auto L_eig = rank_local_cholesky(A);
  detail::zero_out_upper_triangle(L_eig);
  if (l_trange.rank() == 0) l_trange = A.trange();
  return eigen_to_array<Array>(A.world(), l_trange, L_eig);
}

/**
 *  @brief Compute the Cholesky factorization of a HPD rank-2 tensor
 *
 *  A(i,j) = L(i,k) * conj(L(j,k))
 *
 *  Example Usage:
 *
 *  auto L = cholesky(A, ...)
 *
 *  @tparam ContiguousTensor a contiguous tensor type (i.e., @c
 * is_contiguous_tensor_v<ContiguousTensor> is true)
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @return The lower triangular Cholesky factor L as a DistArray
 *  @note this is a non-collective operation, only computes on the rank on which
 * invoked
 *  @throw TiledArray::Exception if A is not HPD
 */
template <typename ContiguousTensor,
          typename = std::enable_if_t<
              TiledArray::detail::is_contiguous_tensor_v<ContiguousTensor>>>
auto cholesky(const ContiguousTensor& A) {
  auto A_eig = detail::make_matrix(A);
  linalg::rank_local::cholesky(A_eig);
  detail::zero_out_upper_triangle(A_eig);
  return detail::make_array<ContiguousTensor>(A_eig, A.range());
}

/**
 *  @brief Compute the inverse of the Cholesky factor of an HPD rank-2 tensor.
 *  Optionally return the Cholesky factor itself
 *
 *  A(i,j) = L(i,k) * conj(L(j,k)) -> compute Linv
 *
 *  Example Usage:
 *
 *  auto Linv     = cholesky_Linv(A, ...)
 *  auto [L,Linv] = cholesky_Linv<true>(A, ...)
 *
 *  @tparam Both  Whether or not to return the Cholesky factor
 *  @tparam Array the type of `A`, a DistArray type
 *          (i.e., @c is_array_v<Array> is true)
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] l_trange    TiledRange for resulting inverse Cholesky factor.
 *                         If left empty, will default to array.trange()
 *
 *  @returns The inverse of the lower-triangular Cholesky
 *           factor (i.e., only the lower triangle elements are nonzero)
 *           as a DistArray
 *  @note this is a collective operation with respect to the world of @p A
 *  @throw TiledArray::Exception if A is not HPD
 */
template <bool Both, typename Array,
          typename = std::enable_if_t<TiledArray::detail::is_array_v<Array>>>
auto cholesky_linv(const Array& A, TiledRange l_trange = TiledRange()) {
  World& world = A.world();
  auto L_eig = rank_local_cholesky(A);
  if constexpr (Both) detail::zero_out_upper_triangle(L_eig);

  // if need to return L use its copy to compute inverse
  decltype(L_eig) L_inv_eig;

  std::optional<lapack::Error> error_opt;
  if (world.rank() == 0) {
    try {
      if (Both) L_inv_eig = L_eig;
      auto& L_inv_eig_ref = Both ? L_inv_eig : L_eig;
      linalg::rank_local::cholesky_linv(L_inv_eig_ref);
      detail::zero_out_upper_triangle(L_inv_eig_ref);
    } catch (lapack::Error& err) {
      error_opt = err;
    }
  }
  world.gop.broadcast_serializable(error_opt, 0);
  if (error_opt) {
    throw error_opt.value();
  }
  world.gop.broadcast_serializable(Both ? L_inv_eig : L_eig, 0);

  if (l_trange.rank() == 0) l_trange = A.trange();
  if constexpr (Both)
    return std::make_tuple(eigen_to_array<Array>(world, l_trange, L_eig),
                           eigen_to_array<Array>(world, l_trange, L_inv_eig));
  else
    return eigen_to_array<Array>(world, l_trange, L_eig);
  abort();  // unreachable
}

template <typename Array,
          typename = std::enable_if_t<TiledArray::detail::is_array_v<Array>>>
auto cholesky_solve(const Array& A, const Array& B,
                    TiledRange x_trange = TiledRange()) {
  using numeric_type = typename Array::numeric_type;
  static_assert(std::is_same_v<numeric_type, typename Array::element_type>,
                "TA::lapack::{cholesky*} are only usable with a DistArray of "
                "scalar types");

  auto A_eig = detail::make_matrix(A);
  auto X_eig = detail::make_matrix(B);
  World& world = A.world();
  TA_LAPACK_ON_RANK_ZERO(cholesky_solve, world, A_eig, X_eig);
  world.gop.broadcast_serializable(X_eig, 0);
  if (x_trange.rank() == 0) x_trange = B.trange();
  return eigen_to_array<Array>(world, x_trange, X_eig);
}

template <typename Array,
          typename = std::enable_if_t<TiledArray::detail::is_array_v<Array>>>
auto cholesky_lsolve(Op transpose, const Array& A, const Array& B,
                     TiledRange l_trange = TiledRange(),
                     TiledRange x_trange = TiledRange()) {
  World& world = A.world();
  auto L_eig = rank_local_cholesky(A);
  detail::zero_out_upper_triangle(L_eig);

  using numeric_type = typename Array::numeric_type;
  static_assert(std::is_same_v<numeric_type, typename Array::element_type>,
                "TA::lapack::{cholesky*} are only usable with a DistArray of "
                "scalar types");

  auto X_eig = detail::make_matrix(B);
  TA_LAPACK_ON_RANK_ZERO(cholesky_lsolve, world, transpose, L_eig, X_eig);
  world.gop.broadcast_serializable(X_eig, 0);
  if (l_trange.rank() == 0) l_trange = A.trange();
  if (x_trange.rank() == 0) x_trange = B.trange();
  return std::make_tuple(eigen_to_array<Array>(world, l_trange, L_eig),
                         eigen_to_array<Array>(world, x_trange, X_eig));
}

}  // namespace TiledArray::math::linalg::non_distributed

#endif  // TILEDARRAY_MATH_LINALG_NON_DISTRIBUTED_CHOL_H__INCLUDED
