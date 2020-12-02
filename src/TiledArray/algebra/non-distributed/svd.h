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
#ifndef TILEDARRAY_ALGEBRA_NON_DISTRIBUTED_SVD_H__INCLUDED
#define TILEDARRAY_ALGEBRA_NON_DISTRIBUTED_SVD_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/algebra/non-distributed/util.h>
#include <TiledArray/algebra/rank-local.h>
#include <TiledArray/conversions/eigen.h>

namespace TiledArray::non_distributed {

/**
 *  @brief Compute the singular value decomposition (SVD) via ScaLAPACK
 *
 *  A(i,j) = S(k) U(i,k) conj(V(j,k))
 *
 *  Example Usage:
 *
 *  auto S          = svd<SVDValuesOnly>  (A, ...)
 *  auto [S, U]     = svd<SVDLeftstd::vectors> (A, ...)
 *  auto [S, VT]    = svd<SVDRightstd::vectors>(A, ...)
 *  auto [S, U, VT] = svd<SVDAllstd::vectors>  (A, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *
 *  @param[in] A           Input array to be decomposed. Must be rank-2
 *  @param[in] u_trange    TiledRange for resulting left singular vectors.
 *  @param[in] vt_trange   TiledRange for resulting right singular vectors
 * (transposed).
 *
 *  @returns A tuple containing the eigenvalues and eigenvectors of input array
 *  as std::vector and in TA format, respectively.
 */
template <typename SVDType, typename Array,
          typename = TiledArray::detail::enable_if_svd_return_type<SVDType>>
auto svd(const Array& A, TiledRange u_trange = TiledRange(), TiledRange vt_trange = TiledRange()) {

  using T = typename Array::numeric_type;
  using Matrix = algebra::rank_local::Matrix<T>;

  World& world = A.world();
  auto A_eig = detail::to_eigen(A);

  constexpr bool svd_all_vectors = std::is_same_v<SVDType, SVDAllVectors>;
  constexpr bool need_u = std::is_same_v<SVDType, SVDLeftVectors> or svd_all_vectors;
  constexpr bool need_vt = std::is_same_v<SVDType, SVDRightVectors> or svd_all_vectors;

  std::vector<T> S;
  std::unique_ptr<Matrix> U, VT;

  if constexpr (need_u) U = std::make_unique<Matrix>();
  if constexpr (need_vt) VT = std::make_unique<Matrix>();

  if (world.rank() == 0) {
    algebra::rank_local::svd(A_eig, S, U.get(), VT.get());
  }

  world.gop.broadcast_serializable(S, 0);
  if (U) world.gop.broadcast_serializable(*U, 0);
  if (VT) world.gop.broadcast_serializable(*VT, 0);

  auto make_array = [&world](auto && ... args) {
    return eigen_to_array<Array>(world, args...);
  };

  if constexpr (need_u && need_vt) {
    return std::tuple(S, make_array(u_trange, *U), make_array(vt_trange, *VT));
  }
  if constexpr (need_u && !need_vt) {
    return std::tuple(S, make_array(u_trange, *U));
  }
  if constexpr (!need_u && need_vt) {
    return std::tuple(S, make_array(vt_trange, *VT));
  }

  if constexpr (!need_u && !need_vt) return S;

}

}  // namespace TiledArray::non_distributed

#endif  // TILEDARRAY_ALGEBRA_NON_DISTRIBUTED_SVD_H__INCLUDED
