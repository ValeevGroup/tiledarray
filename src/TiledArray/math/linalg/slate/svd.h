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
 *  cholesky.h
 *  Created:    24 July, 2023
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SLATE_SVD_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_SVD_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/util.h>
#include <TiledArray/math/linalg/forward.h>

namespace TiledArray::math::linalg::slate {

template <SVD::Vectors Vectors, typename Array>
auto svd(const Array& A, TA::TiledRange u_trange, TA::TiledRange vt_trange) {

  constexpr bool need_uv = (Vectors == SVD::AllVectors);
  constexpr bool need_u = (Vectors == SVD::LeftVectors) or need_uv;
  constexpr bool need_vt = (Vectors == SVD::RightVectors) or need_uv;
  constexpr bool vals_only = not need_u and not need_vt;

  //static_assert(vals_only, "SLATE + SVD Vectors NYI");
  std::cout << "IN SLATE SVD" << std::endl;

  using element_type   = typename std::remove_cv_t<Array>::element_type;
  auto& world = A.world();
  auto comm   = world.mpi.comm().Get_mpi_comm();

  // Convert to SLATE
  auto matrix = array_to_slate(A);
  using slate_matrix_t = std::decay_t<decltype(matrix)>;

  // Allocate space for singular values
  const auto M = matrix.m();
  const auto N = matrix.n();
  const auto SVD_SIZE = std::min(M,N);
  std::vector<::blas::real_type<element_type>> S(SVD_SIZE);

  // Perform GESVD 
  world.gop.fence();  // stage SLATE execution

  SlateFunctors u_functors(u_trange, A.pmap());
  SlateFunctors vt_functors(vt_trange, A.pmap());

  auto& u_tileMb = u_functors.tileMb();
  auto& u_tileNb = u_functors.tileNb();
  auto& u_tileRank = u_functors.tileRank();
  auto& u_tileDevice = u_functors.tileDevice();

  auto& vt_tileMb = vt_functors.tileMb();
  auto& vt_tileNb = vt_functors.tileNb();
  auto& vt_tileRank = vt_functors.tileRank();
  auto& vt_tileDevice = vt_functors.tileDevice();

  slate_matrix_t U, VT;

  // Allocate if required 
  if(need_u) {
    U = slate_matrix_t(M, SVD_SIZE, u_tileMb, u_tileNb, u_tileRank, u_tileDevice, comm);
    U.insertLocalTiles();
  }
  if(need_vt) {
    VT = slate_matrix_t(SVD_SIZE, N, vt_tileMb, vt_tileNb, vt_tileRank, vt_tileDevice, comm);
    VT.insertLocalTiles();
  }

  // Do SVD
  ::slate::svd(matrix, S, U, VT); 

  Array U_ta, VT_ta;
  if(need_u)  { U_ta  = slate_to_array<Array>(U,  world); }
  if(need_vt) { VT_ta = slate_to_array<Array>(VT, world); } 

  if constexpr (need_uv) {
    return std::tuple(S, U_ta, VT_ta);
  } else if constexpr (need_u) {
    return std::tuple(S, U_ta);
  } else if constexpr (need_vt) {
    return std::tuple(S, VT_ta);
  } else { 
    return S;
  }

}

} // namespace TiledArray::math::linalg::slate

#endif // TILEDARRAY_HAS_SLATE

#endif // TILEDARRAY_MATH_LINALG_SLATE_SVD_H__INCLUDED
