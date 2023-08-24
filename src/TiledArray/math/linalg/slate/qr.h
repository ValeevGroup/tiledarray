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
 *  qr.h
 *  Created:    2 August, 2023
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_SLATE_QR_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_QR_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/util.h>

namespace TiledArray::math::linalg::slate {

template <bool QOnly, typename Array>
auto householder_qr( const Array& V, TiledRange q_trange = TiledRange(),
                     TiledRange r_trange = TiledRange() ) {

  if(q_trange.rank() == 0) {
    q_trange = V.trange();
  }

  if(r_trange.rank() == 0) {
    auto col_tiling = V.trange().dim(1);
    r_trange = TiledRange( {col_tiling, col_tiling} );
  }

  // SLATE does not yet have ORGQR/UNGQR
  // https://github.com/icl-utk-edu/slate/issues/80

  using element_type   = typename std::remove_cv_t<Array>::element_type;
  auto& world = V.world();

  // Convert to SLATE
  auto matrix = array_to_slate(V);

  // Perform GETRF
  ::slate::TriangularFactors<element_type> T;
  ::slate::geqrf(matrix, T);

  // Form Q
  auto Q = matrix.emptyLike(); Q.insertLocalTiles();
  ::slate::set(0.0, 1.0, Q);
  ::slate::unmqr(::slate::Side::Left, ::slate::Op::NoTrans, matrix, T, Q);

  auto Q_ta = slate_to_array<Array>(Q, world);

  if constexpr (QOnly) {
    return Q_ta;
  } else {
    SlateFunctors r_functors( r_trange, V.pmap() );
    const auto N = V.trange().dim(1).extent();
    auto comm = world.mpi.comm().Get_mpi_comm();
    auto R = r_functors.make_matrix<::slate::Matrix<element_type>>(N,N,comm);
    R.insertLocalTiles();
    ::slate::set(0.0, 0.0, R);

    // Triangular views of target operand matrices
    ::slate::TriangularMatrix<element_type> 
      R_tri(::slate::Uplo::Upper, ::slate::Diag::NonUnit, R);
    ::slate::TriangularMatrix<element_type> 
      A_tri(::slate::Uplo::Upper, ::slate::Diag::NonUnit, matrix);

    // Copy upper triangle of QR factors into R
    ::slate::copy(A_tri, R_tri);

    // Convert to TA
    auto R_ta = slate_to_array<Array>(R, world);
    return std::tuple(Q_ta, R_ta);
  }

}

} // namespace TiledArray::math::linalg::slate

#endif // TILEDARRAY_HAS_SLATE

#endif // TILEDARRAY_MATH_LINALG_SLATE_QR_H__INCLUDED
