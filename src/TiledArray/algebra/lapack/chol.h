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
 *  chol.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_LAPACK_CHOL_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LAPACK_CHOL_H__INCLUDED

#include <TiledArray/config.h>

namespace TiledArray {
namespace lapack {

/**
 *  @brief Compute the Cholesky factorization of a HPD rank-2 tensor
 *
 *  A(i,j) = L(i,k) * conj(L(j,k))
 *
 *  Example Usage:
 *
 *  auto L = cholesky(A, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] l_trange    TiledRange for resulting Cholesky factor. If left
 * empty, will default to array.trange()
 *
 *  @returns The lower triangular Cholesky factor L in TA format
 */
template <typename Array>
auto cholesky(const Array& A, TiledRange l_trange = TiledRange()) {
  auto& world = A.world();

  //    // Call lapack verson of LLT dpotrf_, we have to reverse stuff since
  //    lapack
  //    // will think all of our matrices are Col Major
  //    // RowMatrixXd A_copy = A;
  //
  //    char uplo = 'U';  // Do lower, but need to use U because Row -> Col
  //    integer n = A.rows();
  //    real8* a = A.data();
  //    integer lda = n;
  //    integer info;
  //
  //#ifdef MADNESS_LINALG_USE_LAPACKE
  //    dpotrf_(&uplo, &n, a, &lda, &info);
  //#else
  //    dpotrf_(&uplo, &n, a, &lda, &info, sizeof(char));
  //#endif
  //
  //  return L;
  abort();
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
 *  auto [L,Linv] = cholesky_Linv<decltype(A),true>(A, ...)
 *
 *  @tparam Array Input array type, must be convertible to BlockCyclicMatrix
 *  @tparam RetL  Whether or not to return the cholesky factor
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] l_trange    TiledRange for resulting inverse Cholesky factor.
 *                         If left empty, will default to array.trange()
 *
 *  @returns The inverse lower triangular Cholesky factor in TA format
 */
template <typename Array, bool RetL = false>
auto cholesky_linv(const Array& A, TiledRange l_trange = TiledRange()) {
  abort();
}

template <typename Array>
auto cholesky_solve(const Array& A, const Array& B,
                    TiledRange x_trange = TiledRange()) {
  abort();
}

template <typename Array>
auto cholesky_lsolve(TransposeFlag transpose, const Array& A, const Array& B,
                     TiledRange l_trange = TiledRange(),
                     TiledRange x_trange = TiledRange()) {
  abort();
}

}  // namespace lapack
}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_LAPACK_H__INCLUDED
