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
#ifndef TILEDARRAY_MATH_LINALG_SLATE_CHOL_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SLATE_CHOL_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SLATE

#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/util.h>
namespace TiledArray::math::linalg::slate {

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
 *  @param[in] A Input array to be factorized. Must be rank-2
 *
 *  @returns The lower triangular Cholesky factor L in TA format
 */
template <typename Array>
auto cholesky(const Array& A) {

  using element_type   = typename std::remove_cv_t<Array>::element_type;
  auto& world = A.world();
  // Convert to SLATE
  auto matrix = array_to_slate(A);

  // Perform POTRF
  world.gop.fence();  // stage SLATE execution
  ::slate::HermitianMatrix<element_type> AH(::slate::Uplo::Lower, matrix);
  ::slate::potrf(AH);
  zero_triangle(::slate::Uplo::Upper, matrix);
  world.gop.fence();  // stage SLATE execution

  // Convert L to TA and return 
  return slate_to_array<Array>(matrix, world);

}



template <bool Both, typename Array>
auto cholesky_linv(const Array& A) {

  using element_type   = typename std::remove_cv_t<Array>::element_type;
  auto& world = A.world();
  auto matrix = array_to_slate(A);

  // Perform POTRF
  world.gop.fence();  // stage SLATE execution
  ::slate::HermitianMatrix<element_type> AH(::slate::Uplo::Lower, matrix);
  ::slate::potrf(AH);
  zero_triangle(::slate::Uplo::Upper, matrix);

  // Copy L if needed
  using matrix_type = std::decay_t<decltype(matrix)>;
  std::shared_ptr<Array> L_ptr = nullptr;
  if constexpr (Both) {
    L_ptr = std::make_shared<Array>(slate_to_array<Array>(matrix,world));
    world.gop.fence();  // Make sure copy is done before inverting L 
  }

  // Perform TRTRI
  ::slate::TriangularMatrix<element_type> L_slate(::slate::Uplo::Lower, 
    ::slate::Diag::NonUnit, matrix);
  ::slate::trtri(L_slate);

  // Convert Linv to TA
  auto Linv = slate_to_array<Array>(matrix, world);
  world.gop.fence();  // Make sure copy is done before return

  // Return Linv or L + Linv (in that order)
  if constexpr (Both) {
    return std::make_tuple( *L_ptr, Linv );
  } else {
    return Linv;
  }

}




template <typename AArray, typename BArray>
auto cholesky_solve(const AArray& A, const BArray& B) {

  using element_type   = typename std::remove_cv_t<AArray>::element_type;
  auto& world = A.world();
  /*
  if( world != B.world() ) {
    TA_EXCEPTION("A and B must be distributed on same MADWorld context");
  }
  */

  // Convert to SLATE
  auto A_slate = array_to_slate(A);
  auto B_slate = array_to_slate(B);

  // Solve linear system
  world.gop.fence();  // stage SLATE execution
  ::slate::HermitianMatrix<element_type> AH(::slate::Uplo::Lower, A_slate);
  ::slate::posv( AH, B_slate );

  // Convert solution to TA
  return slate_to_array<BArray>(B_slate, world);

}



template <typename AArray, typename BArray>
auto cholseky_lsolve(Op trans, const AArray& A, const BArray& B) {

  using element_type   = typename std::remove_cv_t<AArray>::element_type;
  auto& world = A.world();
  /*
  if( world != B.world() ) {
    TA_EXCEPTION("A and B must be distributed on same MADWorld context");
  }
  */

  // Convert to SLATE
  auto A_slate = array_to_slate(A);
  auto B_slate = array_to_slate(B);
  world.gop.fence();  // stage SLATE execution

  // Factorize A
  ::slate::HermitianMatrix<element_type> AH(::slate::Uplo::Lower, A_slate);
  ::slate::potrf(AH);

  // Solve linear system OP(L) * X = B
  ::slate::TriangularMatrix<element_type> L_slate(::slate::Uplo::Lower, 
    ::slate::Diag::NonUnit, A_slate);
  if( trans == Op::Trans )     L_slate = ::slate::transpose(L_slate);
  if( trans == Op::ConjTrans ) L_slate = ::slate::conj_transpose(L_slate);
  ::slate::trsm( ::slate::Side::Left, 1.0, L_slate, B_slate );
    
  // Zero out the upper triangle
  zero_triangle(::slate::Uplo::Upper, A_slate);
  
  // Convert solution and L to TA
  auto L = slate_to_array<AArray>(A_slate, world);
  auto X = slate_to_array<BArray>(B_slate, world);
  return std::make_tuple(L, X);

}

} // namespace TiledArray::math::linalg::slate

#endif // TILEDARRAY_HAS_SLATE

#endif // TILEDARRAY_MATH_LINALG_SLATE_CHOL_H__INCLUDED
