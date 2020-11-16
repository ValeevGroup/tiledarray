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
#ifndef TILEDARRAY_ALGEBRA_LAPACK_HEIG_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LAPACK_HEIG_H__INCLUDED

#include <TiledArray/algebra/lapack/util.h>
#include <TiledArray/config.h>
#include <TiledArray/conversions/eigen.h>

namespace TiledArray {
namespace lapack {

/**
 *  @brief Solve the standard eigenvalue problem with ScaLAPACK
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
  using scalar_type = typename Array::scalar_type;
  using numeric_type = typename Array::numeric_type;
  constexpr const bool is_real = std::is_same_v<scalar_type, numeric_type>;
  static_assert(std::is_same_v<numeric_type, typename Array::element_type>,
                "TA::lapack::{cholesky*} are only usable with a DistArray of "
                "scalar types");

  World& world = A.world();
  auto A_eig = detail::to_eigen(A);
  std::vector<scalar_type> evals;
//   if (world.rank() == 0) {
//     char jobz = 'V';
//     char uplo = 'L';
//     integer n = A_eig.rows();
//     numeric_type* a = A_eig.data();
//     integer lda = n;
//     integer info = 0;
//     evals.resize(n);
//     integer lwork = -1;
//     std::vector<numeric_type> work(1);
//     // run once to query, then to compute
//     while (lwork != static_cast<integer>(work.size())) {
//       if (lwork > 0) {
//         work.resize(lwork);
//       }
//       if constexpr (is_real) {
// #if defined(MADNESS_LINALG_USE_LAPACKE)
//         MADNESS_DISPATCH_LAPACK_FN(syev, &jobz, &uplo, &n, a, &lda,
//                                    evals.data(), work.data(), &lwork, &info);
// #else
//         MADNESS_DISPATCH_LAPACK_FN(syev, &jobz, &uplo, &n, a, &lda,
//                                    evals.data(), work.data(), &lwork, &info,
//                                    sizeof(char), sizeof(char));
// #endif
//       } else {
//         std::vector<scalar_type> rwork;
//         if (lwork == static_cast<integer>(work.size())) rwork.resize(3 * n - 2);
// #if defined(MADNESS_LINALG_USE_LAPACKE)
//         MADNESS_DISPATCH_LAPACK_FN(heev, &jobz, &uplo, &n, a, &lda,
//                                    evals.data(), work.data(), &lwork,
//                                    &rwork.data(), &info);
// #else
//         MADNESS_DISPATCH_LAPACK_FN(
//             heev, &jobz, &uplo, &n, a, &lda, evals.data(), work.data(), &lwork,
//             &rwork.data(), &info, sizeof(char), sizeof(char));
// #endif
//       }
//       if (lwork == -1) {
//         if constexpr (is_real) {
//           lwork = static_cast<integer>(work[0]);
//         } else {
//           lwork = static_cast<integer>(work[0].real());
//         }
//         TA_ASSERT(lwork > 1);
//       }
//     };

//     if (info != 0) {
//       if (is_real)
//         TA_EXCEPTION("LAPACK::syev failed");
//       else
//         TA_EXCEPTION("LAPACK::heev failed");
//     }
//   }

  world.gop.broadcast_serializable(A_eig, 0);
  world.gop.broadcast_serializable(evals, 0);
  if (evec_trange.rank() == 0) evec_trange = A.trange();
  return std::tuple(evals,
                    eigen_to_array<Array>(A.world(), evec_trange, A_eig));
}

/**
 *  @brief Solve the generalized eigenvalue problem with ScaLAPACK
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
 *  @tparam Array Input array type
 *
 *  @param[in] A           Input array to be diagonalized. Must be rank-2
 *  @param[in] B           Positive-definite matrix
 *  @param[in] evec_trange TiledRange for resulting eigenvectors. If left empty,
 *                         will default to array.trange()
 *  @param[in] NB          ScaLAPACK block size. Defaults to 128
 *
 *  @returns A tuple containing the eigenvalues and eigenvectors of input array
 *  as std::vector and in TA format, respectively.
 */
template <typename ArrayA, typename ArrayB, typename EVecType = ArrayA>
auto heig(const ArrayA& A, const ArrayB& B,
          TiledRange evec_trange = TiledRange()) {
  using scalar_type = typename ArrayA::scalar_type;
  using numeric_type = typename ArrayA::numeric_type;
  constexpr const bool is_real = std::is_same_v<scalar_type, numeric_type>;
  static_assert(std::is_same_v<numeric_type, typename ArrayA::element_type>,
                "TA::lapack::{cholesky*} are only usable with a DistArray of "
                "scalar types");

  abort();
  return std::tuple(std::vector<scalar_type>{}, EVecType{});
}

}  // namespace lapack
}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_SCALAPACK_HEIG_H__INCLUDED
