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
 *  lapack.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_LAPACK_LAPACK_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LAPACK_LAPACK_H__INCLUDED

#include <TiledArray/algebra/types.h>
#include <TiledArray/config.h>
#include <madness/tensor/clapack.h>
#include <Eigen/Core>

/// TA_LAPACK_CALL(fn,args) can be called only from template context, with `T`
/// defining the element type
#define TA_LAPACK_CALL(name, args...)                                    \
  using numeric_type = T;                                                \
  if constexpr (std::is_same_v<numeric_type, double>)                    \
    d##name##_(args);                                                    \
  else if constexpr (std::is_same_v<numeric_type, float>)                \
    s##name##_(args);                                                    \
  else if constexpr (std::is_same_v<numeric_type, std::complex<double>>) \
    z##name##_(args);                                                    \
  else if constexpr (std::is_same_v<numeric_type, std::complex<float>>)  \
    c##name##_(args);                                                    \
  else                                                                   \
    std::abort();

#define TA_LAPACK_GESV(...) TA_LAPACK_CALL(gesv, __VA_ARGS__)
#define TA_LAPACK_GETRF(...) TA_LAPACK_CALL(getrf, __VA_ARGS__)
#define TA_LAPACK_GETRI(...) TA_LAPACK_CALL(getri, __VA_ARGS__)

#ifdef MADNESS_LINALG_USE_LAPACKE

#define TA_LAPACK_POTRF(...) TA_LAPACK_CALL(potrf, __VA_ARGS__)
#define TA_LAPACK_POTRS(...) TA_LAPACK_CALL(potrs, __VA_ARGS__)
#define TA_LAPACK_POSV(...) TA_LAPACK_CALL(posv, __VA_ARGS__)
#define TA_LAPACK_GESVD(...) TA_LAPACK_CALL(gesvd, __VA_ARGS__)
#define TA_LAPACK_TRTRI(...) TA_LAPACK_CALL(trtri, __VA_ARGS__)
#define TA_LAPACK_TRTRS(...) TA_LAPACK_CALL(trtrs, __VA_ARGS__)
#define TA_LAPACK_SYEV(...) TA_LAPACK_CALL(syev, __VA_ARGS__)
#define TA_LAPACK_SYEVR(...) TA_LAPACK_CALL(syevr, __VA_ARGS__)
#define TA_LAPACK_SYGV(...) TA_LAPACK_CALL(sygv, __VA_ARGS__)

#else

#ifdef FORTRAN_LINKAGE_LCU
#define dtrtrs dtrtrs_
#define dposv dposv_
#define dpotrs dpotrs_
#define dsyevr dsyevr_
#endif

extern "C" {  // these arent in madness/clapack_fortran.h
void dtrtrs(const char* uplo, const char* trans, const char* diag,
            const integer* n, const integer* nrhs, const real8* a,
            const integer* lda, const real8* b, const integer* ldb,
            integer* info, char_len, char_len, char_len);
void dposv(const char* uplo, const integer* n, const integer* nrhs,
           const real8* a, const integer* lda, const real8* b,
           const integer* ldb, integer* info, char_len);
void dpotrs(char* uplo, integer* n, integer* nrhs, const real8* a, integer* lda,
            real8* b, integer* ldb, integer* info, char_len);
void dsyevr(char* jobz, char* range, char* uplo, integer* n, real8* a,
            integer* lda, real8* vl, real8* vu, integer* il, integer* iu,
            real8* abstol, integer* m, real8* w, real8* z, integer* ldz,
            integer* isuppz, real8* work, integer* lwork, integer* iwork,
            integer* liwork, integer* info, char_len, char_len, char_len);
}

#define TA_LAPACK_POTRF(...) TA_LAPACK_CALL(potrf, __VA_ARGS__, sizeof(char))
#define TA_LAPACK_POTRS(...) TA_LAPACK_CALL(potrs, __VA_ARGS__, sizeof(char))
#define TA_LAPACK_POSV(...) TA_LAPACK_CALL(posv, __VA_ARGS__, sizeof(char))
#define TA_LAPACK_GESVD(...) \
  TA_LAPACK_CALL(gesvd, __VA_ARGS__, sizeof(char), sizeof(char))
#define TA_LAPACK_TRTRI(...) \
  TA_LAPACK_CALL(trtri, __VA_ARGS__, sizeof(char), sizeof(char))
#define TA_LAPACK_TRTRS(...) \
  TA_LAPACK_CALL(trtrs, __VA_ARGS__, sizeof(char), sizeof(char), sizeof(char))
#define TA_LAPACK_SYEV(...) \
  TA_LAPACK_CALL(syev, __VA_ARGS__, sizeof(char), sizeof(char))
#define TA_LAPACK_SYEVR(...) \
  TA_LAPACK_CALL(syevr, __VA_ARGS__, sizeof(char), sizeof(char), sizeof(char))
#define TA_LAPACK_SYGV(...) \
  TA_LAPACK_CALL(sygv, __VA_ARGS__, sizeof(char), sizeof(char))

#endif  // MADNESS_LINALG_USE_LAPACKE

namespace TiledArray::lapack {

template <typename A>
struct array_traits {
  using scalar_type = typename A::scalar_type;
  using numeric_type = typename A::numeric_type;
  static const bool complex = !std::is_same_v<scalar_type, numeric_type>;
  static_assert(std::is_same_v<numeric_type, typename A::element_type>,
                "TA::lapack is only usable with a DistArray of scalar types");
};

template <typename T>
using Matrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
void cholesky(Matrix<T> &A);

template <typename T>
void cholesky_linv(Matrix<T> &A);

template <typename T>
void cholesky_solve(Matrix<T> &A, Matrix<T> &X);

template <typename T>
void cholesky_lsolve(TransposeFlag transpose, Matrix<T> &A, Matrix<T> &X);

template <typename T>
void heig(Matrix<T> &A, std::vector<T> &W);

template <typename T>
void heig(Matrix<T> &A, Matrix<T> &B, std::vector<T> &W);

template <typename T>
void svd(Matrix<T> &A, std::vector<T> &S, Matrix<T> *U, Matrix<T> *VT);

template <typename T>
void lu_solve(Matrix<T> &A, Matrix<T> &B);

template <typename T>
void lu_inv(Matrix<T> &A);

}  // namespace TiledArray::lapack

#endif  // TILEDARRAY_ALGEBRA_LAPACK_LAPACK_H__INCLUDED
