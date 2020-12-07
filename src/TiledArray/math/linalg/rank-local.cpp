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
 *  lapack.cpp
 *  Created:    16 October, 2020
 *
 */

#include <TiledArray/math/linalg/rank-local.h>

#include <lapacke.h>

#define TA_LAPACKE_THROW(F) throw std::runtime_error("lapacke::" #F " failed")

#define TA_LAPACKE_CALL(F, ARGS...)         \
  (LAPACKE_##F(LAPACK_COL_MAJOR, ARGS) == 0 || (TA_LAPACKE_THROW(F), false))

/// TA_LAPACKE(fn,args) can be called only from template context, with `T`
/// defining the element type
#define TA_LAPACKE(name, args...) {                                            \
  using numeric_type = T;                                                      \
  if constexpr (std::is_same_v<numeric_type, double>)                          \
    TA_LAPACKE_CALL(d##name, args);                                            \
  else if constexpr (std::is_same_v<numeric_type, float>)                      \
    TA_LAPACKE_CALL(s##name, args);                                            \
  else if constexpr (std::is_same_v<numeric_type, std::complex<double>>)       \
    TA_LAPACKE_CALL(z##name, args);                                            \
  else if constexpr (std::is_same_v<numeric_type, std::complex<float>>)        \
    TA_LAPACKE_CALL(c##name, args);                                            \
  else std::abort();                                                           \
  }

namespace TiledArray::math::linalg::rank_local {

template <typename T>
void cholesky(Matrix<T>& A) {
  char uplo = 'L';
  lapack_int n = A.rows();
  auto* a = A.data();
  lapack_int lda = n;
  TA_LAPACKE(potrf, uplo, n, a, lda);
}

template <typename T>
void cholesky_linv(Matrix<T>& A) {
  char uplo = 'L';
  char diag = 'N';
  lapack_int n = A.rows();
  auto* l = A.data();
  lapack_int lda = n;
  TA_LAPACKE(trtri, uplo, diag, n, l, lda);
}

template <typename T>
void cholesky_solve(Matrix<T>& A, Matrix<T>& X) {
  char uplo = 'L';
  lapack_int n = A.rows();
  lapack_int nrhs = X.cols();
  auto* a = A.data();
  auto* b = X.data();
  lapack_int lda = n;
  lapack_int ldb = n;
  TA_LAPACKE(posv, uplo, n, nrhs, a, lda, b, ldb);
}

template <typename T>
void cholesky_lsolve(TransposeFlag transpose, Matrix<T>& A, Matrix<T>& X) {
  char uplo = 'L';
  char trans = transpose == TransposeFlag::Transpose
                   ? 'T'
                   : (transpose == TransposeFlag::NoTranspose ? 'N' : 'C');
  char diag = 'N';
  lapack_int n = A.rows();
  lapack_int nrhs = X.cols();
  auto* a = A.data();
  auto* b = X.data();
  lapack_int lda = n;
  lapack_int ldb = n;
  TA_LAPACKE(trtrs, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

template <typename T>
void heig(Matrix<T>& A, std::vector<T>& W) {
  char jobz = 'V';
  char uplo = 'L';
  lapack_int n = A.rows();
  T* a = A.data();
  lapack_int lda = A.rows();
  W.resize(n);
  T* w = W.data();
  lapack_int lwork = -1;
  std::vector<T> work(1);
  TA_LAPACKE(syev_work, jobz, uplo, n, a, lda, w, work.data(), lwork);
  lwork = lapack_int(work[0]);
  work.resize(lwork);
  TA_LAPACKE(syev_work, jobz, uplo, n, a, lda, w, work.data(), lwork);
}

template <typename T>
void heig(Matrix<T>& A, Matrix<T>& B, std::vector<T>& W) {
  lapack_int itype = 1;
  char jobz = 'V';
  char uplo = 'L';
  lapack_int n = A.rows();
  T* a = A.data();
  lapack_int lda = A.rows();
  T* b = B.data();
  lapack_int ldb = B.rows();
  W.resize(n);
  T* w = W.data();
  std::vector<T> work(1);
  lapack_int lwork = -1;
  TA_LAPACKE(sygv_work, itype, jobz, uplo, n, a, lda, b, ldb, w, work.data(), lwork);
  lwork = lapack_int(work[0]);
  work.resize(lwork);
  TA_LAPACKE(sygv_work, itype, jobz, uplo, n, a, lda, b, ldb, w, work.data(), lwork);
}

template <typename T>
void svd(Matrix<T>& A, std::vector<T>& S, Matrix<T>* U, Matrix<T>* VT) {
  lapack_int m = A.rows();
  lapack_int n = A.cols();
  T* a = A.data();
  lapack_int lda = A.rows();

  S.resize(std::min(m, n));
  T* s = S.data();

  char jobu = 'N';
  T* u = nullptr;
  lapack_int ldu = m;
  if (U) {
    jobu = 'A';
    U->resize(m, n);
    u = U->data();
    ldu = U->rows();
  }

  char jobvt = 'N';
  T* vt = nullptr;
  lapack_int ldvt = n;
  if (VT) {
    jobvt = 'A';
    VT->resize(n, m);
    vt = VT->data();
    ldvt = VT->rows();
  }

  std::vector<T> work(1);
  lapack_int lwork = -1;

  TA_LAPACKE(gesvd_work, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
             work.data(), lwork);
  lwork = lapack_int(work[0]);
  work.resize(lwork);
  TA_LAPACKE(gesvd_work, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
             work.data(), lwork);

}

template <typename T>
void lu_solve(Matrix<T>& A, Matrix<T>& B) {
  lapack_int n = A.rows();
  lapack_int nrhs = B.cols();
  T* a = A.data();
  lapack_int lda = A.rows();
  T* b = B.data();
  lapack_int ldb = B.rows();
  std::vector<lapack_int> ipiv(n);
  TA_LAPACKE(gesv, n, nrhs, a, lda, ipiv.data(), b, ldb);
}

template <typename T>
void lu_inv(Matrix<T>& A) {
  lapack_int n = A.rows();
  T* a = A.data();
  lapack_int lda = A.rows();
  std::vector<lapack_int> ipiv(n);
  TA_LAPACKE(getrf, n, n, a, lda, ipiv.data());
  std::vector<T> work(1);
  lapack_int lwork = -1;
  TA_LAPACKE(getri_work, n, a, lda, ipiv.data(), work.data(), lwork);
  lwork = (lapack_int)work[0];
  work.resize(lwork);
  TA_LAPACKE(getri_work, n, a, lda, ipiv.data(), work.data(), lwork);
}

#define TA_LAPACK_EXPLICIT(MATRIX, VECTOR)                        \
  template void cholesky(MATRIX&);                                \
  template void cholesky_linv(MATRIX&);                           \
  template void cholesky_solve(MATRIX&, MATRIX&);                 \
  template void cholesky_lsolve(TransposeFlag, MATRIX&, MATRIX&); \
  template void heig(MATRIX&, VECTOR&);                           \
  template void heig(MATRIX&, MATRIX&, VECTOR&);                  \
  template void svd(MATRIX&, VECTOR&, MATRIX*, MATRIX*);          \
  template void lu_solve(MATRIX&, MATRIX&);                       \
  template void lu_inv(MATRIX&);

TA_LAPACK_EXPLICIT(Matrix<double>, std::vector<double>);
TA_LAPACK_EXPLICIT(Matrix<float>, std::vector<float>);

}  // namespace TiledArray::math::linalg::rank_local
