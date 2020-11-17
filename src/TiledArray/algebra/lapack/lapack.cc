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

#include <TiledArray/algebra/lapack/lapack.h>
#include <TiledArray/algebra/lapack/util.h>
#include <TiledArray/config.h>
#include <madness/tensor/clapack.h>
#include <Eigen/Core>

#define TA_LAPACK_CALL(name, args...)                                    \
  typedef T numeric_type;                                                \
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

namespace TiledArray::lapack {

template <typename T>
void cholesky(Matrix<T>& A) {
  char uplo = 'L';
  integer n = A.rows();
  auto* a = A.data();
  integer lda = n;
  integer info = 0;
#if defined(MADNESS_LINALG_USE_LAPACKE)
  TA_LAPACK_CALL(potrf, &uplo, &n, a, &lda, &info);
#else
  TA_LAPACK_CALL(potrf, &uplo, &n, a, &lda, &info, sizeof(char));
#endif
  if (info != 0) TA_EXCEPTION("LAPACK::potrf failed");
}

template <typename T>
void cholesky_linv(Matrix<T>& A) {
  char uplo = 'L';
  char diag = 'N';
  integer n = A.rows();
  auto* l = A.data();
  integer lda = n;
  integer info = 0;
  TA_LAPACK_CALL(trtri, &uplo, &diag, &n, l, &lda, &info);
  if (info != 0) TA_EXCEPTION("LAPACK::trtri failed");
}

template <typename T>
void cholesky_solve(Matrix<T>& A, Matrix<T>& X) {
  char uplo = 'L';
  integer n = A.rows();
  integer nrhs = X.cols();
  auto* a = A.data();
  auto* b = X.data();
  integer lda = n;
  integer ldb = n;
  integer info = 0;
  // TA_LAPACK_CALL(posv, &uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
  if (info != 0) TA_EXCEPTION("LAPACK::posv failed");
}

template <typename T>
void cholesky_lsolve(TransposeFlag transpose, Matrix<T>& A, Matrix<T>& X) {
  char uplo = 'L';
  char trans = transpose == TransposeFlag::Transpose
                   ? 'T'
                   : (transpose == TransposeFlag::NoTranspose ? 'N' : 'C');
  char diag = 'N';
  integer n = A.rows();
  integer nrhs = X.cols();
  auto* a = A.data();
  auto* b = X.data();
  integer lda = n;
  integer ldb = n;
  integer info = 0;
  // TA_LAPACK_CALL(trtrs, &uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb,
  // &info);
  if (info != 0) TA_EXCEPTION("LAPACK::trtrs failed");
}

template <typename T>
void hereig(Matrix<T>& A, Vector<T>& W) {
  char jobz = 'V';
  char uplo = 'L';
  integer n = A.rows();
  T* a = A.data();
  integer lda = A.rows();
  T* w = W.data();
  integer lwork = -1;
  integer info;
  T lwork_dummy;
#ifndef MADNESS_LINALG_USE_LAPACKE
  TA_LAPACK_CALL(syev, &jobz, &uplo, &n, a, &lda, w, &lwork_dummy, &lwork,
                 &info, sizeof(char), sizeof(char));
#else
  TA_LAPACK_CALL(syev, &jobz, &uplo, &n, a, &lda, w, &lwork_dummy, &lwork,
                 &info);
#endif
  lwork = integer(lwork_dummy);
  Vector<T> work(lwork);
#ifndef MADNESS_LINALG_USE_LAPACKE
  TA_LAPACK_CALL(syev, &jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, &info,
                 sizeof(char), sizeof(char));
#else
  TA_LAPACK_CALL(syev, &jobz, &uplo, &n, a, &lda, w, work.data(), &lwork,
                 &info);
#endif
  if (info != 0) TA_EXCEPTION("lapack::hereig failed");
}

template <typename T>
void hereig_gen(Matrix<T>& A, Matrix<T>& B, Vector<T>& W) {
  integer itype = 1;
  char jobz = 'V';
  char uplo = 'L';
  integer n = A.rows();
  T* a = A.data();
  integer lda = A.rows();
  T* b = B.data();
  integer ldb = B.rows();
  T* w = W.data();
  integer lwork = -1;
  integer info;
  T lwork_dummy;
#ifndef MADNESS_LINALG_USE_LAPACKE
  TA_LAPACK_CALL(sygv, &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w,
                 &lwork_dummy, &lwork, &info, sizeof(char), sizeof(char));
#else
  TA_LAPACK_CALL(sygv, &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w,
                 &lwork_dummy, &lwork, &info);
#endif
  lwork = integer(lwork_dummy);
  Vector<T> work(lwork);
#ifndef MADNESS_LINALG_USE_LAPACKE
  TA_LAPACK_CALL(sygv, &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w,
                 work.data(), &lwork, &info, sizeof(char), sizeof(char));
#else
  TA_LAPACK_CALL(sygv, &itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w,
                 work.data(), &lwork, &info);
#endif
  if (info != 0) TA_EXCEPTION("lapack::hereig_gen failed");
}

template <typename T>
void svd(Matrix<T>& A, Vector<T>& S, Matrix<T>* U, Matrix<T>* VT) {
  integer m = A.rows();
  integer n = A.cols();
  T* a = A.data();
  integer lda = A.rows();

  S.resize(std::max(m, n));
  T* s = S.data();

  char jobu = 'N';
  T* u = nullptr;
  integer ldu = 0;
  if (U) {
    jobu = 'A';
    U->resize(m, n);
    u = U->data();
    ldu = U->rows();
  }

  char jobvt = 'N';
  T* vt = nullptr;
  integer ldvt = 0;
  if (VT) {
    jobvt = 'A';
    VT->resize(n, m);
    vt = VT->data();
    ldvt = VT->rows();
  }

  integer lwork = -1;
  integer info;
  T lwork_dummy;

#ifndef MADNESS_LINALG_USE_LAPACKE
  TA_LAPACK_CALL(gesvd, &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                 &lwork_dummy, &lwork, &info, sizeof(char), sizeof(char));
#else
  TA_LAPACK_CALL(gesvd, &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                 &lwork_dummy, &lwork, &info);
#endif
  lwork = integer(lwork_dummy);
  Vector<T> work(lwork);
#ifndef MADNESS_LINALG_USE_LAPACKE
  TA_LAPACK_CALL(gesvd, &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                 &lwork_dummy, &lwork, &info, sizeof(char), sizeof(char));
#else
  TA_LAPACK_CALL(gesvd, &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
                 &lwork_dummy, &lwork, &info);
#endif
  if (info != 0) TA_EXCEPTION("lapack::hereig_gen failed");
}

#define TA_LAPACK_EXPLICIT(MATRIX, VECTOR)                        \
  template void cholesky(MATRIX&);                                \
  template void cholesky_linv(MATRIX&);                           \
  template void cholesky_solve(MATRIX&, MATRIX&);                 \
  template void cholesky_lsolve(TransposeFlag, MATRIX&, MATRIX&); \
  template void hereig(MATRIX&, VECTOR&);                         \
  template void hereig_gen(MATRIX&, MATRIX&, VECTOR&);            \
  template void svd(MATRIX&, VECTOR&, MATRIX*, MATRIX*);

TA_LAPACK_EXPLICIT(lapack::Matrix<double>, lapack::Vector<double>);

}  // namespace TiledArray::lapack
