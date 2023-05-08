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

#include <TiledArray/math/blas.h>
#include <TiledArray/math/lapack.h>
#include <TiledArray/math/linalg/rank-local.h>

template <class F, typename... Args>
inline int ta_lapack_fortran_call(F f, Args... args) {
  lapack_int info;
  auto ptr = [](auto&& a) {
    using T = std::remove_reference_t<decltype(a)>;
    if constexpr (std::is_pointer_v<T>)
      return a;
    else
      return &a;
  };
  f(ptr(args)..., &info);
  return info;
}

#define TA_LAPACK_ERROR(F) throw lapack::Error("lapack::" #F " failed")

#define TA_LAPACK_FORTRAN_CALL(F, ARGS...) \
  ((ta_lapack_fortran_call(F, ARGS) == 0) || (TA_LAPACK_ERROR(F), 0))

/// \brief Invokes the Fortran LAPACK API

/// \warning TA_LAPACK_FORTRAN(fn,args) can be called only from template
/// context, with `T` defining the element type
#define TA_LAPACK_FORTRAN(name, args...)                    \
  {                                                         \
    using numeric_type = T;                                 \
    if constexpr (std::is_same_v<numeric_type, double>)     \
      TA_LAPACK_FORTRAN_CALL(d##name, args);                \
    else if constexpr (std::is_same_v<numeric_type, float>) \
      TA_LAPACK_FORTRAN_CALL(s##name, args);                \
    else                                                    \
      std::abort();                                         \
  }

/// TA_LAPACK(fn,args) invoked lapack::fn directly and checks the return value
#define TA_LAPACK(name, args...) \
  ((::lapack::name(args) == 0) || (TA_LAPACK_ERROR(name), 0))

namespace TiledArray::math::linalg::rank_local {

using integer = math::blas::integer;

template <typename T>
void cholesky(Matrix<T>& A) {
  auto uplo = lapack::Uplo::Lower;
  integer n = A.rows();
  auto* a = A.data();
  integer lda = n;
  TA_LAPACK(potrf, uplo, n, a, lda);
}

template <typename T>
void cholesky_linv(Matrix<T>& A) {
  auto uplo = lapack::Uplo::Lower;
  auto diag = lapack::Diag::NonUnit;
  integer n = A.rows();
  auto* l = A.data();
  integer lda = n;
  TA_LAPACK(trtri, uplo, diag, n, l, lda);
}

template <typename T>
void cholesky_solve(Matrix<T>& A, Matrix<T>& X) {
  auto uplo = lapack::Uplo::Lower;
  integer n = A.rows();
  integer nrhs = X.cols();
  auto* a = A.data();
  auto* b = X.data();
  integer lda = n;
  integer ldb = n;
  TA_LAPACK(posv, uplo, n, nrhs, a, lda, b, ldb);
}

template <typename T>
void cholesky_lsolve(Op transpose, Matrix<T>& A, Matrix<T>& X) {
  auto uplo = lapack::Uplo::Lower;
  auto diag = lapack::Diag::NonUnit;
  integer n = A.rows();
  integer nrhs = X.cols();
  auto* a = A.data();
  auto* b = X.data();
  integer lda = n;
  integer ldb = n;
  TA_LAPACK(trtrs, uplo, transpose, diag, n, nrhs, a, lda, b, ldb);
}

template <typename T>
void heig(Matrix<T>& A, std::vector<TiledArray::detail::real_t<T>>& W) {
  auto jobz = lapack::Job::Vec;
  auto uplo = lapack::Uplo::Lower;
  integer n = A.rows();
  T* a = A.data();
  integer lda = A.rows();
  W.resize(n);
  auto* w = W.data();
  if constexpr (TiledArray::detail::is_complex_v<T>)
    TA_LAPACK(heev, jobz, uplo, n, a, lda, w);
  else
    TA_LAPACK(syev, jobz, uplo, n, a, lda, w);
}

template <typename T>
void heig(Matrix<T>& A, Matrix<T>& B,
          std::vector<TiledArray::detail::real_t<T>>& W) {
  integer itype = 1;
  auto jobz = lapack::Job::Vec;
  auto uplo = lapack::Uplo::Lower;
  integer n = A.rows();
  T* a = A.data();
  integer lda = A.rows();
  T* b = B.data();
  integer ldb = B.rows();
  W.resize(n);
  auto* w = W.data();
  if constexpr (TiledArray::detail::is_complex_v<T>)
    TA_LAPACK(hegv, itype, jobz, uplo, n, a, lda, b, ldb, w);
  else
    TA_LAPACK(sygv, itype, jobz, uplo, n, a, lda, b, ldb, w);
}

template <typename T>
void svd(Job jobu, Job jobvt, Matrix<T>& A,
         std::vector<TiledArray::detail::real_t<T>>& S, Matrix<T>* U,
         Matrix<T>* VT) {
  integer m = A.rows();
  integer n = A.cols();
  integer k = std::min(m, n);
  T* a = A.data();
  integer lda = A.rows();

  S.resize(k);
  auto* s = S.data();

  T* u = nullptr;
  T* vt = nullptr;
  integer ldu = 1, ldvt = 1;
  if ((jobu == Job::SomeVec or jobu == Job::AllVec) and (not U))
    TA_LAPACK_ERROR(
        "Requested out-of-place right singular vectors with null U input");
  if ((jobvt == Job::SomeVec or jobvt == Job::AllVec) and (not VT))
    TA_LAPACK_ERROR(
        "Requested out-of-place left singular vectors with null VT input");

  if (jobu == Job::SomeVec) {
    U->resize(m, k);
    u = U->data();
    ldu = m;
  }

  if (jobu == Job::AllVec) {
    U->resize(m, m);
    u = U->data();
    ldu = m;
  }

  if (jobvt == Job::SomeVec) {
    VT->resize(k, n);
    vt = VT->data();
    ldvt = k;
  }

  if (jobvt == Job::AllVec) {
    VT->resize(n, n);
    vt = VT->data();
    ldvt = n;
  }

  TA_LAPACK(gesvd, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <typename T>
void lu_solve(Matrix<T>& A, Matrix<T>& B) {
  integer n = A.rows();
  integer nrhs = B.cols();
  T* a = A.data();
  integer lda = A.rows();
  T* b = B.data();
  integer ldb = B.rows();
  std::vector<integer> ipiv(n);
  TA_LAPACK(gesv, n, nrhs, a, lda, ipiv.data(), b, ldb);
}

template <typename T>
void lu_inv(Matrix<T>& A) {
  integer n = A.rows();
  T* a = A.data();
  integer lda = A.rows();
  std::vector<integer> ipiv(n);
  TA_LAPACK(getrf, n, n, a, lda, ipiv.data());
  TA_LAPACK(getri, n, a, lda, ipiv.data());
}

template <bool QOnly, typename T>
void householder_qr(Matrix<T>& V, Matrix<T>& R) {
  integer m = V.rows();
  integer n = V.cols();
  integer k = std::min(m, n);
  integer ldv = V.rows();  // Col Major
  T* v = V.data();
  std::vector<T> tau(k);
  lapack::geqrf(m, n, v, ldv, tau.data());

  // Extract R
  if constexpr (not QOnly) {
    // Resize R just in case
    R.resize(k, n);
    R.fill(0.);
    // Extract Upper triangle into R
    integer ldr = R.rows();
    T* r = R.data();
    lapack::lacpy(lapack::MatrixType::Upper, k, n, v, ldv, r, ldr);
  }

  // Explicitly form Q
  // TODO: This is wrong for complex, but it doesn't look like R/C is caught
  //       anywhere else either...
  if constexpr (TiledArray::detail::is_complex_v<T>)
    lapack::ungqr(m, n, k, v, ldv, tau.data());
  else
    lapack::orgqr(m, n, k, v, ldv, tau.data());
}

#define TA_LAPACK_EXPLICIT(MATRIX, VECTOR)                         \
  template void cholesky(MATRIX&);                                 \
  template void cholesky_linv(MATRIX&);                            \
  template void cholesky_solve(MATRIX&, MATRIX&);                  \
  template void cholesky_lsolve(Op, MATRIX&, MATRIX&);             \
  template void heig(MATRIX&, VECTOR&);                            \
  template void heig(MATRIX&, MATRIX&, VECTOR&);                   \
  template void svd(Job, Job, MATRIX&, VECTOR&, MATRIX*, MATRIX*); \
  template void lu_solve(MATRIX&, MATRIX&);                        \
  template void lu_inv(MATRIX&);                                   \
  template void householder_qr<true>(MATRIX&, MATRIX&);            \
  template void householder_qr<false>(MATRIX&, MATRIX&);

TA_LAPACK_EXPLICIT(Matrix<double>, std::vector<double>);
TA_LAPACK_EXPLICIT(Matrix<float>, std::vector<float>);
TA_LAPACK_EXPLICIT(Matrix<std::complex<double>>, std::vector<double>);
TA_LAPACK_EXPLICIT(Matrix<std::complex<float>>, std::vector<float>);

}  // namespace TiledArray::math::linalg::rank_local
