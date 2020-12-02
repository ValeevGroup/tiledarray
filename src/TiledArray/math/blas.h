/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  blas.h
 *  Nov 17, 2013
 *
 */

#ifndef TILEDARRAY_MATH_BLAS_H__INCLUDED
#define TILEDARRAY_MATH_BLAS_H__INCLUDED

#include <TiledArray/external/eigen.h>
#include <TiledArray/type_traits.h>
#include <madness/tensor/cblas.h>

namespace TiledArray::math::blas {

using TransposeFlag = madness::cblas::CBLAS_TRANSPOSE;
static constexpr auto NoTranspose = madness::cblas::NoTrans;
static constexpr auto Transpose = madness::cblas::Trans;
static constexpr auto ConjTranspose = madness::cblas::ConjTrans;

template<typename T, int Options = ::Eigen::ColMajor>
using Matrix = ::Eigen::Matrix<T, ::Eigen::Dynamic, ::Eigen::Dynamic, Options>;

template <typename T>
using Vector = ::Eigen::Matrix<T, ::Eigen::Dynamic, 1, ::Eigen::ColMajor>;

// BLAS _GEMM wrapper functions

template <typename S1, typename T1, typename T2, typename S2, typename T3>
inline void gemm(TransposeFlag op_a,
                 TransposeFlag op_b, const integer m,
                 const integer n, const integer k, const S1 alpha, const T1* a,
                 const integer lda, const T2* b, const integer ldb,
                 const S2 beta, T3* c, const integer ldc) {
  // Define operations
  static const unsigned int notrans_notrans = 0x00000000,
                            notrans_trans = 0x00000004,
                            trans_notrans = 0x00000001,
                            trans_trans = 0x00000005,
                            notrans_conjtrans = 0x00000008,
                            trans_conjtrans = 0x00000009,
                            conjtrans_notrans = 0x00000002,
                            conjtrans_trans = 0x00000006,
                            conjtrans_conjtrans = 0x0000000a;

  // Construct matrix maps for a, b, and c.
  typedef Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrixA_type;
  typedef Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrixB_type;
  typedef Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrixC_type;
  Eigen::Map<const matrixA_type, Eigen::AutoAlign, Eigen::OuterStride<>> A(
      a, (op_a == madness::cblas::NoTrans ? m : k),
      (op_a == madness::cblas::NoTrans ? k : m), Eigen::OuterStride<>(lda));
  Eigen::Map<const matrixB_type, Eigen::AutoAlign, Eigen::OuterStride<>> B(
      b, (op_b == madness::cblas::NoTrans ? k : n),
      (op_b == madness::cblas::NoTrans ? n : k), Eigen::OuterStride<>(ldb));
  Eigen::Map<matrixC_type, Eigen::AutoAlign, Eigen::OuterStride<>> C(
      c, m, n, Eigen::OuterStride<>(ldc));

  const bool beta_is_nonzero = (beta != static_cast<S2>(0));

  switch (op_a | (op_b << 2)) {
    case notrans_notrans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A * B + beta * C;
      else
        C.noalias() = alpha * A * B;
      break;
    case notrans_trans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A * B.transpose() + beta * C;
      else
        C.noalias() = alpha * A * B.transpose();
      break;
    case trans_notrans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A.transpose() * B + beta * C;
      else
        C.noalias() = alpha * A.transpose() * B;
      break;
    case trans_trans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A.transpose() * B.transpose() + beta * C;
      else
        C.noalias() = alpha * A.transpose() * B.transpose();
      break;

    case notrans_conjtrans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A * B.adjoint() + beta * C;
      else
        C.noalias() = alpha * A * B.adjoint();
      break;
    case trans_conjtrans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A.transpose() * B.adjoint() + beta * C;
      else
        C.noalias() = alpha * A.transpose() * B.adjoint();
      break;
    case conjtrans_notrans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A.adjoint() * B + beta * C;
      else
        C.noalias() = alpha * A.adjoint() * B;
      break;
    case conjtrans_trans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A.adjoint() * B.transpose() + beta * C;
      else
        C.noalias() = alpha * A.adjoint() * B.transpose();
      break;
    case conjtrans_conjtrans:
      if (beta_is_nonzero)
        C.noalias() = alpha * A.adjoint() * B.adjoint() + beta * C;
      else
        C.noalias() = alpha * A.adjoint() * B.adjoint();
      break;
  }
}

inline void gemm(TransposeFlag op_a,
                 TransposeFlag op_b, const integer m,
                 const integer n, const integer k, const float alpha,
                 const float* a, const integer lda, const float* b,
                 const integer ldb, const float beta, float* c,
                 const integer ldc) {
  madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c,
                       ldc);
}

inline void gemm(TransposeFlag op_a,
                 TransposeFlag op_b, const integer m,
                 const integer n, const integer k, const double alpha,
                 const double* a, const integer lda, const double* b,
                 const integer ldb, const double beta, double* c,
                 const integer ldc) {
  madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c,
                       ldc);
}

inline void gemm(TransposeFlag op_a,
                 TransposeFlag op_b, const integer m,
                 const integer n, const integer k,
                 const std::complex<float> alpha, const std::complex<float>* a,
                 const integer lda, const std::complex<float>* b,
                 const integer ldb, const std::complex<float> beta,
                 std::complex<float>* c, const integer ldc) {
  madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c,
                       ldc);
}

inline void gemm(TransposeFlag op_a,
                 TransposeFlag op_b, const integer m,
                 const integer n, const integer k,
                 const std::complex<double> alpha,
                 const std::complex<double>* a, const integer lda,
                 const std::complex<double>* b, const integer ldb,
                 const std::complex<double> beta, std::complex<double>* c,
                 const integer ldc) {
  madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c,
                       ldc);
}

// BLAS _SCAL wrapper functions

template <typename T, typename U>
inline typename std::enable_if<detail::is_numeric_v<T>>::type scale(
    const integer n, const T alpha, U* x) {
  Vector<T>::Map(x, n) *= alpha;
}

inline void scale(const integer n, const float alpha, float* x) {
  madness::cblas::scal(n, alpha, x, 1);
}

inline void scale(const integer n, const double alpha, double* x) {
  madness::cblas::scal(n, alpha, x, 1);
}

inline void scale(const integer n, const std::complex<float> alpha,
                  std::complex<float>* x) {
  madness::cblas::scal(n, alpha, x, 1);
}

inline void scale(const integer n, const std::complex<double> alpha,
                  std::complex<double>* x) {
  madness::cblas::scal(n, alpha, x, 1);
}

inline void scale(const integer n, const float alpha, std::complex<float>* x) {
  madness::cblas::scal(n, alpha, x, 1);
}

inline void scale(const integer n, const double alpha,
                  std::complex<double>* x) {
  madness::cblas::scal(n, alpha, x, 1);
}

// BLAS _DOT wrapper functions

template <typename T, typename U>
T dot(const integer n, const T* x, const U* y) {
  return Vector<T>::Map(x, n).dot(Vector<T>::Map(y, n));
}

inline float dot(integer n, const float* x, const float* y) {
  return madness::cblas::dot(n, x, 1, y, 1);
}

inline double dot(integer n, const double* x, const double* y) {
  return madness::cblas::dot(n, x, 1, y, 1);
}

inline std::complex<float> dot(integer n, const std::complex<float>* x,
                               const std::complex<float>* y) {
  return madness::cblas::dot(n, x, 1, y, 1);
}

inline std::complex<double> dot(integer n, const std::complex<double>* x,
                                const std::complex<double>* y) {
  return madness::cblas::dot(n, x, 1, y, 1);
}

// Import the madness dot functions into the TiledArray namespace
using madness::cblas::dot;

}  // namespace TiledArray::local

namespace TiledArray {
  namespace blas = TiledArray::math::blas;
}

#endif  // TILEDARRAY_MATH_BLAS_H__INCLUDED
