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

#ifndef TILEDARRAY_BLAS_H__INCLUDED
#define TILEDARRAY_BLAS_H__INCLUDED

#include <linalg/cblas.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/math/eigen.h>

namespace TiledArray {
  namespace math {

    // BLAS _GEMM wrapper functions

    template <typename S1, typename T1, typename T2, typename S2, typename T3>
    inline void gemm(madness::cblas::CBLAS_TRANSPOSE op_a,
        madness::cblas::CBLAS_TRANSPOSE op_b, const integer m, const integer n,
        const integer k, const S1 alpha, const T1* a, const integer lda,
        const T2* b, const integer ldb, const S2 beta, T3* c, const integer ldc)
    {
      // Define operations
      static const unsigned int
          notrans_notrans     = 0x00000000,
          notrans_trans       = 0x00000004,
          trans_notrans       = 0x00000001,
          trans_trans         = 0x00000005,
          notrans_conjtrans   = 0x00000008,
          trans_conjtrans     = 0x00000009,
          conjtrans_notrans   = 0x00000002,
          conjtrans_trans     = 0x00000006,
          conjtrans_conjtrans = 0x0000000a;

      // Construct matrix maps for a, b, and c.
      typedef Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrixA_type;
      typedef Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrixB_type;
      typedef Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrixC_type;
      Eigen::Map<const matrixA_type, Eigen::AutoAlign, Eigen::OuterStride<> > A(a,
          (op_a == madness::cblas::NoTrans ? m : k),
          (op_a == madness::cblas::NoTrans ? k : m),
          Eigen::OuterStride<>(lda));
      Eigen::Map<const matrixB_type, Eigen::AutoAlign, Eigen::OuterStride<> > B(b,
          (op_b == madness::cblas::NoTrans ? k : n),
          (op_b == madness::cblas::NoTrans ? n : k),
          Eigen::OuterStride<>(ldb));
      Eigen::Map<matrixC_type, Eigen::AutoAlign, Eigen::OuterStride<> >
          C(c, m, n, Eigen::OuterStride<>(ldc));

      switch(op_a | (op_b << 2)) {
        case notrans_notrans:
          C.noalias() = alpha * A * B + beta * C;
          break;
        case notrans_trans:
          C.noalias() = alpha * A * B.transpose() + beta * C;
          break;
        case trans_notrans:
          C.noalias() = alpha * A.transpose() * B + beta * C;
          break;
        case trans_trans:
          C.noalias() = alpha * A.transpose() * B.transpose() + beta * C;
          break;

        case notrans_conjtrans:
          C.noalias() = alpha * A * B.adjoint() + beta * C;
          break;
        case trans_conjtrans:
          C.noalias() = alpha * A.transpose() * B.adjoint() + beta * C;
          break;
        case conjtrans_notrans:
          C.noalias() = alpha * A.adjoint() * B + beta * C;
          break;
        case conjtrans_trans:
          C.noalias() = alpha * A.adjoint() * B.transpose() + beta * C;
          break;
        case conjtrans_conjtrans:
          C.noalias() = alpha * A.adjoint() * B.adjoint() + beta * C;
          break;
      }
    }

    inline void gemm(madness::cblas::CBLAS_TRANSPOSE op_a,
        madness::cblas::CBLAS_TRANSPOSE op_b, const integer m, const integer n,
        const integer k, const float alpha, const float* a, const integer lda,
        const float* b, const integer ldb, const float beta, float* c, const integer ldc)
    {
      madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

    inline void gemm(madness::cblas::CBLAS_TRANSPOSE op_a,
        madness::cblas::CBLAS_TRANSPOSE op_b, const integer m, const integer n,
        const integer k, const double alpha, const double* a, const integer lda,
        const double* b, const integer ldb, const double beta, double* c, const integer ldc)
    {
      madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

    inline void gemm(madness::cblas::CBLAS_TRANSPOSE op_a,
        madness::cblas::CBLAS_TRANSPOSE op_b, const integer m, const integer n,
        const integer k, const std::complex<float> alpha, const std::complex<float>* a,
        const integer lda, const std::complex<float>* b, const integer ldb,
        const std::complex<float> beta, std::complex<float>* c, const integer ldc)
    {
      madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }

    inline void gemm(madness::cblas::CBLAS_TRANSPOSE op_a,
        madness::cblas::CBLAS_TRANSPOSE op_b, const integer m, const integer n,
        const integer k, const std::complex<double> alpha, const std::complex<double>* a,
        const integer lda, const std::complex<double>* b, const integer ldb,
        const std::complex<double> beta, std::complex<double>* c, const integer ldc)
    {
      madness::cblas::gemm(op_b, op_a, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }


    // BLAS _SCAL wrapper functions

    template <typename T, typename U>
    inline typename madness::enable_if<detail::is_numeric<T> >::type
    scale(const integer n, const T alpha, U* x) {
      eigen_map(x, n) *= alpha;
    }

    inline void scale(const integer n, const float alpha, float* x) {
      madness::cblas::scal(n, alpha, x, 1);
    }

    inline void scale(const integer n, const double alpha, double* x) {
      madness::cblas::scal(n, alpha, x, 1);
    }

    inline void scale(const integer n, const std::complex<float> alpha, std::complex<float>* x) {
      madness::cblas::scal(n, alpha, x, 1);
    }

    inline void scale(const integer n, const std::complex<double> alpha, std::complex<double>* x) {
      madness::cblas::scal(n, alpha, x, 1);
    }

    inline void scale(const integer n, const float alpha, std::complex<float>* x) {
      madness::cblas::scal(n, alpha, x, 1);
    }

    inline void scale(const integer n, const double alpha, std::complex<double>* x) {
      madness::cblas::scal(n, alpha, x, 1);
    }


    // BLAS _DOT wrapper functions

    template <typename T, typename U>
    T dot(const integer n, const T* x, const U* y) {
      return eigen_map(x, n).dot(eigen_map(y, n));
    }

    inline float dot(integer n, const float* x, const float* y) {
      return madness::cblas::dot(n, x, 1, y, 1);
    }

    inline double dot(integer n, const double* x, const double* y) {
      return madness::cblas::dot(n, x, 1, y, 1);
    }

    inline std::complex<float> dot(integer n, const std::complex<float>* x, const std::complex<float>* y) {
      return madness::cblas::dot(n, x, 1, y, 1);
    }

    inline std::complex<double> dot(integer n, const std::complex<double>* x, const std::complex<double>* y) {
      return madness::cblas::dot(n, x, 1, y, 1);
    }

    // Import the madness dot functions into the TiledArray namespace
    using madness::cblas::dot;


  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_BLAS_H__INCLUDED
