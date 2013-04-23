/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_MATH_H__INCLUDED
#define TILEDARRAY_MATH_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/blas.h>
#include <world/enable_if.h>
#include <Eigen/Core>

namespace TiledArray {
  namespace math {

    template <typename T>
    inline void gemm(const integer m, const integer n, const integer k, const T alpha, const T* a, const T* b, T* c) {

      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;
      typedef Eigen::Map<matrix_type, Eigen::AutoAlign> map_type;
      typedef Eigen::Map<const matrix_type, Eigen::AutoAlign> const_map_type;

      const_map_type A(a, m, k);
      const_map_type B(b, k, n);
      map_type C(c, m, n);

      C.noalias() += alpha * (A * B);
    }


#ifdef TILEDARRAY_HAS_BLAS

    // BLAS _GEMM wrapper functions

    inline void gemm(const integer m, const integer n, const integer k, const float alpha, const float* a, const float* b, float* c) {
      static const char *op[] = { "n","t" };
      static const float beta = 1.0;
      F77_SGEMM(op[0], op[0], &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
    }

    inline void gemm(const integer m, const integer n, const integer k, const double alpha, const double* a, const double* b, double* c) {
      static const char *op[] = { "n","t" };
      static const double beta = 1.0;
      F77_DGEMM(op[0], op[0], &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
    }

    inline void gemm(const integer m, const integer n, const integer k, const std::complex<float> alpha, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* c) {
      static const char *op[] = { "n","t","c" };
      static const std::complex<float> beta(1.0, 0.0);
      F77_CGEMM(op[0], op[0], &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
    }

    inline void gemm(const integer m, const integer n, const integer k, const std::complex<double> alpha, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* c) {
      static const char *op[] = { "n","t","c" };
      static const std::complex<double> beta(1.0, 0.0);
      F77_ZGEMM(op[0], op[0], &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
    }

#endif // TILEDARRAY_HAS_CBLAS

    template <typename T, typename U>
    inline typename madness::enable_if<detail::is_numeric<T> >::type
    scale(const integer n, const T alpha, U* x) {
      // Type defs
      typedef Eigen::Matrix<U, Eigen::Dynamic, 1> matrix_type;
      typedef Eigen::Map<matrix_type, Eigen::AutoAlign> map_type;

      map_type X(x, n);

      X *= alpha;
    }

#ifdef TILEDARRAY_HAS_BLAS

    // BLAS _SCAL wrapper functions

    inline void scale(const integer n, float alpha, float* x) {
      static const integer incX = 1;
      F77_SSCAL(&n, &alpha, x, &incX);
    }

    inline void scale(const integer n, double alpha, double* x) {
      static const integer incX = 1;
      F77_DSCAL(&n, &alpha, x, &incX);
    }

    inline void scale(const integer n, std::complex<float> alpha, std::complex<float>* x) {
      static const integer incX = 1;
      F77_CSCAL(&n, &alpha, x, &incX);
    }

    inline void scale(const integer n, std::complex<double> alpha, std::complex<double>* x) {
      static const integer incX = 1;
      F77_ZSCAL(&n, &alpha, x, &incX);
    }

    inline void scale(const integer n, float alpha, std::complex<float>* x) {
      static const integer incX = 1;
      F77_CSSCAL(&n, &alpha, x, &incX);
    }

    inline void scale(const integer n, double alpha, std::complex<double>* x) {
      static const integer incX = 1;
      F77_ZDSCAL(&n, &alpha, x, &incX);
    }

#endif // TILEDARRAY_HAS_CBLAS


    template <typename T, typename U>
    T dot(const integer n, const T* x, const U* y) {
      // Type defs
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> matrixt_type;
      typedef Eigen::Matrix<U, Eigen::Dynamic, 1> matrixu_type;
      typedef Eigen::Map<const matrixt_type, Eigen::AutoAlign> mapt_type;
      typedef Eigen::Map<const matrixu_type, Eigen::AutoAlign> mapu_type;

      // construct vector maps
      mapt_type X(x, n);
      mapu_type Y(y, n);

      return X.dot(Y);
    }

#ifdef TILEDARRAY_HAS_BLAS


    // BLAS _DOT wrapper functions

    inline float dot(integer n, const float* x, const float* y) {
      static const integer incX = 1, incY = 1;
      return F77_SDOT(&n, x, &incX, y, &incY);
    }

    inline double dot(integer n, const double* x, const double* y) {
      static const integer incX = 1, incY = 1;
      return F77_DDOT(&n, x, &incX, y, &incY);
    }

    inline std::complex<float> dot(integer n, const std::complex<float>* x, const std::complex<float>* y) {
      static const integer incX = 1, incY = 1;
      std::complex<float> result(0.0, 0.0);
      F77_CDOTU(&n, x, &incX, y, &incY, &result);
      return result;
    }

    inline std::complex<double> dot(integer n, const std::complex<double>* x, const std::complex<double>* y) {
      static const integer incX = 1, incY = 1;
      std::complex<double> result(0.0, 0.0);
      F77_ZDOTU(&n, x, &incX, y, &incY, &result);
      return result;
    }

    inline float dot(integer n, const float* x, integer incX, const float* y, integer incY) {
      return F77_SDOT(&n, x, &incX, y, &incY);
    }

    inline double dot(integer n, const double* x, integer incX, const double* y, integer incY) {
      return F77_DDOT(&n, x, &incX, y, &incY);
    }

    inline std::complex<float> dot(integer n, const std::complex<float>* x, integer incX,
                                   const std::complex<float>* y, integer incY) {
      std::complex<float> result(0.0, 0.0);
      F77_CDOTU(&n, x, &incX, y, &incY, &result);
      return result;
    }

    inline std::complex<double> dot(integer n, const std::complex<double>* x, integer incX,
                                    const std::complex<double>* y, integer incY) {
      std::complex<double> result(0.0, 0.0);
      F77_ZDOTU(&n, x, &incX, y, &incY, &result);
      return result;
    }

#endif // TILEDARRAY_HAS_CBLAS


    template <typename T>
    T square_norm(const integer n, const T* x) {
      // Type defs
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> matrix_type;
      typedef Eigen::Map<const matrix_type, Eigen::AutoAlign> map_type;

      map_type X(x, n);
      return X.squaredNorm();
    }

    namespace detail {
      template <typename T>
      struct abs_compare {
        bool operator()(T x, T y) { return std::fabs(x) < std::fabs(y); }
      };
    }

    template <typename T>
    T maxabs(const integer n, const T* x) {
      detail::abs_compare<T> cmp;
      return *(std::max_element(x, x+n, cmp));
    }

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_H__INCLUDED
