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

namespace TiledArray {
  namespace math {

    // BLAS _GEMM wrapper functions

    template <typename T>
    inline void gemm(const integer m, const integer n, const integer k, const T alpha, const T* a, const T* b, T* c) {
      eigen_map(c, m, n).noalias() += alpha * (eigen_map(a, m, k) * eigen_map(b, k, n));
    }

    inline void gemm(const integer m, const integer n, const integer k, const float alpha, const float* a, const float* b, float* c) {
      madness::cblas::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
          n, m, k, alpha, b, n, a, k, 1.0, c, n);
    }

    inline void gemm(const integer m, const integer n, const integer k, const double alpha, const double* a, const double* b, double* c) {
      madness::cblas::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
          n, m, k, alpha, b, n, a, k, 1.0, c, n);
    }

    inline void gemm(const integer m, const integer n, const integer k, const std::complex<float> alpha, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* c) {
      madness::cblas::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
          n, m, k, alpha, b, n, a, k, std::complex<float>(1.0, 0.0), c, n);
    }

    inline void gemm(const integer m, const integer n, const integer k, const std::complex<double> alpha, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* c) {
      madness::cblas::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
          n, m, k, alpha, b, n, a, k, std::complex<double>(1.0, 0.0), c, n);
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
