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
 */

#ifndef TILEDARRAY_MATH_BLAS_H__INCLUDED
#define TILEDARRAY_MATH_BLAS_H__INCLUDED

#include <TiledArray/config.h>
#include <fortran_ctypes.h>

typedef const integer* fint;

#ifdef TILEDARRAY_HAS_BLAS

extern "C" {

  // BLAS _GEMM declarations
  void F77_SGEMM(const char*, const char*, fint, fint, fint, const float*,
      const float*, fint, const float*, fint, const float*, float*, fint);
  void F77_DGEMM(const char*, const char*, fint, fint, fint, const double*,
      const double*, fint, const double*, fint, const double*, double*, fint);
  void F77_CGEMM(const char*, const char*, fint, fint, fint,
      const std::complex<float>*, const std::complex<float>*,
      fint, const std::complex<float>*, fint, const std::complex<float>*,
      std::complex<float>*, fint);
  void F77_ZGEMM(const char*, const char*, fint, fint, fint,
      const std::complex<double>*, const std::complex<double>*, fint,
      const std::complex<double>*, fint, const std::complex<double>*,
      std::complex<double>*, fint);

  // BLAS _GEMV declarations
  void F77_SGEMV(const char* OpA, fint m, fint n, const float* alpha,
      const float* A, fint lda, const float* X, fint incX, const float* beta,
      float* Y, fint incY);
  void F77_DGEMV(const char* OpA, fint m, fint n, const double* alpha,
      const double* A, fint lda, const double* X, fint incX, const double* beta,
      double* Y, fint incY);
  void F77_CGEMV(const char* OpA, fint m, fint n, const std::complex<float>* alpha,
      const std::complex<float>* A, fint lda, const std::complex<float>* X,
      fint incX, const std::complex<float>* beta, std::complex<float>* Y,
      fint incY);
  void F77_ZGEMV(const char* OpA, fint m, fint n, const std::complex<double>* alpha,
      const std::complex<double>* A, fint lda, const std::complex<double>* X,
      fint incX, const std::complex<double>* beta, std::complex<double>* Y,
      fint incY);

  // BLAS _GER declarations
  void F77_SGER(fint m, fint n, const float* alpha, const float* X, fint incX,
      const float* Y, fint incY, float* A, fint lda);
  void F77_DGER(fint m, fint n, const double* alpha, const double* X, fint incX, const double* Y, fint incY,
      double* A, fint lda);
  void cger(fint m, fint n, const std::complex<float>* alpha,
      const std::complex<float>* X, fint incX, const std::complex<float>* Y,
      fint incY, std::complex<float>* A,
      fint lda);
  void F77_ZGER(fint m, fint n, const std::complex<double>* alpha,
      const std::complex<double>* X, fint incX, const std::complex<double>* Y,
      fint incY, std::complex<double>* A, fint lda);

  // BLAS _SCAL declarations
  void F77_SSCAL(fint, const float*, float*, fint);
  void F77_DSCAL(fint, const double*, double*, fint);
  void F77_CSCAL(fint, const std::complex<float>*, std::complex<float>*, fint);
  void F77_CSSCAL(fint, const float*, std::complex<float>*, fint);
  void F77_ZSCAL(fint, const std::complex<double>*, std::complex<double>*, fint);
  void F77_ZDSCAL(fint, const double*, std::complex<double>*, fint);

  // BLAS _DOT declarations
  float F77_SDOT(fint, const float*, fint, const float*, fint);
  double F77_DDOT(fint, const double *, fint, const double *, fint);
  void F77_CDOTU(fint, const std::complex<float>*, fint,
      const std::complex<float>*, fint, std::complex<float>*);
  void F77_ZDOTU(fint, const std::complex<double>*, fint,
      const std::complex<double>*, fint, std::complex<double>*);
}

#endif

#endif // TILEDARRAY_MATH_BLAS_H__INCLUDED
