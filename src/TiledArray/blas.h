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

#ifndef TILEDARRAY_BLAS_H__INCLUDED
#define TILEDARRAY_BLAS_H__INCLUDED

#include <TiledArray/config.h>
#include <fortran_ctypes.h>

typedef const integer* fint;

#ifdef TILEDARRAY_HAS_BLAS

extern "C" {

  // BLAS _GEMM declarations
  void sgemm(const char*, const char*, fint, fint, fint, const float*,
      const float*, fint, const float*, fint, const float*, float*, fint);
  void dgemm(const char*, const char*, fint, fint, fint, const double*,
      const double*, fint, const double*, fint, const double*, double*, fint);
  void cgemm(const char*, const char*, fint, fint, fint,
      const std::complex<float>*, const std::complex<float>*,
      fint, const std::complex<float>*, fint, const std::complex<float>*,
      std::complex<float>*, fint);
  void zgemm(const char*, const char*, fint, fint, fint,
      const std::complex<double>*, const std::complex<double>*, fint,
      const std::complex<double>*, fint, const std::complex<double>*,
      std::complex<double>*, fint);

  // BLAS _GEMV declarations
  void sgemv(const char* OpA, fint m, fint n, const float* alpha,
      const float* A, fint lda, const float* X, fint incX, const float* beta,
      float* Y, fint incY);
  void dgemv(const char* OpA, fint m, fint n, const double* alpha,
      const double* A, fint lda, const double* X, fint incX, const double* beta,
      double* Y, fint incY);
  void cgemv(const char* OpA, fint m, fint n, const std::complex<float>* alpha,
      const std::complex<float>* A, fint lda, const std::complex<float>* X,
      fint incX, const std::complex<float>* beta, std::complex<float>* Y,
      fint incY);
  void zgemv(const char* OpA, fint m, fint n, const std::complex<double>* alpha,
      const std::complex<double>* A, fint lda, const std::complex<double>* X,
      fint incX, const std::complex<double>* beta, std::complex<double>* Y,
      fint incY);

  // BLAS _GER declarations
  void sger(fint m, fint n, const float* alpha, const float* X, fint incX,
      const float* Y, fint incY, float* A, fint lda);
  void dger(fint m, fint n, const double* alpha, const double* X, fint incX, const double* Y, fint incY,
      double* A, fint lda);
  void cger(fint m, fint n, const std::complex<float>* alpha,
      const std::complex<float>* X, fint incX, const std::complex<float>* Y,
      fint incY, std::complex<float>* A,
      fint lda);
  void zger(fint m, fint n, const std::complex<double>* alpha,
      const std::complex<double>* X, fint incX, const std::complex<double>* Y,
      fint incY, std::complex<double>* A, fint lda);

  // BLAS _SCAL declarations
  void sscal(fint, const float*, float*, fint);
  void dscal(fint, const double*, double*, fint);
  void cscal(fint, const std::complex<float>*, std::complex<float>*, fint);
  void csscal(fint, const float*, std::complex<float>*, fint);
  void zscal(fint, const std::complex<double>*, std::complex<double>*, fint);
  void zdscal(fint, const double*, std::complex<double>*, fint);

  // BLAS _DOT declarations
  float sdot(fint, const float*, fint, const float*, fint);
  double ddot(fint, const double *, fint, const double *, fint);
  void cdotu(fint, const std::complex<float>*, fint,
      const std::complex<float>*, fint, std::complex<float>*);
  void zdotu(fint, const std::complex<double>*, fint,
      const std::complex<double>*, fint, std::complex<double>*);
}

#endif

#endif // TILEDARRAY_BLAS_H__INCLUDED
