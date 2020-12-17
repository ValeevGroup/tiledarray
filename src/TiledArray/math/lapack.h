#ifndef TILEDARRAY_MATH_LAPACK_H__INCLUDED
#define TILEDARRAY_MATH_LAPACK_H__INCLUDED

#include <TiledArray/config.h>

#if defined(BTAS_HAS_INTEL_MKL)

#include <mkl_lapack.h>
#include <mkl_lapacke.h> // lapack_int

#elif defined(BTAS_HAS_LAPACKE)

    // see https://github.com/xianyi/OpenBLAS/issues/1992 why this is needed to prevent lapacke.h #define'ing I
#   include <complex>
#   ifndef lapack_complex_float
#     define lapack_complex_float std::complex<float>
#   else // lapack_complex_float
      static_assert(sizeof(std::complex<float>)==sizeof(lapack_complex_float), "sizes of lapack_complex_float and std::complex<float> do not match");
#   endif // lapack_complex_float
#   ifndef lapack_complex_double
#     define lapack_complex_double std::complex<double>
#   else // lapack_complex_double
      static_assert(sizeof(std::complex<double>)==sizeof(lapack_complex_double), "sizes of lapack_complex_double and std::complex<double> do not match");
#   endif // lapack_complex_double

#if defined(BTAS_LAPACKE_HEADER)
#include BTAS_LAPACKE_HEADER
#else
#include <lapacke.h>
#endif

#elif defined(__APPLE__)

#include <Accelerate/Accelerate.h>
using lapack_int = __CLPK_integer;

#else

#error "Could not find Lapack/e"

#endif

#endif // TILEDARRAY_MATH_LAPACK_H__INCLUDED
