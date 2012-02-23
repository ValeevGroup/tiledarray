#ifndef TILEDARRAY_CBLAS_H__INCLUDED
#define TILEDARRAY_CBLAS_H__INCLUDED

#include <TiledArray/config.h>
#include <Eigen/Core>

// Include CBLAS header
#ifdef TILEDARRAY_HAS_CBLAS
#if  defined(HAVE_MKL_H)
#include <mkl.h>
#elif defined(HAVE_CBLAS_H)
extern "C" {
#include <cblas.h>
}
#else
#error No sutable CBLAS header found.
#endif // HAVE_MKL_H  || HAVE_CBLAS_H
#endif // TILEDARRAY_HAS_CBLAS

namespace TiledArray {
  namespace detail {

    template <typename T>
    inline void mxmT(const long m, const long n, const long k, const T* a, const T* b, T* c) {

      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;
      typedef Eigen::Map<matrix_type, Eigen::AutoAlign> map_type;
      typedef Eigen::Map<const matrix_type, Eigen::AutoAlign> const_map_type;

      const_map_type A(a, m, k);
      const_map_type B(b, n, k);
      map_type C(c, m, n);

      C.noalias() += A * B.transpose();
    }


#ifdef TILEDARRAY_HAS_CBLAS

    inline void mxmT(const long m, const long n, const long k, const double* a, const double* b, double* c) {
      const double one = 1.0;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, one,
          const_cast<double*>(a), k, const_cast<double*>(b), k, one, c, n);
    }

    inline void mxmT(const long m, const long n, const long k, const float* a, const float* b, float* c) {
      const float one = 1.0;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, one,
          const_cast<float*>(a), k, const_cast<float*>(b), k, one, c, n);
    }

    inline void mxmT(const long m, const long n, const long k, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* c) {
      std::complex<double> one(1.0);
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, &one,
          const_cast<double*>(reinterpret_cast<const double*>(a)), k,
          const_cast<double*>(reinterpret_cast<const double*>(b)), k, &one,
          reinterpret_cast<double*>(c), n);
    }

    inline void mxmT(const long m, const long n, const long k, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* c) {
      std::complex<float> one(1.0);
      cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, &one,
          const_cast<float*>(reinterpret_cast<const float*>(a)), k,
          const_cast<float*>(reinterpret_cast<const float*>(b)), k, &one,
          reinterpret_cast<float*>(c), n);
    }

#endif // TILEDARRAY_HAS_CBLAS

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_CBLAS_H__INCLUDED
