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


#ifdef TILEDARRAY_HAS_CBLAS

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const double alpha, const double* a,
        const long lda, const double* b, const long ldb,
        const double beta, double* c, const long ldc)
    {
      cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const std::complex<double> alpha, const std::complex<double>* a,
        const long lda, const std::complex<double>* b, const long ldb,
        const std::complex<double> beta, std::complex<double>* c, const long ldc)
    {
      cblas_zgemm(CblasRowMajor, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const float alpha, const float* a, const long lda, const float* b,
        const long ldb, const float beta, float* c, const long ldc)
    {
      cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    inline void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
        const long m, const long n, const long k,
        const std::complex<float> alpha, const std::complex<float>* a,
        const long lda, const std::complex<float>* b, const long ldb,
        const std::complex<float> beta, std::complex<float>* c, const long ldc)
    {
      cblas_cgemm(CblasRowMajor, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    }

#endif // TILEDARRAY_HAS_CBLAS

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

    inline void mxmT(const long m, const long n, const long k, const double* a, const double* b, double* c) {
      const double one = 1.0;
      gemm(CblasNoTrans, CblasTrans, m, n, k, one, a, k, b, k, one, c, n);
    }

    inline void mxmT(const long m, const long n, const long k, const float* a, const float* b, float* c) {
      const double one = 1.0;
      gemm(CblasNoTrans, CblasTrans, m, n, k, one, a, k, b, k, one, c, n);
    }

    inline void mxmT(const long m, const long n, const long k, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* c) {
      std::complex<double> one(1.0);
      gemm(CblasNoTrans, CblasTrans, m, n, k, one, a, k, b, k, one, c, n);
    }

    inline void mxmT(const long m, const long n, const long k, const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* c) {
      std::complex<float> one(1.0);
      gemm(CblasNoTrans, CblasTrans, m, n, k, one, a, k, b, k, one, c, n);
    }

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_CBLAS_H__INCLUDED
