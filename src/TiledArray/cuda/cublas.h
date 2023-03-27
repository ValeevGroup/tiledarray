/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  July 23, 2018
 *
 */

#ifndef TILEDARRAY_MATH_CUBLAS_H__INCLUDED
#define TILEDARRAY_MATH_CUBLAS_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/error.h>
#include <TiledArray/tensor/complex.h>
#include <cublas_v2.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <TiledArray/math/blas.h>

#define CublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)

inline void __cublasSafeCall(cublasStatus_t err, const char *file,
                             const int line) {
#ifdef TILEDARRAY_CHECK_CUDA_ERROR
  if (CUBLAS_STATUS_SUCCESS != err) {
    std::stringstream ss;
    ss << "cublasSafeCall() failed at: " << file << "(" << line << ")";
    std::string what = ss.str();
    throw std::runtime_error(what);
  }
#endif

  return;
}

namespace TiledArray {

namespace detail {

template <typename T>
auto cublasPointer(T *std_complex_ptr) {
  using Scalar = TiledArray::detail::scalar_t<T>;
  static_assert(std::is_same_v<Scalar, double> ||
                std::is_same_v<Scalar, float>);
  constexpr bool DP = std::is_same_v<Scalar, double>;
  using cuT = std::conditional_t<std::is_same_v<Scalar, double>,
                                 cuDoubleComplex, cuComplex>;
  if constexpr (std::is_const_v<
                    std::remove_pointer_t<decltype(std_complex_ptr)>>) {
    return reinterpret_cast<const cuT *>(std_complex_ptr);
  } else
    return reinterpret_cast<cuT *>(std_complex_ptr);
};

}  // namespace detail

/*
 * cuBLAS interface functions
 */

/**
 * cuBLASHandlePool
 *
 * assign 1 cuBLAS handle / thread, use thread-local storage to manage
 *
 */
class cuBLASHandlePool {
 public:
  static const cublasHandle_t &handle() {
    static thread_local cublasHandle_t *handle_{nullptr};
    if (handle_ == nullptr) {
      handle_ = new cublasHandle_t;
      CublasSafeCall(cublasCreate(handle_));
      CublasSafeCall(cublasSetPointerMode(*handle_, CUBLAS_POINTER_MODE_HOST));
    }
    return *handle_;
  }
};
// thread_local cublasHandle_t *cuBLASHandlePool::handle_;

inline cublasOperation_t to_cublas_op(math::blas::Op cblas_op) {
  cublasOperation_t result{};
  switch (cblas_op) {
    case math::blas::Op::NoTrans:
      result = CUBLAS_OP_N;
      break;
    case math::blas::Op::Trans:
      result = CUBLAS_OP_T;
      break;
    case math::blas::Op::ConjTrans:
      result = CUBLAS_OP_C;
      break;
  }
  return result;
}

/// GEMM interface functions

template <typename T>
cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const T *alpha, const T *A, int lda, const T *B,
                          int ldb, const T *beta, T *C, int ldc);
template <>
inline cublasStatus_t cublasGemm<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
template <>
inline cublasStatus_t cublasGemm<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}
template <>
inline cublasStatus_t cublasGemm<std::complex<float>>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const std::complex<float> *alpha,
    const std::complex<float> *A, int lda, const std::complex<float> *B,
    int ldb, const std::complex<float> *beta, std::complex<float> *C, int ldc) {
  using detail::cublasPointer;
  return cublasCgemm(handle, transa, transb, m, n, k, cublasPointer(alpha),
                     cublasPointer(A), lda, cublasPointer(B), ldb,
                     cublasPointer(beta), cublasPointer(C), ldc);
}
template <>
inline cublasStatus_t cublasGemm<std::complex<double>>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const std::complex<double> *alpha,
    const std::complex<double> *A, int lda, const std::complex<double> *B,
    int ldb, const std::complex<double> *beta, std::complex<double> *C,
    int ldc) {
  using detail::cublasPointer;
  return cublasZgemm(handle, transa, transb, m, n, k, cublasPointer(alpha),
                     cublasPointer(A), lda, cublasPointer(B), ldb,
                     cublasPointer(beta), cublasPointer(C), ldc);
}

/// AXPY interface functions

template <typename T, typename Scalar>
cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const Scalar *alpha,
                          const T *x, int incx, T *y, int incy);
template <>
inline cublasStatus_t cublasAxpy<float, float>(cublasHandle_t handle, int n,
                                               const float *alpha,
                                               const float *x, int incx,
                                               float *y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, double>(cublasHandle_t handle, int n,
                                                 const double *alpha,
                                                 const double *x, int incx,
                                                 double *y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<std::complex<float>, std::complex<float>>(
    cublasHandle_t handle, int n, const std::complex<float> *alpha,
    const std::complex<float> *x, int incx, std::complex<float> *y, int incy) {
  using detail::cublasPointer;
  return cublasCaxpy(handle, n, cublasPointer(alpha), cublasPointer(x), incx,
                     cublasPointer(y), incy);
}

template <>
inline cublasStatus_t cublasAxpy<std::complex<double>, std::complex<double>>(
    cublasHandle_t handle, int n, const std::complex<double> *alpha,
    const std::complex<double> *x, int incx, std::complex<double> *y,
    int incy) {
  using detail::cublasPointer;
  return cublasZaxpy(handle, n, cublasPointer(alpha), cublasPointer(x), incx,
                     cublasPointer(y), incy);
}

template <>
inline cublasStatus_t cublasAxpy<float, int>(cublasHandle_t handle, int n,
                                             const int *alpha, const float *x,
                                             int incx, float *y, int incy) {
  const float alpha_float = float(*alpha);
  return cublasSaxpy(handle, n, &alpha_float, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<float, double>(cublasHandle_t handle, int n,
                                                const double *alpha,
                                                const float *x, int incx,
                                                float *y, int incy) {
  const float alpha_float = float(*alpha);
  return cublasSaxpy(handle, n, &alpha_float, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, int>(cublasHandle_t handle, int n,
                                              const int *alpha, const double *x,
                                              int incx, double *y, int incy) {
  const double alpha_double = double(*alpha);
  return cublasDaxpy(handle, n, &alpha_double, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, float>(cublasHandle_t handle, int n,
                                                const float *alpha,
                                                const double *x, int incx,
                                                double *y, int incy) {
  const double alpha_double = double(*alpha);
  return cublasDaxpy(handle, n, &alpha_double, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<float, detail::ComplexConjugate<void>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<void> *alpha,
    const float *x, int incx, float *y, int incy) {
  return CUBLAS_STATUS_SUCCESS;
}

template <>
inline cublasStatus_t
cublasAxpy<float, detail::ComplexConjugate<detail::ComplexNegTag>>(
    cublasHandle_t handle, int n,
    const detail::ComplexConjugate<detail::ComplexNegTag> *alpha,
    const float *x, int incx, float *y, int incy) {
  const float alpha_float = float(-1.0);
  return cublasSaxpy(handle, n, &alpha_float, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<float, detail::ComplexConjugate<int>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<int> *alpha,
    const float *x, int incx, float *y, int incy) {
  const float alpha_float = float(alpha->factor());
  return cublasSaxpy(handle, n, &alpha_float, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<float, detail::ComplexConjugate<float>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<float> *alpha,
    const float *x, int incx, float *y, int incy) {
  const float alpha_float = float(alpha->factor());
  return cublasSaxpy(handle, n, &alpha_float, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<float, detail::ComplexConjugate<double>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<double> *alpha,
    const float *x, int incx, float *y, int incy) {
  const float alpha_float = float(alpha->factor());
  return cublasSaxpy(handle, n, &alpha_float, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, detail::ComplexConjugate<void>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<void> *alpha,
    const double *x, int incx, double *y, int incy) {
  return CUBLAS_STATUS_SUCCESS;
}

template <>
inline cublasStatus_t
cublasAxpy<double, detail::ComplexConjugate<detail::ComplexNegTag>>(
    cublasHandle_t handle, int n,
    const detail::ComplexConjugate<detail::ComplexNegTag> *alpha,
    const double *x, int incx, double *y, int incy) {
  const double alpha_double = double(-1.0);
  return cublasDaxpy(handle, n, &alpha_double, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, detail::ComplexConjugate<int>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<int> *alpha,
    const double *x, int incx, double *y, int incy) {
  const double alpha_double = double(alpha->factor());
  return cublasDaxpy(handle, n, &alpha_double, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, detail::ComplexConjugate<float>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<float> *alpha,
    const double *x, int incx, double *y, int incy) {
  const double alpha_double = double(alpha->factor());
  return cublasDaxpy(handle, n, &alpha_double, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasAxpy<double, detail::ComplexConjugate<double>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<double> *alpha,
    const double *x, int incx, double *y, int incy) {
  const double alpha_double = double(alpha->factor());
  return cublasDaxpy(handle, n, &alpha_double, x, incx, y, incy);
}

/// DOT interface functions

template <typename T>
cublasStatus_t cublasDot(cublasHandle_t handle, int n, const T *x, int incx,
                         const T *y, int incy, T *result);
template <>
inline cublasStatus_t cublasDot<float>(cublasHandle_t handle, int n,
                                       const float *x, int incx, const float *y,
                                       int incy, float *result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}

template <>
inline cublasStatus_t cublasDot<double>(cublasHandle_t handle, int n,
                                        const double *x, int incx,
                                        const double *y, int incy,
                                        double *result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

/// SCAL interface function
template <typename T, typename Scalar>
cublasStatus_t cublasScal(cublasHandle_t handle, int n, const Scalar *alpha,
                          T *x, int incx);

template <>
inline cublasStatus_t cublasScal<float, float>(cublasHandle_t handle, int n,
                                               const float *alpha, float *x,
                                               int incx) {
  return cublasSscal(handle, n, alpha, x, incx);
};

template <>
inline cublasStatus_t cublasScal<double, double>(cublasHandle_t handle, int n,
                                                 const double *alpha, double *x,
                                                 int incx) {
  return cublasDscal(handle, n, alpha, x, incx);
};

template <>
inline cublasStatus_t cublasScal<float, int>(cublasHandle_t handle, int n,
                                             const int *alpha, float *x,
                                             int incx) {
  const float alpha_float = float(*alpha);
  return cublasSscal(handle, n, &alpha_float, x, incx);
};

template <>
inline cublasStatus_t cublasScal<float, double>(cublasHandle_t handle, int n,
                                                const double *alpha, float *x,
                                                int incx) {
  const float alpha_float = float(*alpha);
  return cublasSscal(handle, n, &alpha_float, x, incx);
};

//
template <>
inline cublasStatus_t cublasScal<double, int>(cublasHandle_t handle, int n,
                                              const int *alpha, double *x,
                                              int incx) {
  const double alpha_double = double(*alpha);
  return cublasDscal(handle, n, &alpha_double, x, incx);
};

template <>
inline cublasStatus_t cublasScal<double, float>(cublasHandle_t handle, int n,
                                                const float *alpha, double *x,
                                                int incx) {
  const double alpha_double = double(*alpha);
  return cublasDscal(handle, n, &alpha_double, x, incx);
};

template <>
inline cublasStatus_t cublasScal<float, detail::ComplexConjugate<void>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<void> *alpha,
    float *x, int incx) {
  return CUBLAS_STATUS_SUCCESS;
}

template <>
inline cublasStatus_t
cublasScal<float, detail::ComplexConjugate<detail::ComplexNegTag>>(
    cublasHandle_t handle, int n,
    const detail::ComplexConjugate<detail::ComplexNegTag> *alpha, float *x,
    int incx) {
  const float alpha_float = float(-1.0);
  return cublasSscal(handle, n, &alpha_float, x, incx);
}

template <>
inline cublasStatus_t cublasScal<float, detail::ComplexConjugate<int>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<int> *alpha,
    float *x, int incx) {
  const float alpha_float = float(alpha->factor());
  return cublasSscal(handle, n, &alpha_float, x, incx);
}

template <>
inline cublasStatus_t cublasScal<float, detail::ComplexConjugate<float>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<float> *alpha,
    float *x, int incx) {
  const float alpha_float = float(alpha->factor());
  return cublasSscal(handle, n, &alpha_float, x, incx);
}

template <>
inline cublasStatus_t cublasScal<float, detail::ComplexConjugate<double>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<double> *alpha,
    float *x, int incx) {
  const float alpha_float = float(alpha->factor());
  return cublasSscal(handle, n, &alpha_float, x, incx);
}

template <>
inline cublasStatus_t cublasScal<double, detail::ComplexConjugate<void>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<void> *alpha,
    double *x, int incx) {
  return CUBLAS_STATUS_SUCCESS;
}

template <>
inline cublasStatus_t
cublasScal<double, detail::ComplexConjugate<detail::ComplexNegTag>>(
    cublasHandle_t handle, int n,
    const detail::ComplexConjugate<detail::ComplexNegTag> *alpha, double *x,
    int incx) {
  const double alpha_double = double(-1.0);
  return cublasDscal(handle, n, &alpha_double, x, incx);
}

template <>
inline cublasStatus_t cublasScal<double, detail::ComplexConjugate<int>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<int> *alpha,
    double *x, int incx) {
  const double alpha_double = double(alpha->factor());
  return cublasDscal(handle, n, &alpha_double, x, incx);
}

template <>
inline cublasStatus_t cublasScal<double, detail::ComplexConjugate<float>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<float> *alpha,
    double *x, int incx) {
  const double alpha_double = double(alpha->factor());
  return cublasDscal(handle, n, &alpha_double, x, incx);
}

template <>
inline cublasStatus_t cublasScal<double, detail::ComplexConjugate<double>>(
    cublasHandle_t handle, int n, const detail::ComplexConjugate<double> *alpha,
    double *x, int incx) {
  const double alpha_double = double(alpha->factor());
  return cublasDscal(handle, n, &alpha_double, x, incx);
}

/// COPY inerface function
template <typename T>
cublasStatus_t cublasCopy(cublasHandle_t handle, int n, const T *x, int incx,
                          T *y, int incy);

template <>
inline cublasStatus_t cublasCopy(cublasHandle_t handle, int n, const float *x,
                                 int incx, float *y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasCopy(cublasHandle_t handle, int n, const double *x,
                                 int incx, double *y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}

}  // end of namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_MATH_CUBLAS_H__INCLUDED
