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

#ifndef TILEDARRAY_MATH_MATH_H__INCLUDED
#define TILEDARRAY_MATH_MATH_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/math/functional.h>
#include <world/enable_if.h>
#include <Eigen/Core>

namespace TiledArray {
  namespace math {

    /// Construct a const Eigen::Map object for a given Tensor object

    /// \tparam T The element type
    /// \param t The buffer pointer
    /// \param m The number of rows in the result matrix
    /// \param n The number of columns in the result matrix
    /// \return An m x n Eigen matrix map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(const T* t, const std::size_t m, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(m > 0);
      TA_ASSERT(n > 0);
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(t, m, n);
    }

    /// Construct an Eigen::Map object for a given Tensor object

    /// \tparam T The tensor element type
    /// \tparam A The tensor allocator type
    /// \param tensor The tensor object
    /// \param m The number of rows in the result matrix
    /// \param n The number of columns in the result matrix
    /// \return An m x n Eigen matrix map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T, typename A>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(T* t, const std::size_t m, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(m > 0);
      TA_ASSERT(n > 0);
      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(t, m, n);
    }

    /// Construct a const Eigen::Map object for a given Tensor object

    /// \tparam T The tensor element type
    /// \tparam A The tensor allocator type
    /// \param tensor The tensor object
    /// \param n The number of elements in the result matrix
    /// \return An n element Eigen vector map for \c tensor
    /// \throw TiledArray::Exception When n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>
    eigen_map(const T* t, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(n > 0);
      return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>(t, n);
    }

    /// Construct an Eigen::Map object for a given Tensor object

    /// \tparam T The tensor element type
    /// \tparam A The tensor allocator type
    /// \param tensor The tensor object
    /// \param n The number of elements in the result matrix
    /// \return An n element Eigen vector map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>
    eigen_map(T* t, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(n > 0);
      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::AutoAlign>(t, n);
    }

    template <typename T>
    inline void gemm(const integer m, const integer n, const integer k, const T alpha, const T* a, const T* b, T* c) {
      eigen_map(c, m, n).noalias() += alpha * (eigen_map(a, m, k) * eigen_map(b, k, n));
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

#endif // TILEDARRAY_HAS_BLAS

    template <typename T, typename U>
    inline typename madness::enable_if<detail::is_numeric<T> >::type
    scale(const integer n, const T alpha, U* x) {
      eigen_map(x, n) *= alpha;
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
      return eigen_map(x, n).dot(eigen_map(y, n));
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


    template <typename T, typename U>
    inline void add_to(const integer n, T* t, const U* u) {
      eigen_map(t, n) += eigen_map(u, n);
    }

    template <typename T, typename U>
    inline void subt_to(const integer n, T* t, const U* u) {
      eigen_map(t, n) -= eigen_map(u, n);
    }

    template <typename T, typename U>
    inline void mult_to(const integer n, T* t, const U* u) {
      eigen_map(t, n) *= eigen_map(u, n);
    }

    template <typename T, typename U>
    inline void add_const(const integer n, T* t, const U& u) {
      eigen_map(t, n).array() += u;
    }

    template <typename T, typename U>
    inline void subt_const(const integer n, T* t, const U& u) {
      eigen_map(t, n).array() -= u;
    }


    template <typename T>
    inline T square_norm(const integer n, const T* t) {
      return eigen_map(t, n).squaredNorm();
    }

    template <int p, typename T>
    inline T norm_2(const integer n, const T* t) {
      return eigen_map(t, n).norm();
    }

    namespace detail {

      template <typename Op>
      struct BinaryVectorOpHelper {
        template <typename T, typename U, typename V>
        static void eval(const integer n, const T* t, const U* u, V* v, const Op& op) {
          eigen_map(v, n) = eigen_map(t, n).binaryExpr(eigen_map(u, n), op);
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U, typename V>
      struct BinaryVectorOpHelper<TiledArray::detail::Plus<T, U, V> > {
        static void eval(const integer n, const T* t, const U* u, V* v, const TiledArray::detail::Plus<T, U, V>&) {
          eigen_map(v, n) = eigen_map(t, n) + eigen_map(u, n);
        }
      }; //  struct VectorOpHelper

      template <typename T>
      struct BinaryVectorOpHelper<std::plus<T> > {
        static void eval(const integer n, const T* t, const T* u, T* v, const std::plus<T>&) {
          eigen_map(v, n) = eigen_map(t, n) + eigen_map(u, n);
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U, typename V>
      struct BinaryVectorOpHelper<TiledArray::detail::ScalPlus<T, U, V> > {
        static void eval(const integer n, const T* t, const U* u, V* v, const TiledArray::detail::ScalPlus<T, U, V>& op) {
          eigen_map(v, n) = op.factor() * (eigen_map(t, n) + eigen_map(u, n));
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U, typename V>
      struct BinaryVectorOpHelper<TiledArray::detail::Minus<T, U, V> > {
        static void eval(const integer n, const T* t, const U* u, V* v, const TiledArray::detail::Minus<T, U, V>&) {
          eigen_map(v, n) = eigen_map(t, n) - eigen_map(u, n);
        }
      }; //  struct VectorOpHelper


      template <typename T>
      struct BinaryVectorOpHelper<std::minus<T> > {
        static void eval(const integer n, const T* t, const T* u, T* v, const std::minus<T>&) {
          eigen_map(v, n) = eigen_map(t, n) - eigen_map(u, n);
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U, typename V>
      struct BinaryVectorOpHelper<TiledArray::detail::ScalMinus<T, U, V> > {
        static void eval(const integer n, const T* t, const U* u, V* v, const TiledArray::detail::ScalMinus<T, U, V>& op) {
          eigen_map(v, n) = op.factor() * (eigen_map(t, n) - eigen_map(u, n));
        }
      }; //  struct VectorOpHelper


      template <typename T, typename U, typename V>
      struct BinaryVectorOpHelper<TiledArray::detail::Multiplies<T, U, V> > {
        static void eval(const integer n, const T* t, const U* u, V* v, const TiledArray::detail::Multiplies<T, U, V>&) {
          eigen_map(v, n).array() =
              eigen_map(t, n).array() * eigen_map(u, n).array();
        }
      }; //  struct VectorOpHelper

      template <typename T>
      struct BinaryVectorOpHelper<std::multiplies<T> > {
        static void eval(const integer n, const T* t, const T* u, T* v, const std::multiplies<T>&) {
          eigen_map(v, n).array() = eigen_map(t, n).array() * eigen_map(u, n).array();
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U, typename V>
      struct BinaryVectorOpHelper<TiledArray::detail::ScalMultiplies<T, U, V> > {
        static void eval(const integer n, const T* t, const U* u, V* v, const TiledArray::detail::ScalMultiplies<T, U, V>& op) {
          eigen_map(v, n).array() =
              op.factor() * (eigen_map(t, n).array() * eigen_map(u, n).array());
        }
      }; //  struct VectorOpHelper

      template <typename Op>
      struct UnaryVectorOpHelper {
        template <typename T, typename U>
        static void eval(const integer n, const T* t, U* u, const Op& op) {
          eigen_map(u, n) = eigen_map(t, n).unaryExpr(op);
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U>
      struct UnaryVectorOpHelper<TiledArray::detail::Negate<T, U> > {
        static void eval(const integer n, const T* t, U* u, const TiledArray::detail::Negate<T, U>&) {
          eigen_map(u, n) = -(eigen_map(t, n));
        }
      }; //  struct VectorOpHelper


      template <typename T>
      struct BinaryVectorOpHelper<std::negate<T> > {
        static void eval(const integer n, const T* t, T* u, const std::negate<T>&) {
          eigen_map(u, n) = -(eigen_map(t, n));
        }
      }; //  struct VectorOpHelper

      template <typename T, typename U>
      struct UnaryVectorOpHelper<TiledArray::detail::ScalNegate<T, U> > {
        static void eval(const integer n, const T* t, U* u, const TiledArray::detail::ScalNegate<T, U>& op) {
          eigen_map(u, n) = op.factor() * eigen_map(t, n);
        }
      }; //  struct VectorOpHelper

    }  // namespace detail

    template <typename T, typename U, typename V, typename Op>
    inline void vector_op(const integer n, const T* t, const U* u, V* v, const Op& op) {
      detail::BinaryVectorOpHelper<Op>::eval(n, t, u, v, op);
    }

    template <typename T, typename U, typename Op>
    inline void vector_op(const integer n, const T* t, U* u, const Op& op) {
      detail::UnaryVectorOpHelper<Op>::eval(n, t, u, op);
    }

    template <typename T>
    inline T maxabs(const integer n, const T* t) {
      T temp = 0;
      const integer n4 = n - (n % 4);
      integer i = 0;
      for(; i < n4; i += 4) {
        const T abs0 = std::abs(t[i]);
        const T abs1 = std::abs(t[i+1]);
        const T abs2 = std::abs(t[i+2]);
        const T abs3 = std::abs(t[i+3]);

        if(temp < abs0) temp = abs0;
        if(temp < abs1) temp = abs1;
        if(temp < abs2) temp = abs2;
        if(temp < abs3) temp = abs3;
      }
      for(; i < n; ++i) {
        const T abs = std::abs(t[i]);
        if(temp < abs) temp = abs;
      }
      return temp;
    }

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_MATH_H__INCLUDED
