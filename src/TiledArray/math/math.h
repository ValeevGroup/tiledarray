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

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/math/functional.h>
#include <world/enable_if.h>
#include <linalg/cblas.h>
#include <Eigen/Core>

#ifndef TILEDARRAY_LOOP_UNWIND
#define TILEDARRAY_LOOP_UNWIND 1
#endif // TILEDARRAY_LOOP_UNWIND

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
    /// \param t The tensor object
    /// \param m The number of rows in the result matrix
    /// \param n The number of columns in the result matrix
    /// \return An m x n Eigen matrix map for \c tensor
    /// \throw TiledArray::Exception When m * n is not equal to \c tensor size
    template <typename T>
    inline Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::AutoAlign>
    eigen_map(T* t, const std::size_t m, const std::size_t n) {
      TA_ASSERT(t);
      TA_ASSERT(m > 0);
      TA_ASSERT(n > 0);
      return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::RowMajor>, Eigen::AutoAlign>(t, m, n);
    }

    /// Construct a const Eigen::Map object for a given Tensor object

    /// \tparam T The element type
    /// \param t The vector pointer
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

    /// \tparam T The element type
    /// \param t The vector pointer
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


    // BLAS _GEMM wrapper functions

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

    template <typename T, typename U>
    inline typename madness::enable_if<detail::is_numeric<T> >::type
    scale(const integer n, const T alpha, U* x) {
      eigen_map(x, n) *= alpha;
    }

    // BLAS _SCAL wrapper functions

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



    template <typename T, typename U>
    T dot(const integer n, const T* x, const U* y) {
      return eigen_map(x, n).dot(eigen_map(y, n));
    }


    // BLAS _DOT wrapper functions

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


    template <typename T>
    inline T square_norm(const integer n, const T* t) {
      return eigen_map(t, n).squaredNorm();
    }

    template <int p, typename T>
    inline T norm_2(const integer n, const T* t) {
      return eigen_map(t, n).norm();
    }

    namespace {

      /// Vector loop unwind helper class

      /// This object will unwind \c N steps of a vector operation loop.
      /// \tparam N The number of steps to unwind
      template <unsigned int N>
      struct VectorOpUnwind {

        /// Evaluate a binary operation and store the result

        /// \tparam T The left-hand argument type
        /// \tparam U The right-hand argument type
        /// \tparam V The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The left-hand argument pointer
        /// \param u The right-hand argument pointer
        /// \param v The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename V, typename Op>
        static inline void eval(const unsigned int i, const T* t, const U* u, V* v, const Op& op) {
          VectorOpUnwind<N-1>::eval(i, t, u, v, op);
          v[i+N] = op(t[i+N], u[i+N]);
        }

        /// Evaluate a unary operation and store the result

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename Op>
        static inline void eval(const unsigned int i, const T* t, U* u, const Op& op) {
          VectorOpUnwind<N-1>::eval(i, t, u, op);
          u[i+N] = op(t[i+N]);
        }

        /// Evaluate a binary operation and store the result

        /// \tparam T The left-hand argument type
        /// \tparam U The right-hand argument type
        /// \tparam V The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The left-hand argument pointer
        /// \param u The right-hand argument pointer
        /// \param v The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename V, typename Op>
        static inline void eval_to_temp(const unsigned int i, const T* t, const U* u, V* v, const Op& op) {
          VectorOpUnwind<N-1>::eval_to_temp(i, t, u, v, op);
          v[N] = op(t[i+N], u[i+N]);
        }

        /// Evaluate a unary operation and store the result

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename Op>
        static inline void eval_to_temp(const unsigned int i, const T* t, U* u, const Op& op) {
          VectorOpUnwind<N-1>::eval_to_temp(i, t, u, op);
          u[N] = op(t[i+N]);
        }

        /// Assign vector

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The assignment operation
        template <typename T, typename U, typename Op>
        static inline void assign(const unsigned int i, const T* t, U* u, const Op& op) {
          VectorOpUnwind<N-1>::assign(i, t, u, op);
          op(u[i+N], t[i+N]);
        }

        /// Assign vector

        /// \tparam T The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The result pointer
        /// \param op The assignment operation
        template <typename T, typename Op>
        static inline void assign(const unsigned int i, T* t, const Op& op) {
          VectorOpUnwind<N-1>::assign(i, t, op);
          op(t[i+N]);
        }

        /// Assign vector to temporary

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The assignment operation
        template <typename T, typename U, typename Op>
        static inline void assign_to_temp(const unsigned int i, const T* t, U* u, const Op& op) {
          VectorOpUnwind<N-1>::assign_to_temp(i, t, u, op);
          op(u[N], t[i+N]);
        }

        /// Assign vector from temporary

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The assignment operation
        template <typename T, typename U, typename Op>
        static inline void assign_from_temp(const unsigned int i, const T* t, U* u, const Op& op) {
          VectorOpUnwind<N-1>::assign_from_temp(i, t, u, op);
          op(u[i+N], t[N]);
        }
        /// Evaluate a reduction operation and store the result

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename Op>
        static inline void reduce(const unsigned int i, const T* t, U& u, const Op& op) {
          VectorOpUnwind<N-1>::reduce(i, t, u, op);
          u = op(u, t[i+N]);
        }
      }; //  struct VectorOpUnwind

      /// Vector loop unwind helper class

      /// This object will unwind \c 1 step of a vector operation loop, and
      /// terminate the loop
      template <>
      struct VectorOpUnwind<0u> {

        /// Evaluate a binary operation and store the result

        /// \tparam T The left-hand argument type
        /// \tparam U The right-hand argument type
        /// \tparam V The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The left-hand argument pointer
        /// \param u The right-hand argument pointer
        /// \param v The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename V, typename Op>
        static inline void eval(const unsigned int i, const T* t, const U* u, V* v, const Op& op) {
          v[i] = op(t[i], u[i]);
        }

        /// Evaluate a unary operation and store the result

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename Op>
        static inline void eval(const unsigned int i, const T* t, U* u, const Op& op) {
          u[i] = op(t[i]);
        }

        /// Evaluate a binary operation and store the result

        /// \tparam T The left-hand argument type
        /// \tparam U The right-hand argument type
        /// \tparam V The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The left-hand argument pointer
        /// \param u The right-hand argument pointer
        /// \param v The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename V, typename Op>
        static inline void eval_to_temp(const unsigned int i, const T* t, const U* u, V* v, const Op& op) {
          v[0u] = op(t[i], u[i]);
        }

        /// Evaluate a unary operation and store the result

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename Op>
        static inline void eval_to_temp(const unsigned int i, const T* t, U* u, const Op& op) {
          u[0u] = op(t[i]);
        }

        /// Assign vector

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The assignment operation
        template <typename T, typename U, typename Op>
        static inline void assign(const unsigned int i, const T* t, U* u, const Op& op) {
           op(u[i], t[i]);
        }

        /// Assign vector

        /// \tparam T The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The result pointer
        /// \param op The assignment operation
        template <typename T, typename Op>
        static inline void assign(const unsigned int i, T* t, const Op& op) {
           op(t[i]);
        }

        /// Assign vector to temporary

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The assignment operation
        template <typename T, typename U, typename Op>
        static inline void assign_to_temp(const unsigned int i, const T* t, U* u, const Op& op) {
           op(u[0u], t[i]);
        }

        /// Assign vector from temporary

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The assignment operation type
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The assignment operation
        template <typename T, typename U, typename Op>
        static inline void assign_from_temp(const unsigned int i, const T* t, U* u, const Op& op) {
           op(u[i], t[0u]);
        }

        /// Evaluate a reduction operation and store the result

        /// \tparam T The argument type
        /// \tparam U The result type
        /// \tparam Op The binary operation
        /// \param i The starting position of the vector offset
        /// \param t The argument pointer
        /// \param u The result pointer
        /// \param op The binary operation
        template <typename T, typename U, typename Op>
        static inline void reduce(const unsigned int i, const T* t, U& u, const Op& op) {
          u = op(u, t[i]);
        }
      }; //  struct VectorOpUnwind

    }  // namespace

    template <typename T, typename U, typename V, typename Op>
    inline void vector_op(const unsigned int n, const T* t, const U* u, V* v, const Op& op) {
      unsigned int i = 0;

#if TILEDARRAY_LOOP_UNWIND > 1
      const unsigned int nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND)
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::eval(i, t, u, v, op);
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        v[i] = op(t[i], u[i]);
    }

    template <typename T, typename U, typename Op>
    inline void vector_op(const unsigned int n, const T* t, U* u, const Op& op) {
      unsigned int i = 0;

#if TILEDARRAY_LOOP_UNWIND > 1
      const unsigned int nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND)
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::eval(i, t, u, op);
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        u[i] = op(t[i]);
    }

    template <typename T, typename U, typename Op>
    inline void vector_assign(const unsigned int n, const T* t, U* u,const Op& op) {
      unsigned int i = 0;

#if TILEDARRAY_LOOP_UNWIND > 1
      const unsigned int nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND)
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::assign(i, t, u, op);
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        op(u[i], t[i]);
    }

    template <typename T, typename Op>
    inline void vector_assign(const unsigned int n, T* t, const Op& op) {
      unsigned int i = 0;

#if TILEDARRAY_LOOP_UNWIND > 1
      const unsigned int nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND)
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::assign(i, t, op);
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        op(t[i]);
    }

    namespace detail {

      template <typename T>
      inline T abs(const T t) { return std::abs(t); }

      template <typename T>
      inline T max(const T t1, const T t2) { return std::max(t1, t2); }

      template <typename T>
      inline T min(const T t1, const T t2) { return std::min(t1, t2); }

    } // namespace

    template <typename T>
    inline T maxabs(const unsigned int n, const T* t) {
      T result = 0;
      unsigned int i = 0u;
#if TILEDARRAY_LOOP_UNWIND > 1
      const unsigned int nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
        T temp[TILEDARRAY_LOOP_UNWIND];
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::eval_to_temp(i, t,
            temp, TiledArray::math::detail::abs<T>);
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::reduce(0u, temp,
            result, TiledArray::math::detail::max<T>);
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1
      for(; i < n; ++i)
        result = std::max(result, std::abs(t[i]));
      return result;
    }

    template <typename T>
    inline T minabs(const unsigned int n, const T* t) {
      T result = std::numeric_limits<T>::max();
      unsigned int i = 0u;
#if TILEDARRAY_LOOP_UNWIND > 1
      const unsigned int nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
        T temp[TILEDARRAY_LOOP_UNWIND];
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::eval_to_temp(i, t,
            temp, TiledArray::math::detail::abs<T>);
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::reduce(0u, temp,
            result, TiledArray::math::detail::min<T>);
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1
      for(; i < n; ++i)
        result = std::min(result, std::abs(t[i]));
      return result;
    }

  }  // namespace math
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_MATH_H__INCLUDED
