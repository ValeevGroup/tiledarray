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
 *  vector_op.h
 *  Nov 17, 2013
 *
 */

#ifndef TILEDARRAY_MATH_VECTOR_OP_H__INCLUDED
#define TILEDARRAY_MATH_VECTOR_OP_H__INCLUDED

#ifndef TILEDARRAY_LOOP_UNWIND
#define TILEDARRAY_LOOP_UNWIND 8
#endif // TILEDARRAY_LOOP_UNWIND

#include <TiledArray/math/math.h>
#include <TiledArray/math/eigen.h>
#include <stdint.h>

// Add macro TILEDARRAY_FORCE_INLINE which does as the name implies.
#if (defined _MSC_VER) || (defined __INTEL_COMPILER)

#define TILEDARRAY_FORCE_INLINE __forceinline

#elif(__clang__)

#define TILEDARRAY_FORCE_INLINE __attribute__((always_inline)) inline

#elif defined(__GNUC__)

#if (__GNUC__ >= 4)
#define TILEDARRAY_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define TILEDARRAY_FORCE_INLINE inline
#endif // (__GNUC__ >= 4)

#else

#define TILEDARRAY_FORCE_INLINE inline

#endif


namespace TiledArray {
  namespace math {

    typedef uint_fast32_t uint_type;

    /// Vector loop unwind helper class

    /// This object will unwind \c N steps of a vector operation loop.
    /// \tparam N The number of steps to unwind
    template <uint_type N>
    struct VectorOpUnwind {

      // Binary operations

      template <typename T, typename U, typename V, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      binary_eval(register const T* const t, register const U* const u,
          register V* const v, const Op& op)
      {
        VectorOpUnwind<N-1>::binary_eval(t, u, v, op);
        v[N] = op(t[N], u[N]);
      }

      template <typename T, typename U, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      binary_eval(register const T* t, register U* u, const Op& op) {
        VectorOpUnwind<N-1>::binary_eval(t, u, op);
        op(u[N], t[N]);
      }

      // Unary operations

      template <typename T, typename U, typename Op>
      static TILEDARRAY_FORCE_INLINE void
      unary_eval(register const T* t, register U* u, const Op& op) {
        VectorOpUnwind<N-1>::unary_eval(t, u, op);
        u[N] = op(t[N]);
      }

      template <typename T, typename Op>
      static TILEDARRAY_FORCE_INLINE void unary_eval(T* t, const Op& op) {
        VectorOpUnwind<N-1>::unary_eval(t, op);
        op(t[N]);
      }

      // Reduce operations

      template <typename T, typename U, typename Op>
      static TILEDARRAY_FORCE_INLINE void reduce(const T* t, U& u, const Op& op) {
        VectorOpUnwind<N-1>::reduce(t, u, op);
        u = op(u, t[N]);
      }
    }; //  struct VectorOpUnwind

    /// Vector loop unwind helper class

    /// This object will unwind \c 1 step of a vector operation loop, and
    /// terminate the loop
    template <>
    struct VectorOpUnwind<0ul> {

      template <typename T, typename U, typename V, typename Op>
      static TILEDARRAY_FORCE_INLINE void binary_eval(register const T* const t,
          register const U* const u, register V* const v, const Op& op)
      {
        v[0ul] = op(t[0ul], u[0ul]);
      }

      template <typename T, typename U, typename Op>
      static TILEDARRAY_FORCE_INLINE void binary_eval(register const T* t, register U* u, const Op& op) {
        op(u[0ul], t[0ul]);
      }

      // Unary operations

      template <typename T, typename U, typename Op>
      static TILEDARRAY_FORCE_INLINE void unary_eval(register const T* t, register U* u, const Op& op) {
        u[0ul] = op(t[0ul]);
      }

      template <typename T, typename Op>
      static TILEDARRAY_FORCE_INLINE void unary_eval(register T* t, const Op& op) {
        op(t[0ul]);
      }

      // Reduction operations

      template <typename T, typename U, typename Op>
      static TILEDARRAY_FORCE_INLINE void reduce(const T* t, U& u, const Op& op) {
        u = op(u, t[0ul]);
      }
    }; //  struct VectorOpUnwind


    template <typename T, typename U, typename V, typename Op>
    inline void binary_vector_op(const uint_type n, register const T* const t,
        register const U* const u, register V* const v, const Op& op)
    {
      uint_type i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1
      {
        const uint_type nx = n - (n % TILEDARRAY_LOOP_UNWIND);
        for(; i < nx; i += TILEDARRAY_LOOP_UNWIND)
          VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1ul>::binary_eval(t + i, u + i, v + i, op);
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        v[i] = op(t[i], u[i]);
    }

    template <typename T, typename U, typename Op>
    inline void binary_vector_op(const uint_type n, register const T* const t, register U* const u, const Op& op) {
      uint_type i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1
      const uint_type nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1ul>::binary_eval(t + i, u + i, op);
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        op(u[i], t[i]);
    }

    template <typename T, typename U, typename Op>
    inline void unary_vector_op(const uint_type n, register const T* const t, register U* const u, const Op& op) {
      uint_type i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1
      const uint_type nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::unary_eval(t + i, u + i, op);
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        u[i] = op(t[i]);
    }

    template <typename T, typename Op>
    inline void unary_vector_op(const uint_type n, register T* const t, const Op& op) {
      uint_type i = 0ul;

#if TILEDARRAY_LOOP_UNWIND > 1
      const uint_type nx = n - (n % TILEDARRAY_LOOP_UNWIND);
      for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
        VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::unary_eval(t + i, op);
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1

      for(; i < n; ++i)
        op(t[i]);
    }

    template <typename T>
    inline T maxabs(const uint_type n, register const T* const t) {
      T result = 0;
      uint_type i = 0ul;
#if TILEDARRAY_LOOP_UNWIND > 1
      {
        const uint_type nx = n - (n % TILEDARRAY_LOOP_UNWIND);
        T temp[TILEDARRAY_LOOP_UNWIND];
        for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
          VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::unary_eval(t + i, temp,
              TiledArray::math::abs<T>);
          VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::reduce(temp, result,
              TiledArray::math::max<T>);
        }
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1
      for(; i < n; ++i)
        result = std::max(result, std::abs(t[i]));
      return result;
    }

    template <typename T>
    inline T minabs(const uint_type n, const T* t) {
      T result = std::numeric_limits<T>::max();
      uint_type i = 0ul;
#if TILEDARRAY_LOOP_UNWIND > 1
      {
        const uint_type nx = n - (n % TILEDARRAY_LOOP_UNWIND);
        T temp[TILEDARRAY_LOOP_UNWIND];
        for(; i < nx; i += TILEDARRAY_LOOP_UNWIND) {
          VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::unary_eval(t + i, temp,
              TiledArray::math::abs<T>);
          VectorOpUnwind<TILEDARRAY_LOOP_UNWIND - 1>::reduce(temp, result,
              TiledArray::math::min<T>);
        }
      }
#endif // TILEDARRAY_LOOP_UNWIND > 1
      for(; i < n; ++i)
        result = std::min(result, std::abs(t[i]));
      return result;
    }

    template <typename T>
    inline T square_norm(const uint_type n, const T* t) {
      return eigen_map(t, n).squaredNorm();
    }

    template <int p, typename T>
    inline T norm_2(const uint_type n, const T* t) {
      return eigen_map(t, n).norm();
    }

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_MATH_VECTOR_OP_H__INCLUDED
