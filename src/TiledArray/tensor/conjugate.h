/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  conjugate.h
 *  Oct 14, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
#define TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED

#include <complex>
#include <TiledArray/type_traits.h>

namespace TiledArray {

  // Forward declaration

  template <typename T>
  class Conjugate;

  namespace detail {

    /// Trait to detect floating point and complex types
    template <typename T>
    struct is_float_or_complex :
        public std::integral_constant<bool, std::is_floating_point<T>::value ||
        is_complex<T>::value>
    { };

    template <typename T>
    struct is_numeric<Conjugate<T> > : public std::true_type { };

  } // namespace detail

  /// Complex conjugate operator

  /// \tparam T Floating point scalar or complex type
  /// This object defines a complex conjugate operator that may be applied to
  /// arithmetic types via the \c * operator
  template <typename T>
  class Conjugate {

    T factor_; ///< Scaling factor

  public:
    Conjugate(const T factor) : factor_(scalar) { }

    T scalar() const { return factor_; }

    template <typename A,
        typename std::enable_if<
            ! detail::is_complex<A>::value
        >::type* = nullptr>
    A apply(const A arg) const {
      return arg * factor_;
    }


    template <typename A>
    std::complex<A> apply(const std::complex<A> arg) const {
      return std::conj(arg) * factor_;
    }

    template <typename A,
        typename std::enable_if<
            ! detail::is_complex<A>::value
        >::type* = nullptr>
    A& inplace_apply(A& arg) const {
      arg *= factor_;
      return arg;
    }


    template <typename A>
    std::complex<A>& inplace_apply(std::complex<A>& arg) const {
      arg = std::conj(arg) * factor_;
      return arg;
    }

  }; // Conjugate

  /// Complex conjugate operator

  /// This object defines a complex conjugate operator that may be applied to
  /// arithmetic types via the \c * operator
  template <>
  class Conjugate<void> {
  public:

    template <typename A,
        typename std::enable_if<
            ! detail::is_complex<A>::value
        >::type* = nullptr>
    A apply(const A arg) const {
      return arg;
    }


    template <typename A>
    std::complex<A> apply(const std::complex<A> arg) const {
      return std::conj(arg);
    }

    template <typename A,
        typename std::enable_if<
            ! detail::is_complex<A>::value
        >::type* = nullptr>
    A& inplace_apply(A& arg) const {
      return arg;
    }


    template <typename A>
    std::complex<A>& inplace_apply(const std::complex<A> arg) const {
      arg.imag(-arg.imag());
      return arg;
    }

  }; // class Conjugate<void>

  inline Conjugate<void> conj() { return Conjugate<void>(); }

  template <typename T>
  inline Conjugate<T> conj(const T scalar) { return Conjugate<T>(scalar); }

  template <typename L, typename R>
  inline auto operator*(const Conjugate<L> op, const R value) ->
      decltype(op.apply(value))
  { return op.apply(value); }

  template <typename L, typename R>
  inline auto operator*(const L value, const Conjugate<R> op) ->
      decltype(op.apply(value))
  { return op.apply(value); }

  template <typename L, typename R>
  inline auto operator*=(L& value, const Conjugate<R> op) ->
      decltype(op.implace_apply(value))
  { return op.inplace_apply(value); }


} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
