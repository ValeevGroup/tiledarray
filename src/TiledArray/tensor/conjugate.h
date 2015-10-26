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
    static_assert(detail::is_float_or_complex<T>::value,
        "Conjugate scalar type must be a floating point or complex type.");

    T scalar_;

  public:
    Conjugate() : scalar_(1) { }
    Conjugate(const T scalar) : scalar_(scalar) { }

    // Compiler generated functions
    Conjugate(const Conjugate&) = default;
    Conjugate(Conjugate&&) = default;
    ~Conjugate() = default;
    Conjugate& operator=(const Conjugate&) = default;
    Conjugate& operator=(Conjugate&&) = default;

    T scalar() const { return scalar_; }

    template <typename U,
        typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
    auto operator()(const U value) const -> decltype(value * std::declval<T>())
    { return value * scalar_; }


    template <typename U,
        typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
    auto operator()(const std::complex<U> value) const
        -> decltype(std::conj(value) * std::declval<T>())
    { return std::conj(value) * scalar_; }

    template <typename U,
        typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
    auto operator()(const Conjugate<U> conj) const
        -> decltype(conj.scalar_ * std::declval<T>())
    { return conj.scalar_ * scalar_; }


    template <typename U,
        typename std::enable_if<std::is_floating_point<U>::value>::type* = nullptr>
    auto operator()(const Conjugate<std::complex<U> > conj) const
        -> decltype(std::conj(conj.scalar_) * std::declval<T>())
    { return std::conj(conj.scalar_) * scalar_; }

  }; // Conjugate

  inline Conjugate<double> conj() { return Conjugate<double>(); }

  template <typename T>
  inline Conjugate<T> conj() { return Conjugate<T>(); }

  template <typename T>
  inline Conjugate<T> conj(const T scalar) { return Conjugate<T>(scalar); }

  template <typename L, typename R,
      typename std::enable_if<detail::is_float_or_complex<R>::value>::type* = nullptr>
  inline auto operator*(const Conjugate<L> left, const R right) ->
      decltype(TiledArray::conj(left.scalar() * right))
  { return TiledArray::conj(left.scalar() * right); }

  template <typename L, typename R,
      typename std::enable_if<detail::is_float_or_complex<L>::value>::type* = nullptr>
  inline auto operator*(const L left, const Conjugate<R> right) ->
      decltype(right(left))
  { return right(left); }


  template <typename L, typename R,
      typename std::enable_if<detail::is_float_or_complex<L>::value>::type* = nullptr>
  inline L& operator*=(L& left, const Conjugate<R> right) {
    left = right(left);
    return left;
  }

  template <typename L, typename R>
  inline auto operator*(const Conjugate<L> left, const Conjugate<R> right) ->
      decltype(left(right))
  { return left(right); }


} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
