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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  complex.h
 *  Dec 11, 2015
 *
 */

#ifndef TILEDARRAY_SRC_TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
#define TILEDARRAY_SRC_TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED

#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace detail {


    /// Wrapper function for `std::conj`

    /// This function disables the call to `std::conj` for real values to
    /// prevent the result from being converted into a complex value.
    /// \tparam R A real scalar type
    /// \param r The real scalar
    /// \return `r`
    template <typename R,
        typename std::enable_if<! is_complex<R>::value>::type* = nullptr>
    TILEDARRAY_FORCE_INLINE R conj(const R r) {
      return r;
    }

    /// Wrapper function for std::conj

    /// \tparam R The scalar type
    /// \param z The complex scalar
    /// \return The complex conjugate of `z`
    template <typename R>
    TILEDARRAY_FORCE_INLINE std::complex<R> conj(const std::complex<R> z) {
      return std::conj(z);
    }

    /// Wrapper function for `std::norm`

    /// This function disables the call to `std::conj` for real values to
    /// prevent the result from being converted into a complex value.
    /// \tparam R A real scalar type
    /// \param r The real scalar
    /// \return `r`
    template <typename R,
        typename std::enable_if<! is_complex<R>::value>::type* = nullptr>
    TILEDARRAY_FORCE_INLINE R norm(const R r) {
      return r * r;
    }

    /// Compute the norm of a complex number `z`

    /// \f[
    ///   {\rm norm}(z) = zz^* = {\rm Re}(z)^2 + {\rm Im}(z)^2
    /// \f]
    /// \tparam R The scalar type
    /// \param z The complex scalar
    /// \return The complex conjugate of `z`
    template <typename R>
    TILEDARRAY_FORCE_INLINE R norm(const std::complex<R> z) {
      const R real = z.real();
      const R imag = z.imag();
      return real * real + imag * imag;
    }

    template <typename S>
    class ComplexConjugate {
      S factor_;

    public:
      ComplexConjugate(const S factor) : factor_(factor) { }

      TILEDARRAY_FORCE_INLINE S factor() const { return factor_; }

      TILEDARRAY_FORCE_INLINE ComplexConjugate<S> operator-() const {
        return ComplexConjugate<S>(-factor_);
      }

      friend std::ostream& operator<<(std::ostream& os, const ComplexConjugate<S>& cc) {
        os << "conj()] [" << cc.factor_;
        return os;
      }
    };


    struct ComplexNegTag { };

    template <>
    class ComplexConjugate<void> {
    public:

      inline ComplexConjugate<ComplexNegTag> operator-() const;

      friend std::ostream&
      operator<<(std::ostream& os, const ComplexConjugate<void>& cc) {
        os << "conj()";
        return os;
      }
    };

    template <>
    class ComplexConjugate<ComplexNegTag> {
    public:

      inline ComplexConjugate<void> operator-() const;

      friend std::ostream&
      operator<<(std::ostream& os, const ComplexConjugate<ComplexNegTag>& cc) {
        os << "conj()] [-1";
        return os;
      }
    };


    inline ComplexConjugate<void>
    ComplexConjugate<ComplexNegTag>::operator-() const {
      return ComplexConjugate<void>();
    }

    inline ComplexConjugate<ComplexNegTag>
    ComplexConjugate<void>::operator-() const {
      return ComplexConjugate<ComplexNegTag>();
    }

    template <typename S>
    struct is_numeric<ComplexConjugate<S> > : public std::true_type { };

    /// ComplexConjugate operator factory function

    /// \tparam S The scalar type
    /// \param factor The scaling factor
    /// \return A scaling complex conjugate operator
    template <typename S>
    inline ComplexConjugate<S> conj_op(const S factor) {
      return ComplexConjugate<S>(factor);
    }

    /// ComplexConjugate operator factory function

    /// \return A complex conjugate operator
    inline ComplexConjugate<void> conj_op() {
      return ComplexConjugate<void>();
    }

    template <typename L, typename R>
    TILEDARRAY_FORCE_INLINE auto
    operator*(const L value, const ComplexConjugate<R> op)
    { return TiledArray::detail::conj(value) * op.factor(); }

    template <typename L>
    TILEDARRAY_FORCE_INLINE auto
    operator*(const L value, const ComplexConjugate<void>&)
    { return TiledArray::detail::conj(value); }

    template <typename L>
    TILEDARRAY_FORCE_INLINE auto
    operator*(const L value, const ComplexConjugate<ComplexNegTag>&)
    { return -TiledArray::detail::conj(value); }


    template <typename L, typename R>
    TILEDARRAY_FORCE_INLINE auto
    operator*(const ComplexConjugate<L> op, const R value)
    { return TiledArray::detail::conj(value) * op.factor(); }

    template <typename R>
    TILEDARRAY_FORCE_INLINE auto
    operator*(const ComplexConjugate<void>, const R value)
    { return TiledArray::detail::conj(value); }

    template <typename R>
    TILEDARRAY_FORCE_INLINE auto
    operator*(const ComplexConjugate<ComplexNegTag>, const R value)
    { return -TiledArray::detail::conj(value); }


    template <typename L, typename R,
        typename std::enable_if<! std::is_void<R>::value>::type* = nullptr>
    TILEDARRAY_FORCE_INLINE L&
    operator*=(L& value, const ComplexConjugate<R> op) {
      value = TiledArray::detail::conj(value) * op.factor();
      return value;
    }

    template <typename L>
    TILEDARRAY_FORCE_INLINE L&
    operator*=(L& value, const ComplexConjugate<void>&) {
      value = TiledArray::detail::conj(value);
      return value;
    }

    template <typename L>
    TILEDARRAY_FORCE_INLINE L&
    operator*=(L& value, const ComplexConjugate<ComplexNegTag>&) {
      value = -TiledArray::detail::conj(value);
      return value;
    }

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_SRC_TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
