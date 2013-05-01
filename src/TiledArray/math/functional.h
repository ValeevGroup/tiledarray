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

#ifndef TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED
#define TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED

namespace TiledArray {
  namespace detail {

    /// Square function object

    /// \tparam T argument and result type
    template <typename T>
    struct Square {
      typedef T result_type;
      typedef T argument_type;

      /// Square \c t

      /// \param t The value to be squared
      /// \return t * t
      result_type operator()(argument_type t) const { return t * t; }

    }; // class Square


    /// generalization of std::multiplies, but multiplying T by U yielding V
    /// \tparam T argument1 type
    /// \tparam U argument2 type
    /// \tparam V result type
    template <typename T, typename U, typename V>
    struct multiplies {

      typedef T first_argument_type;
      typedef U second_argument_type;
      typedef V result_type;

      /// Returns \c t * \c scale = \c v
      /// \param t first factor
      /// \param scale second factor
      /// \return \c t * \c scale
      typename madness::enable_if<TiledArray::detail::is_numeric<U>, V>::type
      operator()(T t, U scale) const { return t * scale; }

    }; // class multiplies

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED
