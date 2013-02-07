/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_FUNCTIONAL_H__INCLUDED
#define TILEDARRAY_FUNCTIONAL_H__INCLUDED

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

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_FUNCTIONAL_H__INCLUDED
