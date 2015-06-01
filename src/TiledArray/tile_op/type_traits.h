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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  type_traits.h
 *  Oct 6, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TILE_OP_TYPE_TRAITS_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <TiledArray/tile_op/eval_trait.h>

namespace TiledArray {
  namespace detail {

    // Forward declaration
    template <typename, typename> class LazyArrayTile;

    /// Detect lazy evaluation tiles

    /// \c is_lazy_type evaluates to \c std::true_type when T is a tile that
    /// uses the lazy evaluation mechanism (i.e. when <tt>T != T::eval_type</tt>),
    /// otherwise it evaluates to \c std::false_type .
    /// \tparam T The tile type to test
    template <typename T>
    struct is_lazy_tile :
        public std::integral_constant<bool, ! std::is_same<T, typename eval_trait<T>::type>::value>
    { }; // struct is_lazy_tile

    /// Detect tiles used by \c ArrayEvalImpl

    /// \c is_lazy_type evaluates to \c std::true_type when T is a tile from
    /// \c ArrayEvalImpl (i.e. when <tt>T != LazyArrayTile</tt>),
    /// otherwise it evaluates to \c std::false_type .
    /// \tparam T The tile type to test
    template <typename T>
    struct is_array_tile : public std::false_type { };

    template <typename T, typename Op>
    struct is_array_tile<TiledArray::detail::LazyArrayTile<T, Op> > :
        public std::true_type
    { }; // struct is_array_tile

    /// Detect a lazy evaluation tile that are not a \c LazyArrayTile

    /// \c is_non_array_lazy_tile evaluates to \c std::true_type when T is a
    /// tile that uses the lazy evaluation mechanism (i.e. when
    /// <tt>T != T::eval_type</tt>), and not a \c LazyArrayTile , otherwise it
    /// evaluates to \c std::false_type .
    /// \tparam T The tile type to test
    template <typename T>
    struct is_non_array_lazy_tile :
        public std::integral_constant<bool, is_lazy_tile<T>::value && (! is_array_tile<T>::value)>
    { }; // struct is_non_array_lazy_tile

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_TYPE_TRAITS_H__INCLUDED
