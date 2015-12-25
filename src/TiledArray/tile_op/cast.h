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
 *  cast.h
 *  Dec 24, 2015
 *
 */

#ifndef TILEDARRAY_TILE_OP_CAST_H__INCLUDED
#define TILEDARRAY_TILE_OP_CAST_H__INCLUDED

#include <type_traits>

namespace TiledArray {

  // Forward declarations
  namespace detail {
    template <typename, typename> class LazyArrayTile;
  } // namespace detail

  template <typename Result, typename Tile, typename Enabler = void>
  class Cast {
  public:

    typedef Result result_type;
    typedef Tile tile_type;

    result_type operator()(const tile_type& arg) const {
      return arg;
    }

  }; // class Cast

  template <typename Tile, typename Op>
  class Cast<Tile, TiledArray::detail::LazyArrayTile<Tile, Op>, void> {
  public:

    typedef Tile result_type;
    typedef TiledArray::detail::LazyArrayTile<Tile, Op> tile_type;

    result_type operator()(const tile_type& arg) const {
      return arg;
    }

  }; // class Cast

  template <typename Result, typename Tile, typename Op>
  class Cast<Result, TiledArray::detail::LazyArrayTile<Tile, Op>,
      typename std::enable_if<! std::is_same<Result, Tile>::value>::type> :
      public Cast<Result, Tile>
  {
  public:
    typedef Result result_type;
    typedef TiledArray::detail::LazyArrayTile<Tile, Op> tile_type;

    result_type operator()(const tile_type& arg) const {
      return Cast<Result, Tile>::operator()(arg);
    }

  }; // class Cast

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_CAST_H__INCLUDED
