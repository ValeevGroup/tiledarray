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

#include <TiledArray/type_traits.h>

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
      return static_cast<result_type>(arg);
    }

  }; // class Cast

  template <typename Tile>
  class Cast<typename TiledArray::eval_trait<Tile>::type, Tile,
      typename std::enable_if<is_lazy_tile<Tile>::value>::type>
  {
  public:

    typedef typename TiledArray::eval_trait<Tile>::type result_type;
    typedef Tile tile_type;

    result_type operator()(const tile_type& arg) const {
      return static_cast<typename TiledArray::eval_trait<Tile>::type>(arg);
    }

  }; // class Cast

  template <typename Result, typename Tile>
  class Cast<Result, Tile,
      typename std::enable_if<
          is_lazy_tile<Tile>::value &&
          ! std::is_same<Result, typename TiledArray::eval_trait<Tile>::type>::value
      >::type> :
      public Cast<Result, typename TiledArray::eval_trait<Tile>::type>
  {
  public:
    typedef Result result_type;
    typedef Tile tile_type;

    result_type operator()(const tile_type& arg) const {
      return Cast<Result, typename TiledArray::eval_trait<Tile>::type>::operator()(
          static_cast<typename TiledArray::eval_trait<Tile>::type>(arg));
    }

  }; // class Cast

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_CAST_H__INCLUDED
