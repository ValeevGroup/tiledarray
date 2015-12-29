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
 *  Edward Valeev, Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  clone.h
 *  Dec 27, 2015
 *
 */

#ifndef TILEDARRAY_TILE_OP_CLONE_H__INCLUDED
#define TILEDARRAY_TILE_OP_CLONE_H__INCLUDED

#include <type_traits>
#include <TiledArray/tile_op/cast.h>

namespace TiledArray {

  template <typename Result, typename Tile>
  class Clone {
  public:

    typedef Result result_type;
    typedef Tile tile_type;

    template <typename Tile_, typename = typename std::enable_if<std::is_same<Tile,Tile_>::value && not std::is_same<Result,Tile>::value>::type >
    result_type operator()(const Tile_& arg) const {
      Cast<Result,Tile> caster;
      return caster(arg);
    }
  };

  template <typename Tile>
  class Clone<Tile,Tile> {
  public:

    typedef Tile result_type;
    typedef Tile tile_type;

    result_type operator()(const Tile& arg) const {
      using TiledArray::clone;
      return clone(arg);
    }

  }; // class Clone

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_CAST_H__INCLUDED
