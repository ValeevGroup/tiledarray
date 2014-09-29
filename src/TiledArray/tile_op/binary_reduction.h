/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  binary_reduction.h
 *  Apr 2, 2014
 *
 */

#ifndef TILEDARRAY_TILE_OP_BINARY_REDUCTION_H__INCLUDED
#define TILEDARRAY_TILE_OP_BINARY_REDUCTION_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>

namespace TiledArray {
  namespace math {

    /// Vector dot product tile reduction

    /// This reduction operation computes the vector dot product of a tile.
    /// \tparam Tile The tile type
    template <typename Left, typename Right>
    class DotReduction {
    public:
      // typedefs
      typedef typename Left::numeric_type result_type;
      typedef Left first_argument_type;
      typedef Right second_argument_type;

      // Reduction functions

      // Make an empty result object
      result_type operator()() const { return result_type(0); }

      // Post process the result
      const result_type& operator()(const result_type& result) const { return result; }

      // Reduce two result objects
      void operator()(result_type& result, const result_type& arg) const {
        result += arg;
      }

      // Reduce an argument pair
      void operator()(result_type& result, const first_argument_type& left,
          const second_argument_type& right) const {
        result += TiledArray::dot(left, right);
      }

    }; // class DotReduction

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_BINARY_REDUCTION_H__INCLUDED
