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
 *  truncate.h
 *  Apr 15, 2015
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_TRUNCATE_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_TRUNCATE_H__INCLUDED

#include <TiledArray/conversions/foreach.h>

namespace TiledArray {

  /// Forward declarations
  template <typename, unsigned int, typename, typename>
  class Array;
  class DensePolicy;
  class SparsePolicy;

  /// Truncate a dense Array

  /// This is a no op
  /// \tparam T Element type of the array
  /// \tparam DIM Dimension of the array
  /// \tparam Tile The tile type of the array
  /// \param array The array object to be truncated
  /// \return \c array
  template <typename T, unsigned int DIM, typename Tile>
  inline Array<T, DIM, Tile, DensePolicy>
  truncate(Array<T, DIM, Tile, DensePolicy>& array) { return array; }

  /// Truncate a sparse Array

  /// \tparam T Element type of the array
  /// \tparam DIM Dimension of the array
  /// \tparam Tile The tile type of the array
  /// \param array The array object to be truncated
  /// \return A truncated copy of \c array
  template <typename T, unsigned int DIM, typename Tile>
  inline Array<T, DIM, Tile, SparsePolicy>
  truncate(Array<T, DIM, Tile, SparsePolicy>& array) {
    typedef typename Array<T, DIM, Tile, SparsePolicy>::value_type value_type;
    Array<T, DIM, Tile, SparsePolicy> result =
        foreach(array, [] (value_type& result_tile, const value_type& arg_tile) {
          typename detail::scalar_type<value_type>::type norm = arg_tile.norm();
          result_tile = arg_tile; // Assume this is shallow copy
          return norm;
        });
    return result;
  }

} // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_TRUNCATE_H__INCLUDED
