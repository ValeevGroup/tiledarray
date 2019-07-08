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
  template <typename, typename> class DistArray;
  class DensePolicy;
  class SparsePolicy;

  /// Truncate a dense Array

  /// This is a no-op
  /// \tparam Tile The tile type of \c array
  /// \tparam Policy The policy type of \c array
  /// \param[in,out] array The array object to be truncated
  template <typename Tile, typename Policy>
  inline
  std::enable_if_t<is_dense_v<Policy>, void>
  truncate(DistArray<Tile, Policy>& array) { }

  /// Truncate a sparse Array

  /// \tparam Tile The tile type of \c array
  /// \tparam Policy The policy type of \c array
  /// \param[in,out] array The array object to be truncated
  template <typename Tile, typename Policy>
  inline
  std::enable_if_t<!is_dense_v<Policy>, void>
  truncate(DistArray<Tile, Policy>& array) {
    typedef typename DistArray<Tile, Policy>::value_type value_type;
    array =
        foreach(array, [] (value_type& result_tile, const value_type& arg_tile) {
          typename detail::scalar_type<value_type>::type arg_tile_norm = norm(arg_tile);
          result_tile = arg_tile; // Assume this is shallow copy
          return arg_tile_norm;
        });
  }

} // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_TRUNCATE_H__INCLUDED
