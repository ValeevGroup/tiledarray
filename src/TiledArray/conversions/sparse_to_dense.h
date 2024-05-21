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
 *  Drew Lewis
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_to_dense.h
 *  Feb 02, 2015
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_SPARSE_TO_DENSE_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_SPARSE_TO_DENSE_H__INCLUDED

#include "../dist_array.h"

namespace TiledArray {

template <typename Tile, typename ResultPolicy = DensePolicy,
          typename ArgPolicy>
inline std::enable_if_t<is_dense_v<ResultPolicy> && !is_dense_v<ArgPolicy>,
                        DistArray<Tile, ResultPolicy>>
to_dense(DistArray<Tile, ArgPolicy> const& sparse_array) {
  typedef DistArray<Tile, ResultPolicy> ArrayType;
  ArrayType dense_array(sparse_array.world(), sparse_array.trange());

  typedef typename ArrayType::pmap_interface pmap_interface;
  auto& pmap = dense_array.pmap();

  typename pmap_interface::const_iterator end = pmap->end();

  // iterate over sparse tiles
  for (typename pmap_interface::const_iterator it = pmap->begin(); it != end;
       ++it) {
    const std::size_t ord = *it;
    if (!sparse_array.is_zero(ord)) {
      // clone because tiles are shallow copied
      Tile tile(sparse_array.find(ord).get().clone());
      dense_array.set(ord, tile);
    } else {
      if constexpr (detail::is_tensor_of_tensor_v<Tile>) {
        // `zero' tiles that satisfy detail::is_tensor_of_tensor_v<Tile>
        //  will be left uninitialized
      } else {
        // see DistArray::set(ordinal, element_type)
        dense_array.set(ord, 0);
      }
    }
  }

  return dense_array;
}

// If array is already dense just use the copy constructor.
template <typename Tile, typename Policy>
inline std::enable_if_t<is_dense_v<Policy>, DistArray<Tile, Policy>> to_dense(
    DistArray<Tile, Policy> const& other) {
  return other;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_SPARSE_TO_DENSE_H__INCLUDED
