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

  template <typename Tile>
  DistArray<Tile, DensePolicy>
  to_dense(DistArray<Tile, SparsePolicy> const& sparse_array) {
      typedef DistArray<Tile, DensePolicy> ArrayType;
      ArrayType dense_array(sparse_array.get_world(), sparse_array.trange());

      typedef typename ArrayType::pmap_interface pmap_interface;
      std::shared_ptr<pmap_interface> const& pmap = dense_array.get_pmap();

      typename pmap_interface::const_iterator end = pmap->end();

      // iteratate over sparse tiles
      for (typename pmap_interface::const_iterator it = pmap->begin(); it != end;
           ++it) {
          const std::size_t ord = *it;
          if (!sparse_array.is_zero(ord)) {
              // clone because tiles are shallow copied
              Tile tile(sparse_array.find(ord).get().clone());
              dense_array.set(ord, tile);
          } else {
              // This is how Array::set_all_local() sets tiles to a value,
              // This likely means that what ever type Tile is must be
              // constructible from a type T
              dense_array.set(ord, 0);  // This is how Array::set_all_local()
          }
      }

      return dense_array;
  }

  // If array is already dense just use the copy constructor.
  template <typename Tile>
  DistArray<Tile, DensePolicy>
  to_dense(DistArray<Tile, DensePolicy> const& other) {
      return other;
  }

}  // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_SPARSE_TO_DENSE_H__INCLUDED
