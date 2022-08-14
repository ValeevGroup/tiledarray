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
 *  sparse_array.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_SPARSE_ARRAY_H__INCLUDED
#define TILEDARRAY_SPARSE_ARRAY_H__INCLUDED

#include <TiledArray/pmap/blocked_pmap.h>
#include <TiledArray/sparse_shape.h>
#include <TiledArray/tiled_range.h>

namespace TiledArray {

class SparsePolicy {
 public:
  typedef TiledArray::TiledRange trange_type;
  typedef trange_type::range_type range_type;
  typedef range_type::index1_type index1_type;
  typedef range_type::ordinal_type ordinal_type;
  typedef TiledArray::SparseShape<float> shape_type;
  typedef TiledArray::Pmap pmap_interface;
  typedef TiledArray::detail::BlockedPmap default_pmap_type;

  /// Create a default process map

  /// \param world The world of the process map
  /// \param size The number of tiles in the array
  /// \return A shared pointer to a process map
  static std::shared_ptr<const pmap_interface> default_pmap(
      World& world, const std::size_t size) {
    return std::make_shared<const default_pmap_type>(world, size);
  }

};  // class SparsePolicy

}  // namespace TiledArray

#endif  // TILEDARRAY_SPARSE_ARRAY_H__INCLUDED
