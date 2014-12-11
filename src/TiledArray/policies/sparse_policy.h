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

#include <TiledArray/tiled_range.h>
#include <TiledArray/pmap/blocked_pmap.h>
#include <TiledArray/sparse_shape.h>
#include <TiledArray/counter_probe.h>

namespace TiledArray {

  class SparsePolicy {
  public:
    typedef TiledArray::TiledRange trange_type;
    typedef trange_type::range_type range_type;
    typedef range_type::size_type size_type;
    typedef TiledArray::SparseShape<float> shape_type;
    typedef TiledArray::Pmap pmap_interface;
    typedef TiledArray::detail::BlockedPmap default_pmap_type;

    /// Create a default process map

    /// \param world The world of the process map
    /// \param size The number of tiles in the array
    /// \return A shared pointer to a process map
    static std::shared_ptr<pmap_interface>
    default_pmap(madness::World& world, const std::size_t size) {
      return std::shared_ptr<pmap_interface>(new default_pmap_type(world, size));
    }

    /// Truncate an Array

    /// \tparam A Array type
    /// \param array The array object to be truncated
    template <typename A>
    static void truncate(A& array) {
      typedef typename A::value_type value_type;
      typedef madness::Future<value_type> future_type;
      typedef std::pair<size_type, future_type> datum_type;

      // Create a vector to hold local tiles
      std::vector<datum_type> tiles;
      tiles.reserve(array.get_pmap()->size());

      // Collect updated shape data.
      TiledArray::Tensor<float> tile_norms(array.trange().tiles(), 0.0f);

      // Construct the new tile norms and
      madness::AtomicInt counter;
      int task_count = 0;
      auto task = [&](const size_type index, const value_type& tile) {
        tile_norms[index] = tile.norm();
        ++counter;
      };
      for(typename A::const_iterator it = array.begin(); it != array.end(); ++it) {
        future_type tile = *it;
        array.get_world().taskq.add(task, it.ordinal(), tile);
        tiles.push_back(datum_type(it.ordinal(), tile));
        ++task_count;
      }

      // Wait for tile data to be collected
      if(task_count > 0) {
        TiledArray::detail::CounterProbe probe(counter, task_count);
        array.get_world().await(probe);
      }

      // Construct the new truncated array
      A result(array.get_world(), array.trange(), shape_type(array.get_world(),
          tile_norms, array.trange()), array.get_pmap());
      for(typename std::vector<datum_type>::const_iterator it = tiles.begin(); it != tiles.end(); ++it) {
        const size_type index = it->first;
        if(! result.is_zero(index))
          result.set(it->first, it->second);
      }

      // Set array with the new data
      array = result;
    }

  }; // class SparsePolicy

} // namespace TiledArray

#endif // TILEDARRAY_SPARSE_ARRAY_H__INCLUDED
