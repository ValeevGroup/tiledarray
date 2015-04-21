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
  inline Array<T, DIM, Tile, DensePolicy>&
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
    typedef Array<T, DIM, Tile, SparsePolicy> array_type;
    typedef typename array_type::value_type value_type;
    typedef typename array_type::size_type size_type;
    typedef typename array_type::shape_type shape_type;
    typedef madness::Future<value_type> future_type;
    typedef std::pair<size_type, future_type> datum_type;

    // Create a vector to hold local tiles
    std::vector<datum_type> tiles;
    tiles.reserve(array.get_pmap()->size());

    // Collect updated shape data.
    TiledArray::Tensor<float> tile_norms(array.trange().tiles(), 0.0f);

    // Construct the new tile norms and
    madness::AtomicInt counter; counter = 0;
    int task_count = 0;
    auto task = [&](const size_type index, const value_type& tile) {
      tile_norms[index] = tile.norm();
      ++counter;
    };
    for(typename array_type::const_iterator it = array.begin(); it != array.end(); ++it) {
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
    array_type result(array.get_world(), array.trange(),
        shape_type(array.get_world(), tile_norms, array.trange()),
        array.get_pmap());
    for(typename std::vector<datum_type>::const_iterator it = tiles.begin(); it != tiles.end(); ++it) {
      const size_type index = it->first;
      if(! result.is_zero(index))
        result.set(it->first, it->second);
    }

    return result;
  }

} // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_TRUNCATE_H__INCLUDED
