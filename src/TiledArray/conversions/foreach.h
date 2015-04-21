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

#ifndef TILEDARRAY_CONVERSIONS_FOREACH_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_FOREACH_H__INCLUDED

#include <TiledArray/type_traits.h>

namespace TiledArray {

  /// Forward declarations
  template <typename, unsigned int, typename, typename>
  class Array;
  class DensePolicy;
  class SparsePolicy;

  /// Apply a function to each tile of a dense Array

  /// The expected signature of the tile operation is:
  /// \code
  /// void op(typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& result_tile,
  ///     const typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& arg_tile);
  /// \endcode
  /// \tparam Op Tile operation
  /// \tparam T Element type of the array
  /// \tparam DIM Dimension of the array
  /// \tparam Tile The tile type of the array
  /// \param op The tile function
  /// \param arg The argument array
  template <typename T, unsigned int DIM, typename Tile, typename Op>
  inline Array<T, DIM, Tile, DensePolicy>
  foreach(const Array<T, DIM, Tile, DensePolicy> arg, Op&& op) {
    typedef Array<T, DIM, Tile, DensePolicy> array_type;
    typedef typename array_type::size_type size_type;

    madness::World& world = arg.get_world();

    // Make an empty result array
    array_type result(world, arg.trange(), arg.get_pmap());

    // Iterate over local tiles of arg
    typename array_type::pmap_interface::const_iterator
    it = arg.pmap()->begin(),
    end = arg.pmap()->end();
    for(; it != end; ++it) {
      // Spawn a task to evaluate the tile
      madness::Future<typename array_type::value_type> tile =
          world.taskq.add([=] (const typename array_type::value_type arg_tile) {
            typename array_type::value_type result_tile(arg_tile.range());
            op(result_tile, arg_tile);
            return result_tile;
          }, arg.find(*it));

      // Store result tile
      result.set(*it, tile);
    }
  }

  /// Apply a function to each tile of a dense Array

  /// The expected signature of the tile operation is:
  /// \code
  /// void op(typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& tile);
  /// \endcode
  /// \tparam Op Mutating tile operation
  /// \tparam T Element type of the array
  /// \tparam DIM Dimension of the array
  /// \tparam Tile The tile type of the array
  /// \param op The mutating tile function
  /// \param arg The argument array
  template <typename T, unsigned int DIM, typename Tile, typename Op>
  inline void
  foreach_inplace(Array<T, DIM, Tile, DensePolicy>& arg, Op&& op) {
    typedef Array<T, DIM, Tile, DensePolicy> array_type;
    typedef typename array_type::size_type size_type;

    madness::World& world = arg.get_world();
    world.gop.fence();

    // Make an empty result array
    array_type result(world, arg.trange(), arg.get_pmap());

    // Iterate over local tiles of arg
    typename array_type::pmap_interface::const_iterator
    it = arg.pmap()->begin(),
    end = arg.pmap()->end();
    for(; it != end; ++it) {
      // Spawn a task to evaluate the tile
      madness::Future<typename array_type::value_type> tile =
          world.taskq.add([=] (typename array_type::value_type& arg_tile) {
            op(arg_tile);
            return arg_tile;
          }, arg.find(*it));

      // Store result tile
      result.set(*it, tile);
    }

    // Set the arg with the new array
    arg = result;
    Array<T, DIM, Tile, DensePolicy>::wait_for_lazy_cleanup(world);
  }

  /// Apply a function to each tile of a sparse Array

  /// The expected signature of the tile operation is:
  /// \code
  /// void op(typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& result_tile,
  ///     const typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& arg_tile);
  /// \endcode
  /// where the return value of \c op is the 2-norm (Fibrinous norm).
  /// \tparam Op Tile operation
  /// \tparam T Element type of the array
  /// \tparam DIM Dimension of the array
  /// \tparam Tile The tile type of the array
  /// \param op The tile function
  /// \param arg The argument array
  template <typename T, unsigned int DIM, typename Tile, typename Op>
  inline Array<T, DIM, Tile, SparsePolicy>
  foreach(const Array<T, DIM, Tile, SparsePolicy> arg, Op&& op) {
    typedef Array<T, DIM, Tile, SparsePolicy> array_type;
    typedef typename array_type::value_type value_type;
    typedef typename array_type::size_type size_type;
    typedef typename array_type::shape_type shape_type;
    typedef madness::Future<value_type> future_type;
    typedef std::pair<size_type, future_type> datum_type;

    // Create a vector to hold local tiles
    std::vector<datum_type> tiles;
    tiles.reserve(arg.get_pmap()->size());

    // Collect updated shape data.
    TiledArray::Tensor<typename shape_type::value_type>
    tile_norms(arg.trange().tiles(), 0);

    // Construct the new tile norms and
    madness::AtomicInt counter; counter = 0;
    int task_count = 0;
    auto task = [&](const size_type index, const value_type& arg_tile) {
      value_type result_tile(arg_tile.range());
      tile_norms[index] = op(result_tile, arg_tile);
      ++counter;
      return result_tile;
    };

    madness::World& world = arg.get_world();

    // Get local tile index iterator
    typename array_type::pmap_interface::const_iterator
        it = arg.get_pmap()->begin(),
        end = arg.get_pmap()->end();
    for(; it != end; ++it) {
      const size_type index = *it;
      if(! arg.is_zero(index)) {
        future_type arg_tile = arg.find(index);
        future_type result_tile = world.taskq.add(task, index, arg_tile);
        ++task_count;
        tiles.push_back(datum_type(index, result_tile));
      }
    }

    // Wait for tile data to be collected
    if(task_count > 0) {
      TiledArray::detail::CounterProbe probe(counter, task_count);
      world.await(probe);
    }

    // Construct the new truncated array
    array_type result(world, arg.trange(),
        shape_type(world, tile_norms, arg.trange()), arg.get_pmap());
    for(typename std::vector<datum_type>::const_iterator it = tiles.begin(); it != tiles.end(); ++it) {
      const size_type index = it->first;
      if(! result.is_zero(index))
        result.set(it->first, it->second);
    }

    return result;
  }


  /// Modify the tiles of

  /// The expected signature of the tile operation is:
  /// \code
  /// void op(typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& result_tile,
  ///     const typename TiledArray::Array<T,DIM,Tile,DensePolicy>::value_type& arg_tile);
  /// \endcode
  /// where the return value of \c op is the 2-norm (Fibrinous norm).
  /// \tparam Op Tile operation
  /// \tparam T Element type of the array
  /// \tparam DIM Dimension of the array
  /// \tparam Tile The tile type of the array
  /// \param op The tile function
  /// \param arg The argument array
  template <typename T, unsigned int DIM, typename Tile, typename Op>
  inline void
  foreach_inplace(Array<T, DIM, Tile, SparsePolicy>& arg, Op&& op) {
    typedef Array<T, DIM, Tile, SparsePolicy> array_type;
    typedef typename array_type::value_type value_type;
    typedef typename array_type::size_type size_type;
    typedef typename array_type::shape_type shape_type;
    typedef madness::Future<value_type> future_type;
    typedef std::pair<size_type, future_type> datum_type;

    // Create a vector to hold local tiles
    std::vector<datum_type> tiles;
    tiles.reserve(arg.get_pmap()->size());

    // Collect updated shape data.
    TiledArray::Tensor<typename shape_type::value_type>
    tile_norms(arg.trange().tiles(), 0);

    // Construct the new tile norms and
    madness::AtomicInt counter; counter = 0;
    int task_count = 0;
    auto task = [&](const size_type index, value_type& arg_tile) {
      tile_norms[index] = op(arg_tile);
      ++counter;
      return arg_tile;
    };

    madness::World& world = arg.get_world();
    world.gop.fence();

    // Get local tile index iterator
    typename array_type::pmap_interface::const_iterator
        it = arg.get_pmap()->begin(),
        end = arg.get_pmap()->end();
    for(; it != end; ++it) {
      const size_type index = *it;
      if(! arg.is_zero(index)) {
        future_type arg_tile = arg.find(index);
        future_type result_tile = world.taskq.add(task, index, arg_tile);
        ++task_count;
        tiles.push_back(datum_type(index, result_tile));
      }
    }

    // Wait for tile data to be collected
    if(task_count > 0) {
      TiledArray::detail::CounterProbe probe(counter, task_count);
      world.await(probe);
    }

    // Construct the new truncated array
    array_type result(world, arg.trange(),
        shape_type(world, tile_norms, arg.trange()), arg.get_pmap());
    for(typename std::vector<datum_type>::const_iterator it = tiles.begin(); it != tiles.end(); ++it) {
      const size_type index = it->first;
      if(! result.is_zero(index))
        result.set(it->first, it->second);
    }

    // Set the arg with the new array
    arg = result;
    Array<T, DIM, Tile, SparsePolicy>::wait_for_lazy_cleanup(world);
  }

} // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_TRUNCATE_H__INCLUDED
