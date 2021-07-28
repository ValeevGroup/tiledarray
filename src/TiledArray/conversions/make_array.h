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
 *  array_init.h
 *  Dec 15, 2015
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_MAKE_ARRAY_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_MAKE_ARRAY_H__INCLUDED

#include "TiledArray/external/madness.h"
#include "TiledArray/shape.h"
#include "TiledArray/type_traits.h"

/// Forward declarations
namespace Eigen {
template <typename>
class aligned_allocator;
}  // namespace Eigen

namespace TiledArray {

/// Construct dense Array

/// This function is used to construct a `DistArray` object. Users must
/// provide a world object, tiled range object, and function/functor that
/// generates the tiles for the new array object. For example, if we want to
/// create an array with were the elements are equal to `1`:
/// \code
/// TiledArray::TArray<double> out_array =
///     make_array<TiledArray::TArray<double> >(world, trange, pmap,
///           [=] (TiledArray::Tensor<double>& tile, const TiledArray::Range&
///           range) {
///             tile = TiledArray::Tensor<double>(range);
///             for(auto& it : tile)
///               *it = 1;
///           });
/// \endcode
/// Note that the result is default constructed before (contains no data) and
/// must be initialized inside the function/functor with the provided range
/// object. The expected signature of the tile operation is:
/// \code
/// void op(tile_t& tile, const range_t& range);
/// \endcode
/// where `tile_t` and `range_t` are your tile type and tile range type,
/// respectively.
/// \tparam Array The `DistArray` type
/// \tparam Op Tile operation
/// \param world The world where the array will live
/// \param trange The tiled range of the array
/// \param op The tile function/functor
/// \return An array object of type `Array`
template <typename Array, typename Op,
          typename std::enable_if<is_dense<Array>::value>::type* = nullptr>
inline Array make_array(World& world, const detail::trange_t<Array>& trange,
                        const std::shared_ptr<detail::pmap_t<Array> >& pmap,
                        Op&& op) {
  typedef typename Array::value_type value_type;
  typedef typename value_type::range_type range_type;

  // Make an empty result array
  Array result(world, trange);

  // Iterate over local tiles of arg
  for (const auto index : *result.pmap()) {
    // Spawn a task to evaluate the tile
    auto tile = world.taskq.add(
        [=](const range_type& range) -> value_type {
          value_type tile;
          op(tile, range);
          return tile;
        },
        trange.make_tile_range(index));

    // Store result tile
    result.set(index, tile);
  }

  return result;
}

/// Construct sparse Array

/// This function is used to construct a `DistArray` object. Users must
/// provide a world object, tiled range object, process map, and function/
/// functor that generates the tiles for the new array object. For example,
/// if we want to create an array with all elements equal to `1`:
/// \code
/// TiledArray::TSpArray<double> array =
///     make_array<TiledArray::TSpArray<double> >(world, trange, pmap,
///           [=] (TiledArray::Tensor<double>& tile, const TiledArray::Range&
///           range) -> double {
///             tile = TiledArray::Tensor<double>(range);
///             for(auto& it : tile)
///               *it = 1;
///             return tile.norm();
///           });
/// \endcode
/// You may choose not to initialize a tile inside the tile initialization
/// function (not shown in the example) by returning `0` for the tile norm.
/// Note that the result is default constructed before (contains no data) and
/// must be initialized inside the function/functor with the provided range
/// object unless the returned tile norm is zero. The expected signature of
/// the tile operation is:
/// \code
/// value_t op(tile_t& tile, const range_t& range);
/// \endcode
/// where `value_t`, `tile_t` and `range_t` are your tile value type, tile
/// type, and tile range type, respectively.
/// \tparam Array The `DistArray` type
/// \tparam Op Tile operation
/// \param world The world where the array will live
/// \param trange The tiled range of the array
/// \param pmap A shared pointer to the array process map
/// \param op The tile function/functor
/// \return An array object of type `Array`
template <typename Array, typename Op,
          typename std::enable_if<!is_dense<Array>::value>::type* = nullptr>
inline Array make_array(World& world, const detail::trange_t<Array>& trange,
                        const std::shared_ptr<detail::pmap_t<Array> >& pmap,
                        Op&& op) {
  typedef typename Array::value_type value_type;
  typedef typename Array::ordinal_type ordinal_type;
  typedef std::pair<ordinal_type, Future<value_type> > datum_type;

  // Create a vector to hold local tiles
  std::vector<datum_type> tiles;
  tiles.reserve(pmap->size());

  // Construct a tensor to hold updated tile norms for the result shape.
  TiledArray::Tensor<typename detail::shape_t<Array>::value_type>
      tile_norms(trange.tiles_range(), 0);

  // Construct the task function used to construct the result tiles.
  madness::AtomicInt counter;
  counter = 0;
  int task_count = 0;
  auto task = [&](const ordinal_type index) -> value_type {
    value_type tile;
    tile_norms[index] = op(tile, trange.make_tile_range(index));
    ++counter;
    return tile;
  };

  for (const auto index : *pmap) {
    auto result_tile = world.taskq.add(task, index);
    ++task_count;
    tiles.emplace_back(index, std::move(result_tile));
  }

  // Wait for tile norm data to be collected.
  if (task_count > 0)
    world.await(
        [&counter, task_count]() -> bool { return counter == task_count; });

  // Construct the new array
  Array result(world, trange,
               typename Array::shape_type(world, tile_norms, trange), pmap);
  for (auto& it : tiles) {
    const auto index = it.first;
    if (!result.is_zero(index)) result.set(it.first, it.second);
  }

  return result;
}

/// Construct an Array

/// This function is used to construct a `DistArray` object. Users must
/// provide a world object, tiled range object, and function/functor that
/// generates the tiles for the new array object. For example, if we want to
/// create an array with were the elements are equal to `1`:
/// \code
/// TiledArray::TSpArray<double> array =
///     make_array<TiledArray::TSpArray<double> >(world, trange,
///           [=] (TiledArray::Tensor<double>& tile, const TiledArray::Range&
///           range) -> double {
///             tile = TiledArray::Tensor<double>(range);
///             for(auto& it : tile)
///               *it = 1;
///             return tile.norm();
///           });
/// \endcode
/// For sparse arrays, you may choose not to initialize a tile inside the
/// tile initialization (not shown in the example) by returning `0` for the
/// tile norm. Note that the result is default constructed before (contains
/// no data) and must be initialized inside the function/functor with the
/// provided range object unless the returned tile norm is zero. The expected
/// signature of the tile operation is:
/// \code
/// value_t op(tile_t& tile, const range_t& range);
/// \endcode
/// where `value_t`, `tile_t` and `range_t` are your tile value type, tile
/// type, and tile range type, respectively.
/// \tparam Array The `DistArray` type
/// \tparam Op Tile operation
/// \param world The world where the array will live
/// \param trange The tiled range of the array
/// \param op The tile function/functor
/// \return An array object of type `Array`
template <typename Array, typename Op>
inline Array make_array(World& world, const detail::trange_t<Array>& trange,
                        Op&& op) {
  return make_array<Array>(world, trange,
                           detail::policy_t<Array>::default_pmap(
                               world, trange.tiles_range().volume()),
                           op);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_MAKE_ARRAY_H__INCLUDED
