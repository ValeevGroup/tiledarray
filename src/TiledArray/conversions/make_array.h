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

#include "TiledArray/array_impl.h"
#include "TiledArray/external/madness.h"
#include "TiledArray/pmap/replicated_pmap.h"
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
inline Array make_array(
    World& world, const detail::trange_t<Array>& trange,
    const std::shared_ptr<const detail::pmap_t<Array>>& pmap, Op&& op) {
  typedef typename Array::value_type value_type;
  typedef typename value_type::range_type range_type;

  // Make an empty result array
  Array result(world, trange);

  // Construct the task function used to construct the result tiles.
  std::atomic<std::int64_t> ntask_completed{0};
  std::int64_t ntask_created{0};

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
    ++ntask_created;
    tile.register_callback(
        new detail::IncrementCounter<decltype(ntask_completed)>(
            ntask_completed));
    // Store result tile
    result.set(index, std::move(tile));
  }

  // Wait for tile tasks to complete
  if (ntask_created > 0)
    world.await([&ntask_completed, ntask_created]() -> bool {
      return ntask_completed == ntask_created;
    });

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
inline Array make_array(
    World& world, const detail::trange_t<Array>& trange,
    const std::shared_ptr<const detail::pmap_t<Array>>& pmap, Op&& op) {
  typedef typename Array::value_type value_type;
  typedef typename Array::ordinal_type ordinal_type;
  typedef std::pair<ordinal_type, Future<value_type>> datum_type;

  // Create a vector to hold local tiles
  std::vector<datum_type> tiles;
  tiles.reserve(pmap->size());

  // Construct a tensor to hold updated tile norms for the result shape.
  TiledArray::Tensor<typename detail::shape_t<Array>::value_type> tile_norms(
      trange.tiles_range(), 0);

  // Construct the task function used to construct the result tiles.
  std::atomic<std::int64_t> ntask_completed{0};
  std::int64_t ntask_created{0};
  auto task = [&](const ordinal_type index) -> value_type {
    value_type tile;
    tile_norms.at_ordinal(index) = op(tile, trange.make_tile_range(index));
    return tile;
  };

  for (const auto index : *pmap) {
    auto result_tile = world.taskq.add(task, index);
    ++ntask_created;
    result_tile.register_callback(
        new detail::IncrementCounter<decltype(ntask_completed)>(
            ntask_completed));
    tiles.emplace_back(index, std::move(result_tile));
  }

  // Wait for tile norm data to be collected.
  if (ntask_created > 0)
    world.await([&ntask_completed, ntask_created]() -> bool {
      return ntask_completed == ntask_created;
    });

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

/// a make_array variant that uses a sequence of {tile_index,tile} pairs
/// to construct a DistArray with default pmap
template <typename Array, typename Tiles>
Array make_array(World& world, const detail::trange_t<Array>& tiled_range,
                 Tiles begin, Tiles end, bool replicated) {
  Array array;
  using Tuple = std::remove_reference_t<decltype(*begin)>;
  using Index = std::tuple_element_t<0, Tuple>;
  using shape_type = typename Array::shape_type;

  std::shared_ptr<typename Array::pmap_interface> pmap;
  if (replicated) {
    size_t ntiles = tiled_range.tiles_range().volume();
    pmap = std::make_shared<detail::ReplicatedPmap>(world, ntiles);
  }

  if constexpr (shape_type::is_dense()) {
    array = Array(world, tiled_range, pmap);
  } else {
    std::vector<std::pair<Index, float>> tile_norms;
    for (Tiles it = begin; it != end; ++it) {
      auto [index, tile] = *it;
      tile_norms.push_back({index, tile.norm()});
    }
    shape_type shape(world, tile_norms, tiled_range);
    array = Array(world, tiled_range, shape, pmap);
  }
  for (Tiles it = begin; it != end; ++it) {
    auto [index, tile] = *it;
    if (array.is_zero(index)) continue;
    array.set(index, tile);
  }
  return array;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_MAKE_ARRAY_H__INCLUDED
