#pragma once
#ifndef TILEDARRAY_TONEWTILETYPE_H__INCLUDED
#define TILEDARRAY_TONEWTILETYPE_H__INCLUDED

#include <TiledArray/array.h>

namespace TiledArray {

namespace detail {
template <typename T>
using result_of_t = typename std::result_of<T>::type;
}

/// Function to convert an array to a new array with a different tile type.

template <typename T, unsigned int DIM, typename Tile, typename Policy,
          typename Op>
Array<T, DIM, detail::result_of_t<Op(Tile)>, Policy> to_new_tile_type(
    Array<T, DIM, Tile, Policy> const &old_array, Op &&op) {
    using OutTileType = detail::result_of_t<Op(Tile)>;
    using OutArray = Array<T, DIM, OutTileType, Policy>;

    static_assert(!std::is_same<Tile, OutTileType>::value,
                  "Can't call new tile type if tile type does not change.");

    auto &world = old_array.get_world();

    // Create new array
    OutArray new_array(world, old_array.trange(), old_array.get_shape(),
                       old_array.get_pmap());

    using pmap_iter = decltype(old_array.get_pmap()->begin());
    pmap_iter it = old_array.get_pmap()->begin();
    pmap_iter end = old_array.get_pmap()->end();

    for (; it != end; ++it) {
        // Must check for zero because pmap_iter does not.
        if (!old_array.is_zero(*it)) {
            // Spawn a task to evaluate the tile
            Future<OutTileType> tile =
                world.taskq.add(op, old_array.find(*it));
            new_array.set(*it, tile);
        }
    }

    return new_array;
}

}  // namespace TiledArray
#endif /* end of include guard: TILEDARRAY_TONEWTILETYPE_H__INCLUDED */
