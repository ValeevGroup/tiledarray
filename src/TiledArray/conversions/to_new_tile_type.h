#pragma once
#ifndef TILEDARRAY_TONEWTILETYPE_H__INCLUDED
#define TILEDARRAY_TONEWTILETYPE_H__INCLUDED

#include <TiledArray/array.h>

namespace TiledArray {
namespace conversion {

namespace detail {
template <typename T>
using result_of_t = typename std::result_of<T>::type;
}

/// Function to convert an array to a new array with a different tile type.

template <typename T, unsigned int DIM, typename Tile, typename Policy,
          typename Fn,
          typename std::enable_if<
              !std::is_same<Tile, detail::result_of_t<Fn(Tile)>>::value,
              Tile>::type * = nullptr>
Array<T, DIM, detail::result_of_t<Fn(Tile)>, Policy> to_new_tile_type(
    Array<T, DIM, Tile, Policy> const &old_array, Fn converting_function) {
    using TileType = detail::result_of_t<Fn(Tile)>;

    auto new_array = Array<T, DIM, TileType, Policy>{
        old_array.get_world(), old_array.trange(), old_array.get_shape()};

    const auto end = old_array.end();
    auto conv = [&](decltype(end) const &it) {
        auto const &old_tile = it->get();
        const auto ord = it.ordinal();
        new_array.set(ord, converting_function(old_tile));
    };

    old_array.get_world().taskq.for_each<madness::rangeT, decltype(conv)>(
        madness::Range<decltype(old_array.begin())>(old_array.begin(),
                                                    old_array.end()),
        conv);

    old_array.get_world().gop.fence();

    return new_array;
}

}  // namespace conversion
}  // namespace TiledArray
#endif /* end of include guard: TILEDARRAY_TONEWTILETYPE_H__INCLUDED */
