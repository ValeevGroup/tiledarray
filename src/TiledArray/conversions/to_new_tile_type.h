#pragma once
#ifndef TILEDARRAY_TONEWTILETYPE_H__INCLUDED
#define TILEDARRAY_TONEWTILETYPE_H__INCLUDED

#include <TiledArray/array.h>

namespace TiledArray {
namespace conversion {
namespace detail {

template <typename NewTile>
class array_creator {
   public:
    template <typename T, unsigned int DIM, typename Tile, typename Policy>
    Array<T, DIM, NewTile, Policy> create_from_prototype(
        Array<T, DIM, Tile, Policy> const &proto) const;
};

template <typename NewTile>
template <typename T, unsigned int DIM, typename Tile>
Array<T, DIM, NewTile, SparsePolicy>
array_creator<NewTile>::create_from_prototype<T, DIM, Tile, SparsePolicy>(
    Array<T, DIM, Tile, SparsePolicy> const &proto) const {
    return Array<T, DIM, NewTile, SparsePolicy>{
        proto.get_world(), proto.trange(), proto.get_shape().clone()};
}

template <typename NewTile>
template <typename T, unsigned int DIM, typename Tile>
Array<T, DIM, NewTile, DensePolicy>
array_creator<NewTile>::create_from_prototype<T, DIM, Tile, DensePolicy>(
    Array<T, DIM, Tile, DensePolicy> const &proto) const {
    return Array<T, DIM, NewTile, DensePolicy>{proto.get_world(),
                                               proto.trange()};
}

}  // namespace detail
}  // namespace conversion

/// Function to convert an array to a new array with a different tile type.

template <
    typename T, unsigned int DIM, typename Tile, typename Policy, typename Fn,
    typename std::enable_if<!std::is_same<Tile, typename Fn::TileType>::value,
                            Tile>::type * = nullptr>
Array<T, DIM, Fn::TileType, Policy> to_new_tile_type(
    Array<T, DIM, Tile, Policy> const &old_array, Fn converting_function) {
    using namespace conversion::detail;
    using TileType = Fn::TileType;

    auto new_array = array_creator<TileType>{}.create_from_prototype(old_array);

    const auto end = old_array.end();
    for (auto it = old_array.begin(); it != end; ++it) {
        auto const &old_tile = it->get();
        const auto ord = it.ordinal();

        new_array.set(ord, converting_function(old_tile));
    }

    return new_array;
}

}  // namespace TiledArray
#endif /* end of include guard: TILEDARRAY_TONEWTILETYPE_H__INCLUDED */
