#pragma once
#ifndef TILEDARRAY_SPARSETODENSE_H__INCLUDED
#define TILEDARRAY_SPARSETODENSE_H__INCLUDED

#include <TiledArray/array.h>

namespace TiledArray {

/// Function to convert a block sparse array into a dense array.

template <typename T, unsigned int DIM, typename Tile, typename Policy,
          typename std::enable_if<std::is_same<SparsePolicy, Policy>::value,
                                  Policy>::type* = nullptr>
Array<T, DIM, Tile, DensePolicy> to_dense(
    Array<T, DIM, Tile, Policy> const& sparse_array) {

    using ArrayType = Array<T, DIM, Tile, DensePolicy>;

    auto& world = sparse_array.get_world();
    ArrayType dense_array(world, sparse_array.trange());

    using pmap_interface = typename ArrayType::pmap_interface;
    using pmap_iter = typename pmap_interface::const_iterator;

    std::shared_ptr<pmap_interface> const& pmap = dense_array.get_pmap();

    auto conv = [&](pmap_iter it) {
        const std::size_t ord = *it;
        if (!sparse_array.is_zero(ord)) {
            Tile tile(sparse_array.find(ord).get().clone());
            dense_array.set(ord, tile);
        } else {
            dense_array.set(ord, T(0.0));
        }
        return madness::Future<bool>(true);
    };

    world.taskq.for_each(madness::Range<pmap_iter>(pmap->begin(), pmap->end()),
                         conv);

    return dense_array;
}

// If array is already dense just use the copy constructor.
template <typename T, unsigned int DIM, typename Tile, typename Policy,
          typename std::enable_if<std::is_same<DensePolicy, Policy>::value,
                                  Policy>::type* = nullptr>
Array<T, DIM, Tile, DensePolicy> to_dense(
    Array<T, DIM, Tile, Policy> const& other) {
    return other;
}

}  // namespace TiledArray

#endif /* end of include guard: TILEDARRAY_SPARSETODENSE_H__INCLUDED */
