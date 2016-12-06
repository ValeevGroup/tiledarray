/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2016  Virginia Tech
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
 *  Drew Lewis
 *  Department of Chemistry, Virginia Tech
 *
 *  diagonal_array.h
 *  Nov 30, 2016
 *
 */

#ifndef TILEDARRAY_SPECIALARRAYS_DIAGONAL_ARRAY_H__INCLUDED
#define TILEDARRAY_SPECIALARRAYS_DIAGONAL_ARRAY_H__INCLUDED

#include <TiledArray/dist_array.h>
#include <TiledArray/range.h>
#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>

#include <vector>

namespace TiledArray {
namespace detail {

// Function that returns a range containing the diagonal elements in the tile
inline Range diagonal_range(Range const &rng) {
    auto lo = rng.lobound();
    auto up = rng.upbound();

    // Determine the largest lower index and the smallest upper index
    auto max_low = *std::max_element(std::begin(lo), std::end(lo));
    auto min_up = *std::min_element(std::begin(up), std::end(up));

    // If the max small elem is less than the min large elem then a diagonal
    // elem is in this tile;
    if (max_low < min_up) {
        return Range({max_low}, {min_up});
    } else {
        return Range();
    }
}

template <typename T>
Tensor<float> diagonal_shape(TiledRange const &trange, T val) {
    Tensor<float> shape(trange.tiles_range(), 0.0);

    auto ext = trange.elements_range().extent();
    auto min_dim_len = *std::min_element(std::begin(ext), std::end(ext));

    auto ndim = trange.rank();
    auto diag_elem = 0ul;
    // the diagonal elements will never be larger than the length of the
    // shortest dimension
    while(diag_elem < min_dim_len){
        // Get the tile index corresponding to the current diagonal_elem
        auto tile_idx = trange.element_to_tile(std::vector<int>(ndim, diag_elem));
        auto tile_range = trange.make_tile_range(tile_idx);

        // Compute the range of diagonal elements in the tile
        auto d_range = diagonal_range(tile_range);

        // Since each diag elem has the same value the  norm of the tile is 
        // \sqrt{\sum_{diag} val^2}  = \sqrt{ndiags * val^2}
        float t_norm = std::sqrt(val * val * d_range.volume());
        shape(tile_idx) = t_norm;

        // Update diag_elem to the next elem not in this tile
        diag_elem = d_range.upbound_data()[0];
    }

    return shape;
}

// Actually do all the work of writing the diagonal tiles
template<typename Array, typename T>
void write_tiles_to_array(Array &A, T val){
    auto const &trange = A.trange();
    const auto ndims = trange.rank();

    // Task to create each tile
    auto tile_task = [val, &trange, ndims](unsigned long ord){
            auto rng = trange.make_tile_range(ord);
            
            // Compute range of diagonal elements in the tile
            auto diags = detail::diagonal_range(rng);

            Tensor<T> tile(rng, 0.0);

            if (diags.volume() > 0) { // If the tile has diagonal elems 

                // Loop over the elements and write val into them
                auto diag_lo = diags.lobound_data()[0];
                auto diag_hi = diags.upbound_data()[0];
                for (auto elem = diag_lo; elem < diag_hi; ++elem) {
                    tile(std::vector<int>(ndims, elem)) = val;
                }
            }

            return tile;
    };

    // SparsePolicy arrays incur a small overhead by looping over all ordinals,
    // Until proven to be a problem we will just keep it. 
    const auto vol = trange.tiles_range().volume();
    for (auto ord = 0ul; ord < vol; ++ord) {
        if (A.is_local(ord) && !A.is_zero(ord)) {
            auto tile_future = A.world().taskq.add(tile_task, ord);
            A.set(ord, tile_future);
        }
    }
}

}  // namespace detail


/// Create a DensePolicy DistArray with only diagonal elements, 
/// the expected behavior is that every element (n,n,n, ..., n) will be nonzero.

/// \param world The world for the array
/// \param trange The trange for the array
/// \param val The value to be written along the diagonal elements 

template <typename T>
DistArray<Tensor<T>, DensePolicy> dense_diagonal_array(World &world,
                                               TiledRange const &trange,
                                               T val = 1) {
    // Init the array
    DistArray<Tensor<T>, DensePolicy> A(world, trange);

    detail::write_tiles_to_array(A, val);

    world.gop.fence();
    return A;
}

/// Create a SparsePolicy DistArray with only diagonal elements, 
/// the expected behavior is that every element (n,n,n, ..., n) will be nonzero.

/// \param world The world for the array
/// \param trange The trange for the array
/// \param val The value to be written along the diagonal elements 

template <typename T>
DistArray<Tensor<T>, SparsePolicy> sparse_diagonal_array(World &world,
                                               TiledRange const &trange,
                                               T val = 1) {

    // Compute shape and init the Array
    auto shape_norm = detail::diagonal_shape(trange, val);
    SparseShape<float> shape(shape_norm, trange);
    DistArray<Tensor<T>, SparsePolicy> A(world, trange, shape);

    detail::write_tiles_to_array(A, val);

    world.gop.fence();
    return A;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_SPECIALARRAYS_DIAGONAL_ARRAY_H__INCLUDED
