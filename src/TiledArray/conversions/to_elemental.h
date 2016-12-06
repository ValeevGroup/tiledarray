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
 *  Dec 1, 2016
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_TO_ELEMENTAL_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_TO_ELEMENTAL_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/elemental.h>
#include <TiledArray/tensor.h>
#include <TiledArray/dist_array.h>

#include <utility>

namespace TiledArray {
namespace detail {
inline bool uniformly_blocked(TiledRange const &trange){
    for(auto i = 0ul; i < trange.rank(); ++i){
        auto const &tr1 = trange.data()[i];

        auto it = tr1.begin();
        auto end = --tr1.end(); // 2nd to last since last can be a different size

        if(it == end){ // only 1 block in this dim
            break;
        }

        // Take first block to define the blocksize
        auto blocksize = it->second - it->first; 

        // Loop over other blocks (except last one) to compare size.
        ++it; 
        for(; it != end; ++it){
            auto size = it->second - it->first;
            if(size != blocksize){
                return false;
            }
        }
    }

    return true;
}

template <typename Array>
El::DistMatrix<typename Array::element_type> matrix_to_el(
        Array const& A, El::Grid const &g){
    
    // Check for matrix and uniform blocking
    TiledRange const &trange = A.trange();
    TA_ASSERT(trange.rank() == 2);

    // Determine matrix total size
    auto elem_extent = trange.elements_range().extent_data();
    const auto nrows = elem_extent[0];
    const auto ncols = elem_extent[1];

    // Construct elem array, zero it to avoid having to write zero tiles
    auto el_A = El::DistMatrix<typename Array::element_type>(nrows, ncols, g);
    El::Zero(el_A);

    // Loop over all tiles
    const auto vol = trange.tiles_range().volume();
    for(auto i = 0ul; i < vol; ++i){
        //
        // Write local tiles into a queue and allow elemental to do all 
        // communication of elements to the remote nodes.
        if(!A.is_zero(i) && A.is_local(i)){
            auto tile = A.find(i).get();
            
            auto lo = tile.range().lobound_data();
            auto up = tile.range().upbound_data();

            for(auto m = lo[0]; m < up[0]; ++m){
                for(auto n = lo[1]; n < up[1]; ++n){
                    el_A.QueueUpdate(m,n,tile(m,n));
                }
            }
        }
    }

    // Initialize the communication process
    el_A.ProcessQueues();

    return el_A;
}


template<typename T>
DistArray<Tensor<T>, DensePolicy> el_to_matrix(
        El::AbstractDistMatrix<T> const &M, World& world, 
        TiledRange const &trange){
    TA_ASSERT(trange.rank() == 2);
    TA_USER_ASSERT(uniformly_blocked(trange), "The output TiledRange must be uniformly blocked.");

    // Determine block size
    TiledRange1 const& tr1_row = trange.data()[0];
    auto const& tr1_col = trange.data()[1];
    auto row_bs = tr1_row.begin()->second - tr1_row.begin()->first;
    auto col_bs = tr1_col.begin()->second - tr1_col.begin()->first;

    // Copy the unknown matrix distribution into a block based distribution 
    El::DistMatrix<T, El::MC, El::MR, El::BLOCK> Mb(M.Height(), M.Width(), M.Grid(), row_bs, col_bs);
    Mb = M;

    DistArray<Tensor<T>, DensePolicy> A(world, trange);

    // Loop over the tiles in the trange
    const auto vol = trange.tiles_range().volume();
    for(auto i = 0ul; i < vol; ++i){
        // Make the tiled range for tile i
        auto range = trange.make_tile_range(i);
        auto lo = range.lobound_data();
        auto up = range.upbound_data();

        using ptr_type = decltype(lo);
        auto task = [&](Range rng){
            auto lo = rng.lobound_data();
            auto up = rng.upbound_data();
            auto tile = Tensor<T>(rng, 0.0);
            
            // Copy by element 
            for(auto m = lo[0]; m != up[0]; ++m){
                for(auto n = lo[1]; n != up[1]; ++n){
                    tile(m,n) = Mb.GetLocal(Mb.LocalRow(m),Mb.LocalCol(n));
                }
            }
            
            return tile;
        };
         
        // Send local writes to task
        if(Mb.IsLocal(lo[0], lo[1])){
            auto tile_future = A.world().taskq.add(task, range);
            A.set(i, tile_future);
        }
    }
    A.world().gop.fence();

    return A;
}

} // namespace detail

/// TensorFlattening contains a pair of vectors which dictate the indices that
/// should be combined to make a matrix. Permutations are not currently supported. 

/// Examples for a 3D tensor could be ({0},{1,2}) or ({0,1},{2}) zero indexing
/// is assumed.

class TensorFlattening {
    private:
        std::pair<std::vector<int>, std::vector<int>> flattening_;

    public:
        /// Default is the standard matrix case. 
        TensorFlattening() : 
            flattening_(std::make_pair(std::vector<int>{0}, std::vector<int>{1})) 
        {}

        TensorFlattening(
                std::pair<std::vector<int>, std::vector<int>> const& flattening) : 
            flattening_(flattening) 
        {}

        TensorFlattening(std::vector<int> const& left, 
                std::vector<int> const &right) : 
            flattening_(std::make_pair(left, right)) 
        {}

        std::vector<int> const &left_dims() const {return flattening_.first; }
        std::vector<int> const &right_dims() const {return flattening_.second; }
};

/// array_to_el converts a DistArray into an Elemental Matrix

/// @param A a TA Array with a TA Tensor Tile type
/// @param g an Elemental Grid
/// @param tf a TensorFlattening object that fuses indices in tensors with more than 2 dimensions
template <typename Array>
El::DistMatrix<typename Array::element_type> array_to_el(
        Array const& A, El::Grid const &g = El::Grid::Default(), 
        TensorFlattening const &tf = TensorFlattening()){

    if(A.trange().rank() == 2){
        // Flattening is not needed for the matrix case.
        return detail::matrix_to_el(A, g);
    } else {
        TA_USER_ASSERT(false, "array_to_el currently only supports converting matrices, higher order tensor conversions will be supported at a latter date.");
    }
}

/// el_to_array converts an Elemental DistMatrix Into a DensePolicy Array

/// @param M An Elemental DistMatrix 
/// @param world A madness world
/// @param trange A TiledRange that must be uniformly blocked in each dimension
/// @param tf A tensor flattening that dictates which dimensions to unfold. 
template<typename T>
DistArray<Tensor<T>, DensePolicy> el_to_array(
        El::AbstractDistMatrix<T> const &M, World& world, 
        TiledRange const &trange, 
        TensorFlattening const &tf = TensorFlattening()){

    if(trange.rank() == 2){
        // Flattening is not needed for the matrix case.
        return detail::el_to_matrix(M, world, trange);
    } else {
        TA_USER_ASSERT(false, "el_to_array currently only supports converting matrices, higher order tensor conversions will be supported at a latter date.");
    }
}

} // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_TO_ELEMENTAL_H__INCLUDED
