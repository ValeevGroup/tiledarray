/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
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
 *  David Williams-Young
 *  Computational Research Division, Lawrence Berkeley National Laboratory
 *
 *  conversion.cpp
 *  Created: 7  Feb, 2020
 *  Edited:  13 May, 2020
 *
 */

#include <tiledarray.h>
#include <random>
#include <slate/slate.hh>
#include <TiledArray/pmap/user_pmap.h>

template <typename Integral1, typename Integral2>
int64_t div_ceil(Integral1 x, Integral2 y) {
  int64_t x_ll = x;
  int64_t y_ll = y;

  auto d = std::div(x_ll, y_ll);
  return d.quot + !!d.rem;
}

TA::TiledRange gen_trange(size_t N, const std::vector<size_t>& TA_NBs) {
  assert(TA_NBs.size() > 0);

  std::default_random_engine gen(0);
  std::uniform_int_distribution<> dist(0, TA_NBs.size() - 1);
  auto rand_indx = [&]() { return dist(gen); };
  auto rand_nb = [&]() { return TA_NBs[rand_indx()]; };

  std::vector<size_t> t_boundaries = {0};
  auto TA_NB = rand_nb();
  while (t_boundaries.back() + TA_NB < N) {
    t_boundaries.emplace_back(t_boundaries.back() + TA_NB);
    TA_NB = rand_nb();
  }
  t_boundaries.emplace_back(N);

  std::vector<TA::TiledRange1> ranges(
      2, TA::TiledRange1(t_boundaries.begin(), t_boundaries.end()));

  return TA::TiledRange(ranges.begin(), ranges.end());
};



auto make_square_proc_grid(MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    int p,q;
    for(p = int( sqrt( mpi_size ) ); p > 0; --p) {
        q = int( mpi_size / p );
        if(p*q == mpi_size) break;
    }
    return std::make_pair(p,q);
}








template <typename Array>
using slate_from_array_t = 
    typename slate::Matrix<typename std::remove_cv_t<Array>::element_type>;




template <typename Array>
slate_from_array_t<Array> array_to_slate( const Array& array ) {

    using slate_int = int64_t;
    using slate_process_idx = std::tuple<slate_int, slate_int>;
    using dim_functor_t  = std::function<slate_int(slate_int)>;
    using tile_functor_t = std::function<int(slate_process_idx)>;
    using element_type   = typename std::remove_cv_t<Array>::element_type;
    using slate_matrix_t = typename slate::Matrix<element_type>;

    using col_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::ColMajor>;
    using row_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::RowMajor>;

    using col_major_map_t = Eigen::Map<col_major_mat_t>;
    using row_major_map_t = Eigen::Map<const row_major_mat_t>;

    /*******************************/
    /*** Generate SLATE Functors ***/
    /*******************************/
    auto&       world  = array.world();
    const auto& trange = array.trange();
    auto        pmap   = array.pmap();
    if( trange.rank() != 2 )
        throw std::runtime_error("Cannot Convert General Tensor to SLATE (RANK != 2)");

    // Tile row dimension (MB)
    dim_functor_t tileMb = [&](slate_int i){ 
        return trange.dim(0).tile(i).extent();
    }; 

    // Tile col dimension (MB)
    dim_functor_t tileNb = [&](slate_int i){ 
        return trange.dim(1).tile(i).extent();
    }; 

    // Tile rank assignment
    tile_functor_t tileRank = [pmap, &trange] (slate_process_idx ij) {
        auto [i,j] = ij;
        return pmap->owner(trange.tiles_range().ordinal(i,j));
    };

    // Tile device assignment
    // TODO: Needs to be more robust
    tile_functor_t tileDevice = [&](slate_process_idx ij) { return 0; };


    /*********************************/
    /*** Create empty slate matrix ***/
    /*********************************/
    const auto M = trange.dim(0).extent();
    const auto N = trange.dim(1).extent();
    slate_matrix_t matrix(M, N, tileMb, tileNb, tileRank, tileDevice,
        world.mpi.comm().Get_mpi_comm());
    
    /************************/
    /*** Copy TA -> SLATE ***/
    /************************/
    matrix.insertLocalTiles();

    // Loop over local tiles via ordinal
    // TODO: Make async
    for( auto local_ordinal : *pmap ) {
        // Compute coordinate of tile ordinal
        auto local_coordinate = trange.tiles_range().idx(local_ordinal);
        const auto it = local_coordinate[0];
        const auto jt = local_coordinate[1];

        // Sanity Check
        if(!matrix.tileIsLocal(it,jt))
            throw std::runtime_error("SLATE PMAP is not valid");

        // Extract shallow copy of local SLATE tile and create
        // data map
        auto local_tile_slate = matrix(it,jt);
        auto local_m = local_tile_slate.mb();
        auto local_n = local_tile_slate.nb();
        col_major_map_t slate_map(local_tile_slate.data(), local_m, local_n);

        // Create data map for TA tile
        // TODO: This should be async in a MADNESS task
        auto& local_tile = array.find_local(local_ordinal).get();
        auto  local_m_ta = local_tile.range().dim(0).extent();
        auto  local_n_ta = local_tile.range().dim(1).extent();
        row_major_map_t ta_map(local_tile.data(), local_m_ta, local_n_ta);

        // Copy TA tile to SLATE tile
        // XXX: This will error out if the dimensions aren't consistent
        slate_map = ta_map;
    } // Loop over local tiles

    return matrix;

}


template <typename Array>
auto slate_to_array( slate_from_array_t<Array>& matrix, TA::World& world ) {


    static_assert(TA::is_dense<Array>::value, "SLATE -> TA Only For Dense Array");
    using value_type = typename Array::value_type; // Tile type
    using element_type   = typename std::remove_cv_t<Array>::element_type;
    using slate_matrix_t = typename slate::Matrix<element_type>;

    using col_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::ColMajor>;
    using row_major_mat_t = Eigen::Matrix<element_type,-1,-1,Eigen::RowMajor>;

    using col_major_map_t = Eigen::Map<const col_major_mat_t>;
    using row_major_map_t = Eigen::Map<row_major_mat_t>;

    // Compute SLATE Tile Statistics
    size_t total_tiles = matrix.nt() * matrix.mt();
    size_t local_tiles = 0;

    // Create a map from tile ordinal to rank
    // to avoid lifetime issues in the internal
    // TA Pmap
    std::vector<size_t> tile2rank(total_tiles);
    for (int64_t it = 0; it < matrix.mt(); ++it)
    for (int64_t jt = 0; jt < matrix.nt(); ++jt) {
        size_t ordinal = it*matrix.nt() + jt; // TODO: Use Range
        tile2rank[ordinal] = matrix.tileRank( it, jt );
        if(matrix.tileIsLocal(it,jt)) local_tiles++;
    }
    

    // Create TA PMap
    std::function<size_t(size_t)> ta_tile_functor = 
        [t2r = std::move(tile2rank)](size_t ordinal) {
            return t2r[ordinal];
        };

    std::shared_ptr<TA::Pmap> slate_pmap = 
        std::make_shared<TA::detail::UserPmap>(world, total_tiles, local_tiles, 
            ta_tile_functor);

    // Create TiledRange
    std::vector<size_t> row_tiling(matrix.mt()+1), col_tiling(matrix.nt()+1);

    row_tiling[0] = 0;
    for(auto i = 0; i < matrix.mt(); ++i) 
        row_tiling[i+1] = row_tiling[i] + matrix.tileMb(i);
    
    col_tiling[0] = 0;
    for(auto i = 0; i < matrix.nt(); ++i) 
        col_tiling[i+1] = col_tiling[i] + matrix.tileNb(i);


    std::vector<TA::TiledRange1> ranges = {
        TA::TiledRange1(row_tiling.begin(), row_tiling.end()),
        TA::TiledRange1(col_tiling.begin(), col_tiling.end())
    };
    TA::TiledRange trange(ranges.begin(), ranges.end());

    // Create TArray
    Array array(world, trange, slate_pmap);
    for (int64_t it = 0; it < matrix.mt(); ++it)
    for (int64_t jt = 0; jt < matrix.nt(); ++jt) 
    if( matrix.tileIsLocal(it,jt) ) {
        auto local_ordinal = trange.tiles_range().ordinal(it,jt);

        auto tile = world.taskq.add(
            [=](slate::Tile<double> slate_tile, TA::Range const& range) {
                // Create tile
                value_type tile(range, 0.0);

                // Create Maps                
                auto local_m = slate_tile.mb();
                auto local_n = slate_tile.nb();
                col_major_map_t slate_map(slate_tile.data(), local_m, local_n);

                auto  local_m_ta = range.dim(0).extent();
                auto  local_n_ta = range.dim(1).extent();
                row_major_map_t ta_map(tile.data(), local_m_ta, local_n_ta);

                // Copy data
                ta_map = slate_map;

                return tile;
            }, matrix(it,jt), trange.make_tile_range(local_ordinal));
        
        array.set(local_ordinal, tile);
    }

    world.gop.fence();
    return array;
}



int main(int argc, char** argv) {
  auto& world = TA::initialize(argc, argv);
  {
    int64_t N = argc > 1 ? std::stoi(argv[1]) : 1000;
    int64_t NB = argc > 2 ? std::stoi(argv[2]) : 128;

    auto make_ta_reference = 
      [&](TA::Tensor<double>& t, TA::Range const& range) {

        t = TA::Tensor<double>(range, 0.0);
        auto lo = range.lobound_data();
        auto up = range.upbound_data();
        for (int m = lo[0]; m < up[0]; ++m) {
          for (int n = lo[1]; n < up[1]; ++n) {
            t(m, n) = m - n;
          }
        }

        return t.norm();
      };

    // Generate Reference TA tensor.
    auto trange = gen_trange(N, {NB});
    auto ref_ta =
        TA::make_array<TA::TArray<double> >(world, trange, make_ta_reference);


    #if 0
    ref_ta.make_replicated();
    world.gop.fence();
    auto ref_eigen = TA::array_to_eigen(ref_ta);
    if(!world.rank()) std::cout << "REF\n" << ref_eigen << std::endl;
    world.gop.fence();

    // Generate Slate Matrix
    slate::Matrix<double> A(N,N, NB, world.size(), 1, MPI_COMM_WORLD);
    A.insertLocalTiles();
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                auto T = A( i, j );
                for(int ii = 0; ii < T.mb(); ++ii)
                for(int jj = 0; jj < T.nb(); ++jj) {
                    T.data()[ii + jj*T.stride()] = (i*NB + ii) - (j*NB + jj);
                }
            }
        }
    }
    world.gop.fence();


    // Slate matrix to eigen
    Eigen::MatrixXd slate_eigen = Eigen::MatrixXd::Zero(N,N);
    for (int64_t j = 0; j < A.nt(); ++j) 
    for (int64_t i = 0; i < A.mt(); ++i) {
        A.tileBcast(i,j, A, slate::Layout::ColMajor);
        auto T = A(i,j);
        Eigen::Map<Eigen::MatrixXd> T_map( T.data(), T.mb(), T.nb() );
        slate_eigen.block(i*NB,j*NB,T.mb(), T.nb()) = T_map; 
    }
    world.gop.fence();
    if(!world.rank()) {
    std::cout << "SLATE\n" << slate_eigen << std::endl;
    }
    #else

    // MB functor
    std::function< int64_t(int64_t) >
    tileMb = [trange](int64_t i) {
        return trange.dim(0).tile(i).extent();
    };
    // NB functor
    std::function< int64_t(int64_t) >
    tileNb = [trange](int64_t j) {
        return trange.dim(1).tile(j).extent();
    };
    std::function< int( std::tuple<int64_t,int64_t> ) >
    tileRank = [pmap = ref_ta.get_pmap(),trange](std::tuple<int64_t,int64_t> ij) {
        auto [i,j] = ij;
        return pmap->owner(i*trange.dim(1).tile_extent() + j);
    };

    std::function< int(std::tuple<int64_t,int64_t>) >
    tileDevice = [](auto) { return 0; };
    slate::Matrix<double> A(N,N, tileNb, tileMb, tileRank, tileDevice,
        MPI_COMM_WORLD);

#if 0
    A.insertLocalTiles();
    for(int it = 0; it < A.mt(); ++it)
    for(int jt = 0; jt < A.nt(); ++jt) {
        //auto ordinal = it * trange.dim(1).tile_extent() + jt;
        auto ordinal = trange.tiles_range().ordinal(it,jt);
        if( ordinal != it * trange.dim(1).tile_extent() + jt ) throw "die die die";
        if(A.tileIsLocal(it,jt)) {
        printf("[RANK %d] Tile(%d,%d): %lu %lu / %lu %lu - %lu\n",
            world.rank(),
            it, jt, A(it,jt).mb(), A(it,jt).nb(),
            trange.dim(0).tile(it).extent(), 
            trange.dim(1).tile(jt).extent(),
            ref_ta.pmap()->owner(ordinal));
        }
    }
#endif
    
#if 1

    #if 0
    // Populte tiles directly
    A.insertLocalTiles();
    for(int it = 0; it < A.mt(); ++it)
    for(int jt = 0; jt < A.nt(); ++jt) {
        if(A.tileIsLocal(it, jt)) {
            auto T = A(it, jt);
            for(int ii = 0; ii < T.mb(); ++ii)
            for(int jj = 0; jj < T.nb(); ++jj) {
                T.at(ii,jj) = (it*NB + ii) - (jt*NB + jj);
            }
        }
    }
    #else
    #if 0
    A.insertLocalTiles();
    for(auto local_ordinal : *ref_ta.pmap()) {
        auto local_coordinate = trange.tiles_range().idx(local_ordinal);
        auto it = local_coordinate[0];
        auto jt = local_coordinate[1];
        if(!A.tileIsLocal(it,jt)) throw std::runtime_error("Something Went Horribly Wrong");

        auto& local_tile = ref_ta.find_local(local_ordinal).get();
        Eigen::Map<Eigen::Matrix<double,-1,-1,Eigen::RowMajor>> 
            local_tile_map(local_tile.data(), local_tile.range().dim(0).extent(), local_tile.range().dim(1).extent());

        auto local_tile_slate = A(it,jt);
        Eigen::Map<Eigen::MatrixXd> local_tile_slate_map( local_tile_slate.data(),
            local_tile_slate.mb(), local_tile_slate.nb() );
        local_tile_slate_map = local_tile_map;
    }
    #else
    auto tmpA = array_to_slate( ref_ta );
    A = std::move(tmpA);

    auto A_ta = slate_to_array<TA::TArray<double>>(A, world);
    #endif
    #endif
    // Slate matrix to eigen
    Eigen::MatrixXd slate_eigen = Eigen::MatrixXd::Zero(N,N);
    for (int64_t j = 0; j < A.nt(); ++j) 
    for (int64_t i = 0; i < A.mt(); ++i) {
        A.tileBcast(i,j, A, slate::Layout::ColMajor);
        auto T = A(i,j);
        Eigen::Map<Eigen::MatrixXd> T_map( T.data(), T.mb(), T.nb() );
        slate_eigen.block(i*NB,j*NB,T.mb(), T.nb()) = T_map; 
    }
    world.gop.fence();
    if(!world.rank()) {
    std::cout << "SLATE\n" << slate_eigen << std::endl;
    }

    //ref_ta.make_replicated();
    //std::cout << ref_ta << std::endl;
    //world.gop.fence();
    A_ta.make_replicated();
    world.gop.fence();
    auto A_eigen = TA::array_to_eigen(A_ta);
    if(!world.rank()) std::cout << "TA\n" << A_eigen << std::endl;
    world.gop.fence();
#endif

    #endif

    
  }

  TA::finalize();
}
