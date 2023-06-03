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
#endif

    #endif

    
  }

  TA::finalize();
}
