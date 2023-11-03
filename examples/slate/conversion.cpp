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

#include <TiledArray/conversions/slate.h>

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









int main(int argc, char** argv) {
  auto& world = TA::initialize(argc, argv);
  {
    int64_t N = argc > 1 ? std::stoi(argv[1]) : 1000;
    size_t NB = argc > 2 ? std::stoi(argv[2]) : 128;

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

    // Do Conversion 
    auto A = TA::array_to_slate( ref_ta );
    auto A_ta = TA::slate_to_array<TA::TArray<double>>(A, world);
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
    //if(!world.rank()) {
    //std::cout << "SLATE\n" << slate_eigen << std::endl;
    //}

    A_ta.make_replicated();
    world.gop.fence();
    auto A_eigen = TA::array_to_eigen(A_ta);
    //if(!world.rank()) std::cout << "TA\n" << A_eigen << std::endl;
    std::cout << (A_eigen - slate_eigen).norm() << std::endl;

    
  }

  TA::finalize();
}
