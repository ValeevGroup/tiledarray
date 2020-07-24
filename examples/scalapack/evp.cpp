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
 *  evp.cpp
 *  Created: 11 May, 2020
 *
 */
#include <tiledarray.h>
#include <random>

#include <scalapackpp/eigenvalue_problem/sevp.hpp>
#include <scalapackpp/pblas/gemm.hpp>

#include <TiledArray/math/scalapack.h>

using Array = TA::TArray<double>;
// using Array = TA::TSpArray<double>;

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

int main(int argc, char** argv) {
  auto& world = TA::initialize(argc, argv);
  {
    size_t N = argc > 1 ? std::stoi(argv[1]) : 1000;
    size_t NB = argc > 2 ? std::stoi(argv[2]) : 128;

    std::default_random_engine gen(world.rank());
    std::normal_distribution<> dist(0., 1.);
    auto rand_gen = [&]() { return dist(gen); };

    // Functor to create random, diagonally dominant tiles
    auto make_random_ta = [&](TA::Tensor<double>& t, TA::Range const& range) {
      t = TA::Tensor<double>(range, 0.0);
      auto lo = range.lobound_data();
      auto up = range.upbound_data();
      for (auto m = lo[0]; m < up[0]; ++m) {
        for (auto n = lo[1]; n < up[1]; ++n) {
          t(m, n) = rand_gen();
          if (m == n) t(m, n) += 100.;
        }
      }

      return t.norm();
    };

    // Create BLACS Grid context
    auto world_comm = world.mpi.comm().Get_mpi_comm();
    blacspp::Grid grid = blacspp::Grid::square_grid(world_comm);

    // Create TA tensor
    auto trange = gen_trange(N, {NB});
    auto tensor = TA::make_array<Array>(world, trange, make_random_ta);

    // Symmetrize
    Array tensor_symm(world, trange);
    tensor_symm("i,j") = 0.5 * (tensor("i,j") + tensor("j,i"));
    tensor("i,j") = tensor_symm("i,j");


    auto [ evals, evecs_ta ] = TA::heig( tensor );


    //// Check EVP with TA
    Array tmp = TA::foreach (evecs_ta, [evals = evals](TA::Tensor<double>& result,
                                           const TA::Tensor<double>& arg) {
      result = TA::clone(arg);

      auto range = arg.range();
      auto lo = range.lobound_data();
      auto up = range.upbound_data();
      for (auto m = lo[0]; m < up[0]; ++m)
        for (auto n = lo[1]; n < up[1]; ++n) {
          result(m, n) = arg(m, n) * evals[n];
        }
    });

    world.gop.fence();
    tensor("i,j") = tensor("i,j") - tmp("i,k") * evecs_ta("j,k");

    world.gop.fence();
    auto err_norm = tensor("i,j").norm(world).get();
    if (~world.rank())
      std::cout << "EVP (Tensor) |A - XEX**T| = " << err_norm << std::endl;

    world.gop.fence();
  }
  TA::finalize();
}
