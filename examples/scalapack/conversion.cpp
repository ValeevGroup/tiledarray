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

int main(int argc, char** argv) {
  auto& world = TA::initialize(argc, argv);
  {
    size_t N = argc > 1 ? std::stoi(argv[1]) : 1000;
    size_t NB = argc > 2 ? std::stoi(argv[2]) : 128;

    // Create Test Matrix
    blacspp::Grid grid = blacspp::Grid::square_grid(MPI_COMM_WORLD);
    TA::BlockCyclicMatrix<double> ref_matrix(world, grid, N, N, NB, NB);

    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        if (ref_matrix.dist().i_own(i, j)) {
          auto [i_local, j_local] = ref_matrix.dist().local_indx(i, j);
          ref_matrix.local_mat()(i_local, j_local) = i + j;
        }

    // Functor to generate identical matrix in tiles
    auto make_ta_reference = [](TA::Tensor<double>& t, TA::Range const& range) {
      t = TA::Tensor<double>(range, 0.0);
      auto lo = range.lobound_data();
      auto up = range.upbound_data();
      for (auto m = lo[0]; m < up[0]; ++m) {
        for (auto n = lo[1]; n < up[1]; ++n) {
          t(m, n) = m + n;
        }
      }

      return t.norm();
    };

    std::cout << std::scientific;

    // Uniform Tiling = NB:  MAT -> TA
    {
      auto trange = gen_trange(N, {NB});

      auto ref_ta =
          TA::make_array<TA::TArray<double> >(world, trange, make_ta_reference);

      world.gop.fence();
      auto test_ta = ref_matrix.tensor_from_matrix<TA::TArray<double>>(trange);
      world.gop.fence();

      double norm_diff = (ref_ta("i,j") - test_ta("i,j")).norm(world).get();

      double ref_norm = ref_ta("i,j").norm(world).get();
      double test_norm = test_ta("i,j").norm(world).get();

      if (!world.rank()) {
        std::cout << "|| REF  ||_2                  = " << ref_norm
                  << std::endl;
        std::cout << "|| TEST ||_2                  = " << test_norm
                  << std::endl;
        std::cout << "|| MAT -> TA DIFF (UNIF) ||_2 = " << norm_diff
                  << std::endl;
      }
    }

    // Uniform Tiling = NB: TA -> MAT
    {
      auto trange = gen_trange(N, {NB});

      auto ref_ta =
          TA::make_array<TA::TArray<double> >(world, trange, make_ta_reference);

      world.gop.fence();
      TA::BlockCyclicMatrix<double> test_matrix(ref_ta, grid, NB, NB);
      world.gop.fence();

      double local_norm_diff =
          (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
      local_norm_diff *= local_norm_diff;

      double norm_diff;
      MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      norm_diff = std::sqrt(norm_diff);
      if (!world.rank())
        std::cout << "|| TA -> MAT DIFF (UNIF) ||_2 = " << norm_diff
                  << std::endl;
    }

    // Random Tiling:  MAT -> TA
    {
      auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});

      auto ref_ta =
          TA::make_array<TA::TArray<double> >(world, trange, make_ta_reference);

      world.gop.fence();
      auto test_ta = ref_matrix.tensor_from_matrix<TA::TArray<double>>(trange);
      world.gop.fence();

      double norm_diff = (ref_ta("i,j") - test_ta("i,j")).norm(world).get();

      double ref_norm = ref_ta("i,j").norm(world).get();
      double test_norm = test_ta("i,j").norm(world).get();

      if (!world.rank()) {
        std::cout << "|| MAT -> TA DIFF (RAND) ||_2 = " << norm_diff
                  << std::endl;
      }
    }

    // Random Tiling: TA -> MAT
    {
      auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});

      auto ref_ta =
          TA::make_array<TA::TArray<double> >(world, trange, make_ta_reference);

      world.gop.fence();
      TA::BlockCyclicMatrix<double> test_matrix(ref_ta, grid, NB, NB);
      world.gop.fence();

      double local_norm_diff =
          (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
      local_norm_diff *= local_norm_diff;

      double norm_diff;
      MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      norm_diff = std::sqrt(norm_diff);
      if (!world.rank())
        std::cout << "|| TA -> MAT DIFF (RAND) ||_2 = " << norm_diff
                  << std::endl;
    }
  }

  TA::finalize();
}
