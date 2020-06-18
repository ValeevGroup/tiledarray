/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions_fixture.h
 *  Jan 19, 2019
 *
 */

#ifndef TILEDARRAY_TEST_EXPRESSIONS_FIXTURE_H
#define TILEDARRAY_TEST_EXPRESSIONS_FIXTURE_H

#include <TiledArray/util/eigen.h>
#include <boost/range/combine.hpp>
#ifdef TILEDARRAY_HAS_RANGEV3
#include <range/v3/view/zip.hpp>
#endif

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_BTAS
#include <TiledArray/external/btas.h>
#endif

#include <boost/mpl/vector.hpp>
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

template <typename Tile, typename Policy>
struct ExpressionsFixture : public TiledRangeFixture {
  using TArray = TA::DistArray<Tile, Policy>;
  using element_type = typename Tile::value_type;
  using scalar_type = typename Tile::scalar_type;
  using Matrix = Eigen::Matrix<element_type, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;

  template <typename P = Policy,
            std::enable_if_t<
                std::is_same<P, TiledArray::SparsePolicy>::value>* = nullptr>
  ExpressionsFixture()
      : s_tr_1(make_random_sparseshape(tr)),
        s_tr_2(make_random_sparseshape(tr)),
        s_tr1_1(make_random_sparseshape(trange1)),
        s_tr1_2(make_random_sparseshape(trange1)),
        s_tr2(make_random_sparseshape(trange2)),
        a(*GlobalFixture::world, tr, s_tr_1),
        b(*GlobalFixture::world, tr, s_tr_2),
        c(*GlobalFixture::world, tr, s_tr_2),
        u(*GlobalFixture::world, trange1, s_tr1_1),
        v(*GlobalFixture::world, trange1, s_tr1_2),
        w(*GlobalFixture::world, trange2, s_tr2) {
    random_fill(a);
    random_fill(b);
    random_fill(u);
    random_fill(v);
    GlobalFixture::world->gop.fence();
    a.truncate();
    b.truncate();
    u.truncate();
    v.truncate();
  }

  template <typename P = Policy,
            std::enable_if_t<std::is_same<P, TiledArray::DensePolicy>::value>* =
                nullptr>
  ExpressionsFixture()
      : a(*GlobalFixture::world, tr),
        b(*GlobalFixture::world, tr),
        c(*GlobalFixture::world, tr),
        u(*GlobalFixture::world, trange1),
        v(*GlobalFixture::world, trange1),
        w(*GlobalFixture::world, trange2) {
    random_fill(a);
    random_fill(b);
    random_fill(u);
    random_fill(v);
    GlobalFixture::world->gop.fence();
  }

  /// make array for SparsePolicy
  template <typename P = Policy,
            std::enable_if_t<
                std::is_same<P, TiledArray::SparsePolicy>::value>* = nullptr>
  static TA::DistArray<Tile, Policy> make_array(TA::TiledRange& range) {
    return TA::DistArray<Tile, Policy>(*GlobalFixture::world, range,
                                       make_random_sparseshape(range));
  }

  /// make array for DensePolicy
  template <typename P = Policy,
            std::enable_if_t<std::is_same<P, TiledArray::DensePolicy>::value>* =
                nullptr>
  static TA::DistArray<Tile, Policy> make_array(TA::TiledRange& range) {
    return TA::DistArray<Tile, Policy>(*GlobalFixture::world, range);
  }

  /// randomly fill an array
  static void random_fill(DistArray<Tile, Policy>& array) {
    auto it = array.pmap()->begin();
    auto end = array.pmap()->end();
    for (; it != end; ++it) {
      if (!array.is_zero(*it))
        array.set(*it,
                  array.world().taskq.add(make_rand_tile,
                                          array.trange().make_tile_range(*it)));
    }
  }

  template <typename T>
  static void set_random(T& t) {
    t = GlobalFixture::world->rand() % 101;
  }

  template <typename T>
  static void set_random(std::complex<T>& t) {
    t = std::complex<T>{T(GlobalFixture::world->rand() % 101),
                        T(GlobalFixture::world->rand() % 101)};
  }

  // Fill a tile with random data
  static Tile make_rand_tile(const typename Tile::range_type& r) {
    Tile tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    return tile;
  }

  // make a tile with 0 data
  static Tile make_zero_tile(const typename Tile::range_type& r) {
    Tile tile(r, 0);
    return tile;
  }

  static void rand_fill_matrix_and_array(Matrix& matrix, TArray& array,
                                         int seed = 42) {
    TA_ASSERT(std::size_t(matrix.size()) ==
              array.trange().elements_range().volume());
    matrix.fill(0);

    GlobalFixture::world->srand(seed);

    // Iterate over local tiles
    for (auto it = array.begin(); it != array.end(); ++it) {
      Tile tile(array.trange().make_tile_range(it.index()));
      for (Range::const_iterator rit = tile.range().begin();
           rit != tile.range().end(); ++rit) {
        const std::size_t elem_index = array.elements_range().ordinal(*rit);
        tile[*rit] =
            (matrix.array()(elem_index) = (GlobalFixture::world->rand() % 101));
      }
      *it = tile;
    }
    GlobalFixture::world->gop.sum(&matrix(0, 0), matrix.size());
  }

  Matrix make_matrix(DistArray<Tile, Policy>& array) {
    // Check that the array will fit in a matrix or vector

    // Construct the Eigen matrix
    Matrix matrix =
        Matrix::Zero(array.trange().elements_range().extent(0),
                     (array.trange().tiles_range().rank() == 2
                          ? array.trange().elements_range().extent(1)
                          : 1));

    // Spawn tasks to copy array tiles to the Eigen matrix
    for (std::size_t i = 0; i < array.size(); ++i) {
      if (!array.is_zero(i))
        tensor_to_eigen_submatrix(array.find(i).get(), matrix);
    }

    return matrix;
  }

  /// make a shape with approximate half dense and half sparse
  static SparseShape<float> make_random_sparseshape(const TiledRange& tr) {
    std::size_t n = tr.tiles_range().volume();
    Tensor<float> norms(tr.tiles_range(), 0.0);

    // make sure all mpi gets the same shape
    if (GlobalFixture::world->rank() == 0) {
      for (std::size_t i = 0; i < n; i++) {
        norms[i] = GlobalFixture::world->drand() > 0.5 ? 0.0 : 1.0;
      }
    }

    GlobalFixture::world->gop.broadcast_serializable(norms, 0);

    return SparseShape<float>(norms, tr);
  }

  ~ExpressionsFixture() { GlobalFixture::world->gop.fence(); }

  const TiledRange trange1 = {{0, 2, 5, 10, 17, 28, 41}};
  const TiledRange trange2 = {{0, 2, 5, 10, 17, 28, 41},
                              {0, 3, 6, 11, 18, 29, 42}};
  SparseShape<float> s_tr_1;
  SparseShape<float> s_tr_2;
  SparseShape<float> s_tr1_1;
  SparseShape<float> s_tr1_2;
  SparseShape<float> s_tr2;
  TArray a;
  TArray b;
  TArray c;
  TArray u;
  TArray v;
  TArray w;
};  // ExpressionsFixture

#endif  // TILEDARRAY_TEST_EXPRESSIONS_FIXTURE_H
