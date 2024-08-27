/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  Justus Calvin, Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions_mixed.cpp
 *  May 10, 2013
 *
 */

#include "TiledArray/special/diagonal_array.h"
#include "TiledArray/special/kronecker_delta.h"
#include "range_fixture.h"
#include "sparse_tile.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

template <long id>
struct tag {};

struct MixedExpressionsFixture : public TiledRangeFixture {
  typedef DistArray<EigenSparseTile<double, tag<0>>, DensePolicy> TArrayDS1;
  typedef DistArray<EigenSparseTile<double, tag<1>>, DensePolicy> TArrayDS2;
  typedef DistArray<KroneckerDeltaTile, DensePolicy> ArrayKronDelta;
  typedef DistArray<KroneckerDeltaTile, SparsePolicy> SpArrayKronDelta;

  MixedExpressionsFixture()
      : u(*GlobalFixture::world, trange2),
        u2(*GlobalFixture::world, trange4),
        e2(*GlobalFixture::world, trange2e),
        e4(*GlobalFixture::world, trange4e),
        v(*GlobalFixture::world, trange2),
        w(*GlobalFixture::world, trange2) {
    random_fill(u);
    random_fill(v);
    u2.fill(0);
    random_fill(e2);
    e4.fill(0);
    GlobalFixture::world->gop.fence();
  }

  template <typename Tile, typename Policy>
  static void random_fill(DistArray<Tile, Policy>& array) {
    array.fill_random();
  }

  template <typename M, typename A>
  static void rand_fill_matrix_and_array(M& matrix, A& array, int seed = 42) {
    TA_ASSERT(std::size_t(matrix.size()) == array.trange().elements().volume());
    matrix.fill(0);

    GlobalFixture::world->srand(seed);

    // Iterate over local tiles
    for (typename A::iterator it = array.begin(); it != array.end(); ++it) {
      typename A::value_type tile(array.trange().make_tile_range(it.index()));
      for (Range::const_iterator rit = tile.range().begin();
           rit != tile.range().end(); ++rit) {
        const std::size_t elem_index = array.elements().ordinal(*rit);
        tile[*rit] =
            (matrix.array()(elem_index) = (GlobalFixture::world->rand() % 101));
      }
      *it = tile;
    }
    GlobalFixture::world->gop.sum(&matrix(0, 0), matrix.size());
  }

  template <typename Tile>
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> make_matrix(
      DistArray<Tile>& array) {
    // Check that the array will fit in a matrix or vector

    // Construct the Eigen matrix
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> matrix(
        array.trange().elements().extent_data()[0],
        (array.trange().tiles().rank() == 2
             ? array.trange().elements().extent_data()[1]
             : 1));

    // Spawn tasks to copy array tiles to the Eigen matrix
    for (std::size_t i = 0; i < array.size(); ++i) {
      if (!array.is_zero(i))
        tensor_to_eigen_submatrix(array.find(i).get(), matrix);
    }

    return matrix;
  }

  template <typename Policy>
  static void init_kronecker_delta(
      DistArray<KroneckerDeltaTile, Policy>& array) {
    array.init_tiles([=](const TiledArray::Range& range) {
      return KroneckerDeltaTile(range);
    });
  }

  ~MixedExpressionsFixture() { GlobalFixture::world->gop.fence(); }

  const static TiledRange trange1;
  const static TiledRange trange2;
  const static TiledRange trange4;
  const static TiledRange trange2e;  // all dimensions are equivalent, to make
                                     // easier testing delta and permutations
  const static TiledRange trange4e;  // all dimensions are equivalent
  TArrayD u;
  TArrayD u1;
  TArrayD u2;
  TArrayD e2;
  TArrayD e4;
  TArrayDS1 v;
  TArrayDS1 v1;
  TArrayDS2 w;
};  // MixedExpressionsFixture

// Instantiate static variables for fixture
const TiledRange MixedExpressionsFixture::trange1{{0, 2, 5, 10, 17, 28, 41}};
const TiledRange MixedExpressionsFixture::trange2{{0, 2, 5, 10, 17, 28, 41},
                                                  {0, 3, 6, 11, 18, 29, 42}};
const TiledRange MixedExpressionsFixture::trange4{
    trange2.data()[0], trange2.data()[1], trange2.data()[0], trange2.data()[1]};
const TiledRange MixedExpressionsFixture::trange2e{trange1.data()[0],
                                                   trange1.data()[0]};
const TiledRange MixedExpressionsFixture::trange4e{
    trange2e.data()[0], trange2e.data()[1], trange2e.data()[0],
    trange2e.data()[1]};

BOOST_FIXTURE_TEST_SUITE(mixed_expressions_suite, MixedExpressionsFixture)

BOOST_AUTO_TEST_CASE(tensor_factories) {
  BOOST_CHECK_NO_THROW(w("a,b") = u("b,a"));
  BOOST_CHECK_NO_THROW(v("a,b") = w("b,a"));
  BOOST_CHECK_NO_THROW(u("a,b") = v("b,a"));
}

BOOST_AUTO_TEST_CASE(add_factories) {
  // compile error: dense + sparse = dense, not sparse
  // BOOST_CHECK_NO_THROW(w("a,b") = u("a,b") + v("a,b"));

  // ok
  BOOST_CHECK_NO_THROW(u1("a,b") = u("a,b") + v("a,b"));
}

BOOST_AUTO_TEST_CASE(mult_factories) {
#if MULT_DENSE_SPARSE_TO_SPARSE
  // compile error: dense * sparse = sparse, not dense
  // BOOST_CHECK_NO_THROW(u1("a,b") = u("a,b") * v("a,b"));

  // ok
  BOOST_CHECK_NO_THROW(v1("a,b") = u("a,b") * v("a,b"));
#else
  // ok
  BOOST_CHECK_NO_THROW(u1("a,b") = u("a,b") * v("a,b"));

  // compile error: dense * sparse = dense, not sparse
  // BOOST_CHECK_NO_THROW(v1("a,b") = u("a,b") * v("a,b"));
#endif

  // compile error: dense * sparse1 != sparse2
  // BOOST_CHECK_NO_THROW(w("a,b") = u("a,b") * v("a,b"));
}

BOOST_AUTO_TEST_CASE(kronecker) {
#if !MULT_DENSE_SPARSE_TO_SPARSE
  // ok
  BOOST_CHECK_NO_THROW(u2("a,b,c,d") += u("a,b") * v("c,d"));
#endif

  // retile test
  TSpArrayD x(*GlobalFixture::world, trange2);
  random_fill(x);

  // includes target tiles that receive contributions from multiple source
  // tiles, tiny target tiles with single contribution, and tiles partially and
  // completely outside the source range
#ifdef TA_SIGNED_1INDEX_TYPE
  TA::TiledRange yrange{{-1, 18, 20, 45, 47}, {-1, 20, 22, 45, 47}};
#else
  TA::TiledRange yrange{{5, 18, 20, 45, 47}, {7, 20, 22, 45, 47}};
#endif
  TA::TSpArrayD y1;
  // identical to y1 = TA::detail::retile_v1(x, yrange);
  TA::TiledRange retiler_range{yrange.dim(0), yrange.dim(1), trange2.dim(0),
                               trange2.dim(1)};
  SpArrayKronDelta retiler(
      *GlobalFixture::world, retiler_range,
      SparseShape(detail::kronecker_shape(retiler_range), retiler_range),
      std::make_shared<detail::ReplicatedPmap>(
          *GlobalFixture::world, retiler_range.tiles_range().volume()));
  init_kronecker_delta(retiler);
  y1("d1,d2") = retiler("d1,d2,s1,s2") * x("s1,s2");
  // std::cout << "y1 = " << y1 << std::endl;

  auto y_ref = TA::retile(x, yrange);
  // std::cout << "y_ref = " << y_ref << std::endl;
  BOOST_CHECK((y1("d1,d2") - y_ref("d1,d2")).norm().get() == 0.);
}

BOOST_AUTO_TEST_SUITE_END()
