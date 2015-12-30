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

#include "tiledarray.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include "sparse_tile.h"

using namespace TiledArray;

template <long id>
struct tag {
};

struct MixedExpressionsFixture : public TiledRangeFixture {

  typedef DistArray<EigenSparseTile<double, tag<0>>, DensePolicy> TArrayDS1;
  typedef DistArray<EigenSparseTile<double, tag<1>>, DensePolicy> TArrayDS2;

  MixedExpressionsFixture() :
    u(*GlobalFixture::world, trange2),
    v(*GlobalFixture::world, trange2),
    w(*GlobalFixture::world, trange2)
  {
    random_fill(u);
    random_fill(v);
    GlobalFixture::world->gop.fence();
  }

  template <typename Tile>
  static void random_fill(DistArray<Tile>& array) {
    typename DistArray<Tile>::pmap_interface::const_iterator it = array.get_pmap()->begin();
    typename DistArray<Tile>::pmap_interface::const_iterator end = array.get_pmap()->end();
    for(; it != end; ++it)
      array.set(*it, array.get_world().taskq.add(& MixedExpressionsFixture::template make_rand_tile<DistArray<Tile> >,
          array.trange().make_tile_range(*it)));
  }


  template <typename T>
  static void set_random(T& t) {
    // with 50% generate nonzero integer value in [0,101)
    auto rand_int = GlobalFixture::world->rand();
    t = (rand_int < 0x8ffffful) ? rand_int % 101 : 0;
  }


  template <typename T>
  static void set_random(std::complex<T>& t) {
    // with 50% generate nonzero value
    auto rand_int1 = GlobalFixture::world->rand();
    if (rand_int1 < 0x8ffffful) {
      t = std::complex<T>{T(rand_int1 % 101),
			  T(GlobalFixture::world->rand() % 101)};
    }
    else
      t = std::complex<T>{0,0};
  }

  // Fill a tile with random data
  template <typename A>
  static typename A::value_type
  make_rand_tile(const typename A::value_type::range_type& r) {
    typename A::value_type tile(r);
    for(const auto& i: r) {
      set_random(tile[i]);
    }
    return tile;
  }

  template <typename M, typename A>
  static void rand_fill_matrix_and_array(M& matrix, A& array, int seed = 42) {
    TA_ASSERT(std::size_t(matrix.size()) == array.trange().elements().volume());
    matrix.fill(0);

    GlobalFixture::world->srand(seed);

    // Iterate over local tiles
    for(typename A::iterator it = array.begin(); it != array.end(); ++it) {
      typename A::value_type tile(array.trange().make_tile_range(it.index()));
      for(Range::const_iterator rit = tile.range().begin(); rit != tile.range().end(); ++rit) {
        const std::size_t elem_index = array.elements().ordinal(*rit);
        tile[*rit] = (matrix.array()(elem_index) = (GlobalFixture::world->rand() % 101));
      }
      *it = tile;
    }
    GlobalFixture::world->gop.sum(& matrix(0,0), matrix.size());
  }

  template <typename Tile>
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>
  make_matrix(DistArray<Tile>& array) {
    // Check that the array will fit in a matrix or vector

    // Construct the Eigen matrix
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>
        matrix(array.trange().elements().extent_data()[0],
            (array.trange().tiles().rank() == 2 ? array.trange().elements().extent_data()[1] : 1));

    // Spawn tasks to copy array tiles to the Eigen matrix
    for(std::size_t i = 0; i < array.size(); ++i) {
      if(! array.is_zero(i))
        tensor_to_eigen_submatrix(array.find(i).get(), matrix);
    }

    return matrix;
  }

  ~MixedExpressionsFixture() {
    GlobalFixture::world->gop.fence();
  }

  const static TiledRange trange1;
  const static TiledRange trange2;
  TArrayD u;
  TArrayD u1;
  TArrayDS1 v;
  TArrayDS1 v1;
  TArrayDS2 w;
}; // MixedExpressionsFixture

// Instantiate static variables for fixture
const TiledRange MixedExpressionsFixture::trange1 =
    { {0, 2, 5, 10, 17, 28, 41}};
const TiledRange MixedExpressionsFixture::trange2 =
    { {0, 2, 5, 10, 17, 28, 41},
      {0, 3, 6, 11, 18, 29, 42} };

BOOST_FIXTURE_TEST_SUITE( mixed_expressions_suite, MixedExpressionsFixture )

BOOST_AUTO_TEST_CASE( tensor_factories )
{
  const auto& ca = a;
  const std::array<int,2> lobound{{3,3}};
  const std::array<int,2> upbound{{5,5}};

  BOOST_CHECK_NO_THROW(w("a,b") = u("b,a"));
  BOOST_CHECK_NO_THROW(v("a,b") = w("b,a"));
  BOOST_CHECK_NO_THROW(u("a,b") = v("b,a"));
}

BOOST_AUTO_TEST_CASE( add_factories )
{
  // compile error: dense + sparse = dense, not sparse
  // BOOST_CHECK_NO_THROW(w("a,b") = u("a,b") + v("a,b"));

  // ok
  BOOST_CHECK_NO_THROW(u1("a,b") = u("a,b") + v("a,b"));

}

BOOST_AUTO_TEST_CASE( mult_factories )
{
  // compile error: dense * sparse = sparse, not dense
  // BOOST_CHECK_NO_THROW(u1("a,b") = u("a,b") * v("a,b"));

  // compile error: dense * sparse2 = sparse1, not sparse2
  //BOOST_CHECK_NO_THROW(w("a,b") = u("a,b") * v("a,b"));

  // ok
  BOOST_CHECK_NO_THROW(v1("a,b") = u("a,b") * v("a,b"));

}

BOOST_AUTO_TEST_SUITE_END()
