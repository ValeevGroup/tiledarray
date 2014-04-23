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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions.cpp
 *  May 10, 2013
 *
 */

#include "TiledArray/expressions.h"
#include "TiledArray/array.h"
#include "TiledArray/eigen.h"
#include "unit_test_config.h"
#include "range_fixture.h"

using namespace TiledArray;

struct ExpressionsFixture : public TiledRangeFixture {
  typedef Array<int,4> Array4;
  typedef Array<int,3> Array3;
  typedef Array<int,2> Array2;
  typedef Array<int,1> Array1;

  ExpressionsFixture() :
    trange1(dims.begin(), dims.begin() + 1),
    trange2(dims.begin(), dims.begin() + 2),
    a(*GlobalFixture::world, tr),
    b(*GlobalFixture::world, tr),
    c(*GlobalFixture::world, tr),
    u(*GlobalFixture::world, trange1),
    v(*GlobalFixture::world, trange1),
    w(*GlobalFixture::world, trange2)
  {
    random_fill(a);
    random_fill(b);
    random_fill(u);
    random_fill(v);
    GlobalFixture::world->gop.fence();
  }

  template <typename T, unsigned int DIM, typename Tile>
  static void random_fill(Array<T, DIM, Tile>& array) {
    typename Array<T, DIM, Tile>::pmap_interface::const_iterator it = array.get_pmap()->begin();
    typename Array<T, DIM, Tile>::pmap_interface::const_iterator end = array.get_pmap()->end();
    for(; it != end; ++it)
      array.set(*it, array.get_world().taskq.add(& ExpressionsFixture::template make_rand_tile<Array<T, DIM, Tile> >,
          array.trange().make_tile_range(*it)));
  }



  // Fill a tile with random data
  template <typename A>
  static typename A::value_type
  make_rand_tile(const typename A::value_type::range_type& r) {
    typename A::value_type tile(r);
    for(std::size_t i = 0ul; i < tile.size(); ++i)
      tile[i] = GlobalFixture::world->rand() / 11;
    return tile;
  }


  template <typename T, unsigned int DIM, typename Tile>
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>
  make_matrix(Array<T, DIM, Tile>& array) {
    // Check that the array will fit in a matrix or vector
    TA_ASSERT((DIM == 2u) || (DIM == 1u));

    // Construct the Eigen matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
        matrix(array.trange().elements().size()[0],
            (DIM == 2 ? array.trange().elements().size()[1] : 1));

    // Spawn tasks to copy array tiles to the Eigen matrix
    for(std::size_t i = 0; i < array.size(); ++i) {
      if(array.get_world().rank() == 1)
        std::cout << i;
      if(! array.is_zero(i))
        tensor_to_eigen_submatrix(array.find(i).get(), matrix);
      if(array.get_world().rank() == 1)
        std::cout << "*\n";
    }

    return matrix;
  }

  ~ExpressionsFixture() { }

  TiledRange trange1;
  TiledRange trange2;
  Array3 a;
  Array3 b;
  Array3 c;
  Array1 u;
  Array1 v;
  Array2 w;
}; // ExpressionsFixture

BOOST_FIXTURE_TEST_SUITE( expressions_suite, ExpressionsFixture )


BOOST_AUTO_TEST_CASE( permute )
{
  Permutation perm(2, 1, 0);
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = b("c,b,a"));

  for(std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index = a.range().ord(perm ^ b.range().idx(i));
    if(a.is_local(perm_index)) {
      Array3::value_type a_tile = a.find(perm_index).get();
      Array3::value_type perm_b_tile = perm ^ b.find(i).get();

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for(std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], perm_b_tile[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE( scale_permute )
{
  Permutation perm(2, 1, 0);
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = 2 * b("c,b,a"));

  for(std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index = a.range().ord(perm ^ b.range().idx(i));
    if(a.is_local(perm_index)) {
      Array3::value_type a_tile = a.find(perm_index).get();
      Array3::value_type perm_b_tile = perm ^ b.find(i).get();

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for(std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], 2 * perm_b_tile[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE( add )
{
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =      a("a,b,c")  +      b("a,b,c") );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) +      b("a,b,c") );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =      a("a,b,c")  + (3 * b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) + (3 * b("a,b,c")));

}

BOOST_AUTO_TEST_CASE( scale_add )
{
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (     a("a,b,c")  +      b("a,b,c")) );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) +      b("a,b,c")) );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (     a("a,b,c")  + (3 * b("a,b,c"))));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) + (3 * b("a,b,c"))));

}


BOOST_AUTO_TEST_CASE( subt )
{
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =      a("a,b,c")  -      b("a,b,c") );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) -      b("a,b,c") );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =      a("a,b,c")  - (3 * b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) - (3 * b("a,b,c")));

}

BOOST_AUTO_TEST_CASE( scale_subt )
{
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (     a("a,b,c")  -      b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) -      b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (     a("a,b,c")  - (3 * b("a,b,c"))));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) - (3 * b("a,b,c"))));

}


BOOST_AUTO_TEST_CASE( mult )
{
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =      a("a,b,c")  *      b("a,b,c") );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) *      b("a,b,c") );
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =      a("a,b,c")  * (3 * b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) * (3 * b("a,b,c")));

}

BOOST_AUTO_TEST_CASE( scale_mult )
{
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (     a("a,b,c")  *      b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) *      b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (     a("a,b,c")  * (3 * b("a,b,c"))));
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) * (3 * b("a,b,c"))));

}

BOOST_AUTO_TEST_CASE( outer_product )
{
  // Generate Eigen matrices from input arrays.
  EigenMatrixXi ev = make_matrix(v);
  EigenMatrixXi eu = make_matrix(u);

  // Generate the expected result
  EigenMatrixXi ew_test = eu * ev.transpose();

  // Test that outer product works
  BOOST_REQUIRE_NO_THROW(w("i,j") = u("i") * v("j"));

  GlobalFixture::world->gop.fence();

  EigenMatrixXi ew = make_matrix(w);

  BOOST_CHECK_EQUAL(ew, ew_test);
}

BOOST_AUTO_TEST_SUITE_END()
