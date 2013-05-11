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
  static void random_fill(Array<T, DIM, Tile>& x) {
    typename Array<T, DIM, Tile>::pmap_interface::const_iterator it = x.get_pmap()->begin();
    typename Array<T, DIM, Tile>::pmap_interface::const_iterator end = x.get_pmap()->end();
    for(; it != end; ++it)
      x.set(*it, x.get_world().taskq.add(& ExpressionsFixture::template make_rand_tile<Array<T, DIM, Tile> >,
          x.trange().make_tile_range(*it)));
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

BOOST_AUTO_TEST_CASE( outer_product )
{
  // Generate Eigen matrices from input arrays.
  EigenMatrixXi ev = array_to_eigen(v);
  EigenMatrixXi eu = array_to_eigen(u);

  // Generate the expected result
  EigenMatrixXi ew_test = eu * ev.transpose();

  // Test that outer produce works
  BOOST_CHECK_NO_THROW(w("i,j") = u("i") * v("j"));

  EigenMatrixXi ew = array_to_eigen(w);

  BOOST_CHECK_EQUAL(ew, ew_test);
}

BOOST_AUTO_TEST_SUITE_END()
