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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  tile_op_permute.cpp
 *  May 7, 2013
 *
 */

#include "TiledArray/tile_op/permute.h"
#include "config.h"
#include "TiledArray/tensor.h"
#include "range_fixture.h"

struct PermuteFixture : public RangeFixture {

  PermuteFixture() :
    a(RangeFixture::r),
    b(RangeFixture::r),
    c(),
    perm(2,0,1)
  {
    GlobalFixture::world->srand(27);
    for(std::size_t i = 0ul; i < r.volume(); ++i) {
      a[i] = GlobalFixture::world->rand() / 101;
      b[i] = GlobalFixture::world->rand() / 101;
    }
  }

  ~PermuteFixture() { }

  Tensor<int> a;
  Tensor<int> b;
  Tensor<int> c;
  Permutation perm;
}; // PermuteFixture

BOOST_FIXTURE_TEST_SUITE( tile_op_permute_suite, PermuteFixture )

BOOST_AUTO_TEST_CASE( permute_function )
{
  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(math::permute(c, perm, a));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that the data in the new tile is correct
  for(std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm ^ a.range().idx(i)], a[i]);
  }
}

BOOST_AUTO_TEST_CASE( permute_unary )
{
  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(math::permute(c, perm, a, std::negate<int>()));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that the data in the new tile is correct
  for(std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm ^ a.range().idx(i)], -a[i]);
  }
}

BOOST_AUTO_TEST_CASE( permute_binary )
{
  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(math::permute(c, perm, a, b, std::plus<int>()));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that the data in the new tile is correct
  for(std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm ^ a.range().idx(i)], a[i] + b[i]);
  }
}

BOOST_AUTO_TEST_CASE( permute_op )
{
  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(c = perm ^ a);

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that the data in the new tile is correct
  for(std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm ^ a.range().idx(i)], a[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
