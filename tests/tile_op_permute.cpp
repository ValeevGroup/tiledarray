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
#include "unit_test_config.h"
#include "TiledArray/tensor.h"
#include "tensor_fixture.h"

struct PermuteFixture : public TensorFixture {

  PermuteFixture() :
    a(make_rand_tensor(40, 31)),
    b(make_rand_tensor(40, 66)),
    c(),
    perm(make_perm())
  {}

  ~PermuteFixture() { }

  static Tensor<int> make_rand_tensor(const int range_seed, const int data_seed) {
    GlobalFixture::world->srand(range_seed);
    std::array<std::size_t, GlobalFixture::dim> start, finish;

    for(unsigned int i = 0ul; i < GlobalFixture::dim; ++i) {
      start[i] = 0ul;
      finish[i] = GlobalFixture::world->rand() % 10 + 5;
    }

    GlobalFixture::world->srand(data_seed);
    Tensor<int> tensor(Range(start, finish));
    for(std::size_t i = 0ul; i < tensor.size(); ++i)
      tensor[i] = GlobalFixture::world->rand() % 42;
    return tensor;
  }

  static Permutation make_perm() {
    std::array<std::size_t, GlobalFixture::dim> temp;
    for(std::size_t i = 0; i < temp.size(); ++i)
      temp[i] = i + 1;

    temp.back() = 0;

    return Permutation(temp);
  }

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
  BOOST_CHECK_EQUAL(c.range(), perm ^ a.range());

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
  BOOST_CHECK_EQUAL(c.range(), perm ^ a.range());

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
  BOOST_CHECK_EQUAL(c.range(), perm ^ a.range());

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
  BOOST_CHECK_EQUAL(c.range(), perm ^ a.range());

  // Check that the data in the new tile is correct
  for(std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm ^ a.range().idx(i)], a[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
