/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2021  Virginia Tech
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
 */

#include "../src/TiledArray/dist_array_vector.h"
#include "unit_test_config.h"

#include <array_fixture.h>

using namespace TiledArray;

using ArrayVecN = DistArrayVector<Tensor<int>, DensePolicy>;
using SpArrayVecN = DistArrayVector<Tensor<int>, SparsePolicy>;

// These are all of the template parameters we are going to test over
using tparams = boost::mpl::list<ArrayVecN, SpArrayVecN>;

BOOST_FIXTURE_TEST_SUITE(arrayvec_suite, ArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(constructors, ArrayVec, tparams) {
  const auto& arr = array<typename ArrayVec::policy_type>();

  // default ctor
  {
    BOOST_REQUIRE_NO_THROW(ArrayVec{});
    ArrayVec av;
    BOOST_CHECK(!av);  // empty vec is null
    BOOST_CHECK_EQUAL(av.size(), 0);
  }

  // vector with null arrays
  {
    const auto sz = 2;
    BOOST_REQUIRE_NO_THROW(ArrayVec{sz});
    ArrayVec av{sz};
    BOOST_CHECK(!av);  // nonempty vec of null arrays is null
    BOOST_CHECK_EQUAL(av.size(), sz);
  }

  // vector with nonnull arrays
  {
    const auto sz = 2;
    ArrayVec av{sz};
    BOOST_CHECK_EQUAL(av.size(), sz);
    BOOST_REQUIRE_NO_THROW(av[0] = arr);
    BOOST_REQUIRE_NO_THROW(av.at(1) = arr);
    BOOST_CHECK(av);  // nonempty vec of non-null arrays is nonnull
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(expressions, ArrayVec, tparams) {
  const auto& arr = array<typename ArrayVec::policy_type>();

  // array -> arrayvec
  {
    ArrayVec av;
    BOOST_REQUIRE_NO_THROW(av("i,j,k") = arr("i,j,k"));
    BOOST_CHECK_EQUAL(av.size(), 1);
  }

  // arrayvec op array -> arrayvec
  {
    ArrayVec av0, av1, av2, av3;
    av0("i,j,k") = arr("i,k,j");
    // these don't compile
    BOOST_REQUIRE_NO_THROW(av1("i,j,k") = av0("i,j,k") + arr("i,k,j"));
    BOOST_CHECK_EQUAL(av1.size(), 1);
    BOOST_REQUIRE_NO_THROW(av2("i,j,k") = arr("i,k,j") + av1("i,j,k"));
    BOOST_CHECK_EQUAL(av2.size(), 1);
    BOOST_REQUIRE_NO_THROW(av3("i,j,k") =
                               3.0 * (1 * av2("i,j,k") - 2 * arr("i,k,j")) *
                               av1("i,j,k"));
    BOOST_CHECK_EQUAL(av3.size(), 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()
