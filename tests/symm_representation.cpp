/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  symm_repreentation.cpp
 *  September 12, 2015
 *
 */

#include <random>
#include <chrono>
#include <iostream>

#include "TiledArray/symm/permutation_group.h"
#include "TiledArray/symm/representation.h"
#include "TiledArray/symm/oper.h"
#include "unit_test_config.h"

using TiledArray::symmetry::Representation;
using TiledArray::symmetry::PermutationGroup;
using TiledArray::symmetry::SymmetricGroup;
using TiledArray::symmetry::Permutation;
using IKOper = TiledArray::symmetry::IKGroupOperator;

struct GroupRepresentationFixture {

  GroupRepresentationFixture() :
    generator(std::chrono::system_clock::now().time_since_epoch().count()),
    uniform_int_distribution(0,100)
  {
  }

  ~GroupRepresentationFixture() { }

  template <size_t N>
  std::array<int, N> random_index() {
    std::array<int, N> result;
    for(auto& value : result)
      value = uniform_int_distribution(generator);
    return result;
  }

  // random number generation
  std::default_random_engine generator;
  std::uniform_int_distribution<int> uniform_int_distribution;

}; // GroupRepresentationFixture

BOOST_FIXTURE_TEST_SUITE( symm_representation_suite, GroupRepresentationFixture )

BOOST_AUTO_TEST_CASE( constructor )
{

  // representation for permutation symmetry
  // <01||23> = -<10||23> = -<01||32> = <10||32> = <23||01>* = -<23||10>* = -<32||01>* = <32||10>*
  {

    // generators are permutations (in cycle notation): (0,1), (2,3), and (0,2)(1,3)
    // the corresponding operators are negate, negate, and compl_conjugate
    std::map<Permutation, IKOper> genops;
    genops[Permutation{1,0,2,3}] = IKOper::I();
    genops[Permutation{0,1,3,2}] = IKOper::I();
    genops[Permutation{2,3,0,1}] = IKOper::K();

    Representation<PermutationGroup, IKOper> rep(genops);

    BOOST_CHECK_EQUAL(rep.order(), 8u);

    for(const auto& g_op_pair: rep.representatives()) {
        auto g = g_op_pair.first;
        auto op = g_op_pair.second;

        if (g == Permutation{0,1,2,3}) BOOST_CHECK(op == IKOper::E());
        if (g == Permutation{1,0,2,3}) BOOST_CHECK(op == IKOper::I());
        if (g == Permutation{0,1,3,2}) BOOST_CHECK(op == IKOper::I());
        if (g == Permutation{1,0,3,2}) BOOST_CHECK(op == IKOper::E());
        if (g == Permutation{2,3,0,1}) BOOST_CHECK(op == IKOper::K());
        if (g == Permutation{2,3,1,0}) BOOST_CHECK(op == IKOper::IK());
        if (g == Permutation{3,2,0,1}) BOOST_CHECK(op == IKOper::IK());
        if (g == Permutation{3,2,1,0}) BOOST_CHECK(op == IKOper::K());
    }
  }

}

BOOST_AUTO_TEST_SUITE_END()
