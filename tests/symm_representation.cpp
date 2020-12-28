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

#include <chrono>
#include <iostream>
#include <random>

#include "TiledArray/symm/permutation_group.h"
#include "TiledArray/symm/representation.h"
#include "unit_test_config.h"

using TiledArray::symmetry::Permutation;
using TiledArray::symmetry::PermutationGroup;
using TiledArray::symmetry::Representation;
using TiledArray::symmetry::SymmetricGroup;

struct GroupRepresentationFixture {
  GroupRepresentationFixture()
      : generator(std::chrono::system_clock::now().time_since_epoch().count()),
        uniform_int_distribution(0, 100) {}

  ~GroupRepresentationFixture() {}

  template <size_t N>
  std::array<int, N> random_index() {
    std::array<int, N> result;
    for (auto& value : result) value = uniform_int_distribution(generator);
    return result;
  }

  // random number generation
  std::default_random_engine generator;
  std::uniform_int_distribution<int> uniform_int_distribution;

};  // GroupRepresentationFixture

struct U1_Operator {
  enum operator_type {
    _i = 0,
    _n = 1,
    _cc = 2,
    _n_cc = 3
  };  // bitwise encoding; multiplication = XOR
 public:
  U1_Operator(operator_type t = _i) : type_(t) {}
  U1_Operator(const U1_Operator&) = default;

  static U1_Operator identity;
  static U1_Operator negate;
  static U1_Operator complex_conjugate;
  static U1_Operator negate_complex_conjugate;

  // computes *this * rhs
  U1_Operator operator*(const U1_Operator& rhs) const {
    return U1_Operator(static_cast<operator_type>(type_ ^ rhs.type_));
  }
  // compares *this and rhs
  bool operator==(const U1_Operator& rhs) const { return type_ == rhs.type_; }

 private:
  operator_type type_;
};

U1_Operator U1_Operator::identity = U1_Operator{_i};
U1_Operator U1_Operator::negate = U1_Operator{_n};
U1_Operator U1_Operator::complex_conjugate = U1_Operator{_cc};
U1_Operator U1_Operator::negate_complex_conjugate = U1_Operator{_n_cc};

namespace TiledArray {
namespace symmetry {
template <>
U1_Operator identity<U1_Operator>() {
  return U1_Operator::identity;
}
}  // namespace symmetry
}  // namespace TiledArray

BOOST_FIXTURE_TEST_SUITE(symm_representation_suite, GroupRepresentationFixture, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(constructor) {
  // representation for permutation symmetry
  // <01||23> = -<10||23> = -<01||32> = <10||32> = <23||01>* = -<23||10>* =
  // -<32||01>* = <32||10>*
  {
    // generators are permutations (in cycle notation): (0,1), (2,3), and
    // (0,2)(1,3) the corresponding operators are negate, negate, and
    // compl_conjugate
    std::map<Permutation, U1_Operator> genops;
    genops[Permutation{1, 0, 2, 3}] = U1_Operator::negate;
    genops[Permutation{0, 1, 3, 2}] = U1_Operator::negate;
    genops[Permutation{2, 3, 0, 1}] = U1_Operator::complex_conjugate;

    Representation<PermutationGroup, U1_Operator> rep(genops);

    BOOST_CHECK_EQUAL(rep.order(), 8u);

    for (const auto& g_op_pair : rep.representatives()) {
      auto g = g_op_pair.first;
      auto op = g_op_pair.second;

      if (g == Permutation{0, 1, 2, 3})
        BOOST_CHECK(op == U1_Operator::identity);
      if (g == Permutation{1, 0, 2, 3}) BOOST_CHECK(op == U1_Operator::negate);
      if (g == Permutation{0, 1, 3, 2}) BOOST_CHECK(op == U1_Operator::negate);
      if (g == Permutation{1, 0, 3, 2})
        BOOST_CHECK(op == U1_Operator::identity);
      if (g == Permutation{2, 3, 0, 1})
        BOOST_CHECK(op == U1_Operator::complex_conjugate);
      if (g == Permutation{2, 3, 1, 0})
        BOOST_CHECK(op == U1_Operator::negate_complex_conjugate);
      if (g == Permutation{3, 2, 0, 1})
        BOOST_CHECK(op == U1_Operator::negate_complex_conjugate);
      if (g == Permutation{3, 2, 1, 0})
        BOOST_CHECK(op == U1_Operator::complex_conjugate);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
