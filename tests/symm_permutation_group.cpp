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
 *  Edward Valeev, Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  symm_permutation_group.cpp
 *  May 14, 2015
 *
 */

#include <chrono>
#include <iostream>
#include <random>

#include "TiledArray/symm/permutation_group.h"
#include "unit_test_config.h"

using TiledArray::symmetry::Permutation;
using TiledArray::symmetry::PermutationGroup;
using TiledArray::symmetry::SymmetricGroup;

struct PermutationGroupFixture {
  PermutationGroupFixture()
      : generator(std::chrono::system_clock::now().time_since_epoch().count()),
        uniform_int_distribution(0, 100) {
    {  // construct set of generators for P4__01__23__02_13 =
       // P4{(0,1),(2,3),(0,2)(1,3)}
      // this group describes symmetries under permutations 0<->1, 2<->3, and
      // {0,1}<->{2,3}
      P4__01__23__02_13_generators.reserve(3);
      P4__01__23__02_13_generators.emplace_back(Permutation{1, 0, 2, 3});
      P4__01__23__02_13_generators.emplace_back(Permutation{0, 1, 3, 2});
      P4__01__23__02_13_generators.emplace_back(Permutation{2, 3, 0, 1});
    }
  }

  ~PermutationGroupFixture() {}

  template <size_t N>
  std::array<int, N> random_index() {
    std::array<int, N> result;
    for (auto& value : result) value = uniform_int_distribution(generator);
    return result;
  }

  // random number generation
  std::default_random_engine generator;
  std::uniform_int_distribution<int> uniform_int_distribution;

  // for testing symmetric group
  static const unsigned int max_degree = 4u;

  std::vector<Permutation> P4__01__23__02_13_generators;

  void validate_group(const PermutationGroup& S) {
    // Check that the group includes the identity element
    BOOST_CHECK_EQUAL(S.identity(), Permutation());
    for (unsigned int i = 0u; i < S.order(); ++i) {
      BOOST_CHECK_EQUAL(S.identity() * S[i], S[i]);
      BOOST_CHECK_EQUAL(S[i] * S.identity(), S[i]);
    }

    // Check that the group forms a closed set
    for (unsigned int i = 0u; i < S.order(); ++i) {
      for (unsigned int j = 0u; j < S.order(); ++j) {
        Permutation e = S[i] * S[j];

        unsigned int k = 0u;
        for (; k < S.order(); ++k) {
          if (e == S[k]) break;
        }

        // Check that e is a member of the group
        BOOST_CHECK(k < S.order());
      }
    }

    // Check that the elements of the set are associative
    for (unsigned int i = 0u; i < S.order(); ++i) {
      for (unsigned int j = 0u; j < S.order(); ++j) {
        for (unsigned int k = 0u; k < S.order(); ++k) {
          BOOST_CHECK_EQUAL((S[i] * S[j]) * S[k], S[i] * (S[j] * S[k]));
        }
      }
    }

    // Check that the group contains the inverse of each element
    for (unsigned int i = 0u; i < S.order(); ++i) {
      Permutation inv = S[i].inv();

      // Search for the inverse of S[i]
      unsigned int j = 0u;
      for (; j < S.order(); ++j)
        if (inv == S[j]) break;

      // Check that inv is a member of the group
      BOOST_CHECK(j < S.order());

      // Check that the any element multiplied by its own inverse is the
      // identity
      BOOST_CHECK_EQUAL(inv * S[i], S.identity());
      BOOST_CHECK_EQUAL(S[i] * inv, S.identity());
    }
  }

};  // PermutationGroupFixture

BOOST_FIXTURE_TEST_SUITE(symm_group_suite, PermutationGroupFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(constructor) {
  // SymmetricGroup "degree" ctor
  {
    unsigned int order = 1u;
    for (unsigned int degree = 1u; degree <= max_degree;
         ++degree, order *= degree) {
      BOOST_REQUIRE_NO_THROW(SymmetricGroup S(degree));
      SymmetricGroup S(degree);

      // Check that the group has the correct degree
      BOOST_CHECK_EQUAL(S.degree(), degree);

      // Check that the number of elements in the group is correct
      BOOST_CHECK_EQUAL(S.order(), order);

      validate_group(S);
    }
  }

  // SymmetricGroup "domain" ctor
  {
    auto domain = {0, 7, 11, 15};
    BOOST_REQUIRE_NO_THROW(SymmetricGroup S(domain));
    SymmetricGroup S(domain.begin(), domain.end());

    // Check that the group has the correct degree
    BOOST_CHECK_EQUAL(S.degree(), 4);

    // Check that the number of elements in the group is correct
    BOOST_CHECK_EQUAL(S.order(), 4 * 3 * 2 * 1);

    validate_group(S);
  }

  // PermutationGroup ctor
  {
    PermutationGroup P4__01__23__02_13(P4__01__23__02_13_generators);

    // Check that the number of elements in the group is correct
    BOOST_CHECK_EQUAL(P4__01__23__02_13.order(), 8u);

    validate_group(P4__01__23__02_13);
  }
}

BOOST_AUTO_TEST_CASE(equality) {
  {  // make S1 in 2 different ways (this also checks that trivial generators
     // are skipped)
    SymmetricGroup S1(1);
    auto I = Permutation{0, 1};
    PermutationGroup P(std::vector<Permutation>{I});
    BOOST_CHECK(S1 == P);
  }
  {  // make S2 in 2 different ways
    SymmetricGroup S2(2);
    auto p10 = Permutation{1, 0};
    PermutationGroup P(std::vector<Permutation>{p10});
    BOOST_CHECK(S2 == P);
  }
  {  // make S3 in 3 different ways
    SymmetricGroup S3(3);
    PermutationGroup P1(
        std::vector<Permutation>{Permutation{1, 0}, Permutation{2, 1, 0}});
    PermutationGroup P2(
        std::vector<Permutation>{Permutation{1, 2, 0}, Permutation{0, 2, 1}});
    BOOST_CHECK(S3 == P1);
    BOOST_CHECK(S3 == P2);
    BOOST_CHECK(P1 == P2);
  }
}

BOOST_AUTO_TEST_CASE(comparison) {
  {
    SymmetricGroup S2(2);
    PermutationGroup P1(std::vector<Permutation>{
        Permutation{1, 2, 0}});  // cyclic subgroup of S3
    SymmetricGroup S3(3);
    BOOST_CHECK(S2 < S3);
    BOOST_CHECK(S3 < P1);
  }
}

BOOST_AUTO_TEST_CASE(domain) {
  {  // symmetric group on a "sparse" index domain
    auto domain = {0, 7, 11, 15};
    SymmetricGroup S(domain);

    auto computed_domain = S.domain<std::set<unsigned int>>();
    BOOST_CHECK(computed_domain.size() == domain.size());
    for (auto e : computed_domain) {
      BOOST_CHECK(std::find(domain.begin(), domain.end(), e) != domain.end());
    }
  }
  {  // permutation group on a sparse domain
    std::vector<Permutation> gens;
    gens.emplace_back(Permutation{0, 1, 2, 4, 5, 3});
    gens.emplace_back(Permutation{0, 1, 3, 2});
    auto ref_domain = {2, 3, 4,
                       5};  // this is the domain of the above 2 permutations

    PermutationGroup P(gens);

    auto computed_domain = P.domain<std::set<unsigned int>>();
    BOOST_CHECK(computed_domain.size() == ref_domain.size());
    for (auto e : computed_domain) {
      BOOST_CHECK(std::find(ref_domain.begin(), ref_domain.end(), e) !=
                  ref_domain.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(conjugation) {
  {  // symmetric group is invariant under any permutation in it
    auto domain = {0, 2, 3, 5};
    SymmetricGroup S(domain);
    Permutation p({2, 1, 5, 0, 4, 3, 6, 7});
    BOOST_CHECK(conjugate(S, p) == S);
  }
  {  // shift symmetric group to a different domain
    auto domain = {0, 2, 3, 5};
    SymmetricGroup S(domain);
    // shift the domain to {1,2,4,7}
    Permutation p({1, 0, 2, 4, 3, 7, 6, 5});
    auto new_domain = {1, 2, 4, 7};
    SymmetricGroup S_shifted_ref(new_domain);

    auto S_shifted = conjugate(S, p);
    BOOST_CHECK(S_shifted == S_shifted_ref);
  }
  {  // another example
    PermutationGroup P4__01__23__02_13(P4__01__23__02_13_generators);
    // {0,1,2,3} -> {0,2,1,3}
    Permutation p({0, 2, 1, 3});
    auto P4__02__13__01_23 = conjugate(P4__01__23__02_13, p);
    BOOST_CHECK(P4__01__23__02_13 != P4__02__13__01_23);
  }
}

BOOST_AUTO_TEST_CASE(intersection) {
  {  // S2 is a subgroup of S3
    SymmetricGroup S2(2);
    SymmetricGroup S3(3);
    BOOST_CHECK(intersect(S2, S3) == S2);
    {  // another S2 is a subgroup of S3
      SymmetricGroup S2({0, 2});
      BOOST_CHECK(intersect(S2, S3) == S2);
    }
    {  // yet another S2 is a subgroup of S3
      SymmetricGroup S2{1, 2};
      BOOST_CHECK(intersect(S2, S3) == S2);
    }
  }
}

BOOST_AUTO_TEST_CASE(set_stabilizer) {
  {  // S2{0,1} is a subgroup of S3{0,1,2} that fixes {2}
    SymmetricGroup S2(2);
    SymmetricGroup S3(3);
    BOOST_CHECK(stabilizer(S3, std::vector<int>{2}) == S2);
    {  // and another S2
      SymmetricGroup S2({0, 2});
      BOOST_CHECK(stabilizer(S3, std::vector<int>{1}) == S2);
    }
    {  // and another S2
      SymmetricGroup S2({1, 2});
      BOOST_CHECK(stabilizer(S3, std::vector<int>{0}) == S2);
    }

    // S1{0} is a subgroup of S3{0,1,2} that fixes {1,2}
    SymmetricGroup S1(1);
    BOOST_CHECK(stabilizer(S3, std::vector<int>{1, 2}) == S1);
    {  // and another S1
      SymmetricGroup S1({1});
      BOOST_CHECK(stabilizer(S3, std::vector<int>{0, 2}) == S1);
    }
    {  // and another S1
      SymmetricGroup S2({2});
      BOOST_CHECK(stabilizer(S3, std::vector<int>{0, 1}) == S1);
    }
  }
}

BOOST_AUTO_TEST_CASE(lexicographical_order) {
  {  // check S5
    typedef std::array<int, 5> index_type;
    SymmetricGroup S5(5);

    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 3, 4, 5}}, S5), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 1, 300, 300, 500}}, S5),
        true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{300, 300, 300, 300, 5}}, S5),
        false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 1, 0, 0, 5}}, S5), false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 3, 4, 0}}, S5), false);
  }

  {  // check P5{(0,2,4)(1,3)}
    typedef std::array<int, 5> index_type;
    PermutationGroup P(std::vector<Permutation>{Permutation{4, 3, 0, 1, 2}});

    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 3, 4, 5}}, P), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 3, 1, 4, 1}}, P), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 1, 1, 1}}, P), false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 1, 2, 0, 3}}, P), false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 2, 3, 1}}, P), false);
  }

  {  // check P4{(0,1),(2,3),(0,2)(1,3)}
    typedef std::array<int, 4> index_type;
    PermutationGroup P(P4__01__23__02_13_generators);

    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 3, 4}}, P), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 3, 2, 4}}, P), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 2, 4}}, P), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{2, 3, 2, 3}}, P), true);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 2, 3, 2}}, P), false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{2, 1, 3, 4}}, P), false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{1, 3, 1, 2}}, P), false);
    BOOST_CHECK_EQUAL(
        is_lexicographically_smallest(index_type{{2, 3, 1, 4}}, P), false);
  }
}

BOOST_AUTO_TEST_SUITE_END()
