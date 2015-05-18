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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  symm_symm_group.cpp
 *  May 14, 2015
 *
 */

#include "TiledArray/symm/symm_group.h"
#include "unit_test_config.h"

struct SymmGroupFixture {

  SymmGroupFixture() { }

  ~SymmGroupFixture() { }

}; // SymmGroupFixture

using TiledArray::SymmGroup;
using TiledArray::Permutation;

BOOST_FIXTURE_TEST_SUITE( symm_group_suite, SymmGroupFixture )

BOOST_AUTO_TEST_CASE( constructor )
{

  unsigned int order = 1u;
  for(unsigned int degree = 1u; degree < 5u; ++degree, order *= degree) {
    BOOST_REQUIRE_NO_THROW(SymmGroup S(degree));
    SymmGroup S(degree);

    std::cout << "S(" << S.degree() << ")\n"
        << "  order = " << S.order() << "\n"
        << "  elements = {\n";
    for(unsigned int i = 0u; i < S.order(); ++i)
      std::cout << "    " << S[i] << "\n";
    std::cout << "}\n";

    // Check that the group has the correct degree
    BOOST_CHECK_EQUAL(S.degree(), degree);

    // Check that the number of elements in the group is correct
    BOOST_CHECK_EQUAL(S.order(), order);


    // Check that the group has the identity property
    BOOST_CHECK_EQUAL(S.identity(), Permutation::identity(degree));
    for(unsigned int i = 0u; i < S.order(); ++i) {
      BOOST_CHECK_EQUAL(S.identity() * S[i], S[i]);
      BOOST_CHECK_EQUAL(S[i] * S.identity(), S[i]);
    }


    // Check that the group forms a closed set
    for(unsigned int i = 0u; i < S.order(); ++i) {
      for(unsigned int j = 0u; j < S.order(); ++j) {
        Permutation e = S[i] * S[j];

        unsigned int k = 0u;
        for(; k < order; ++k) {
          if(e == S[k])
            break;
        }

        // Check that e is a member of the group
        BOOST_CHECK(k < order);
      }
    }

    // Check that the elements of the set are associative
    for(unsigned int i = 0u; i < S.order(); ++i) {
      for(unsigned int j = 0u; j < S.order(); ++j) {
        for(unsigned int k = 0u; k < S.order(); ++k) {

          BOOST_CHECK_EQUAL((S[i] * S[j]) * S[k], S[i] * (S[j] * S[k]));
        }
      }
    }

    // Check that the group contains an inverse element for each element
    for(unsigned int i = 0u; i < S.order(); ++i) {
      Permutation inv = S[i].inv();

      // Search for the inverse of S[i]
      unsigned int j = 0u;
      for(; j < S.order(); ++j)
        if(inv == S[j])
          break;

      // Check that inv is a member of the group
      BOOST_CHECK(j < order);

      // Check that the any element multiplied by it's own inverse is the identity
      BOOST_CHECK_EQUAL(inv * S[i], S.identity());
      BOOST_CHECK_EQUAL(S[i] * inv, S.identity());
    }

  }
}

BOOST_AUTO_TEST_SUITE_END()
