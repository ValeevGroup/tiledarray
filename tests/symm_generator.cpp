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
 *  symm_generator.h
 *  May 13, 2015
 *
 */

#include "TiledArray/symm/generator.h"
#include "unit_test_config.h"

struct CyclePermutationFixture {

  CyclePermutationFixture() { }

  ~CyclePermutationFixture() { }

}; // CyclePermutationFixture

using TiledArray::CyclePermutation;
using TiledArray::GroupElement;
using TiledArray::GroupElement;
using TiledArray::Permutation;
using TiledArray::symm;
using TiledArray::antisymm;
using TiledArray::Exception;

BOOST_FIXTURE_TEST_SUITE( cyclic_permutation_suite, CyclePermutationFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  // Test that the basic constructor does not throw an exception
  BOOST_CHECK_NO_THROW(CyclePermutation({0,1},1));
  BOOST_CHECK_NO_THROW(CyclePermutation({0,1},-1));
}

#ifdef TA_EXCEPTION_ERROR
BOOST_AUTO_TEST_CASE( constructor_error )
{
  // Check that the size of the list is at least 2.
  BOOST_CHECK_THROW(CyclePermutation({0},1), Exception);

  // Check exception when the list is not sorted
  BOOST_CHECK_THROW(CyclePermutation({3,0,1},-1), Exception);

  // Check that there are no duplicates in the list
  BOOST_CHECK_THROW(CyclePermutation({0,1,3,3},1), Exception);

  // Check for assertion when the symmetry flag is not 1 or -1.
  BOOST_CHECK_THROW(CyclePermutation({0,1},0), Exception);
}
#endif // TA_EXCEPTION_ERROR


BOOST_AUTO_TEST_CASE( factory_functions )
{
  // Test the symmetric generator
  {
    CyclePermutation symm_gen = symm({0,1});
    BOOST_CHECK_EQUAL(symm_gen.size(), 2);
    BOOST_CHECK_EQUAL(symm_gen.symmetry(), 1);
    BOOST_CHECK_EQUAL(symm_gen[0], 0);
    BOOST_CHECK_EQUAL(symm_gen[1], 1);
  }

  // Test the antisymmetric generator
  {
    CyclePermutation antisymm_gen = antisymm({0,1,3});
    BOOST_CHECK_EQUAL(antisymm_gen.size(), 3);
    BOOST_CHECK_EQUAL(antisymm_gen.symmetry(), -1);
    BOOST_CHECK_EQUAL(antisymm_gen[0], 0);
    BOOST_CHECK_EQUAL(antisymm_gen[1], 1);
    BOOST_CHECK_EQUAL(antisymm_gen[2], 3);
  }

}

BOOST_AUTO_TEST_CASE( make_group_generators )
{

  CyclePermutation antisymm_gen = antisymm({0,1,3});

#ifdef TA_EXCEPTION_ERROR
  {
    std::vector<GroupElement> elements;
    // Test assertion for the presence of an identity permutation.
    BOOST_CHECK_THROW(antisymm_gen.append_generators(elements), Exception);

    // Test that the size of the identity element must be large enough for the
    // generator elements.
    elements.emplace_back(Permutation::identity(2), 1);
    BOOST_CHECK_THROW(antisymm_gen.append_generators(elements), Exception);

    // Test that the first permutation must be an identity permutation
    elements.front() = GroupElement(Permutation{0,2,1,3}, 1);
    BOOST_CHECK_THROW(antisymm_gen.append_generators(elements), Exception);

    // Test that the first permutation must be an identity permutation
    elements.front() = GroupElement(Permutation::identity(4), -1);
    BOOST_CHECK_THROW(antisymm_gen.append_generators(elements), Exception);
  }
#endif // TA_EXCEPTION_ERROR

  std::vector<GroupElement> elements;
  elements.emplace_back(Permutation::identity(4), 1);
  BOOST_CHECK_NO_THROW(antisymm_gen.append_generators(elements));
}

BOOST_AUTO_TEST_SUITE_END()
