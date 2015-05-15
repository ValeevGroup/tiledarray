/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#include "TiledArray/permutation.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct PermutationFixture {

  PermutationFixture() {}
  ~PermutationFixture() {}

  Permutation p = Permutation({2,0,1});
  const Permutation I = Permutation::identity(3);
}; // struct Fixture

BOOST_FIXTURE_TEST_SUITE( permutation_suite, PermutationFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default constructor
  BOOST_REQUIRE_NO_THROW( Permutation p0 );
  Permutation p0;
  BOOST_CHECK_EQUAL(p0.data().size(), 0ul);

  // check variable list constructor
  BOOST_REQUIRE_NO_THROW( Permutation p1({0,1,2}) );
  Permutation p1({0,1,2});
  BOOST_CHECK_EQUAL(p1.data()[0], 0u);
  BOOST_CHECK_EQUAL(p1.data()[1], 1u);
  BOOST_CHECK_EQUAL(p1.data()[2], 2u);

  // check boost array constructor
  std::vector<unsigned int> a{0, 1, 2};
  BOOST_REQUIRE_NO_THROW( Permutation p2(a) );
  Permutation p2(a);
  BOOST_CHECK_EQUAL(p2.data()[0], 0u);
  BOOST_CHECK_EQUAL(p2.data()[1], 1u);
  BOOST_CHECK_EQUAL(p2.data()[2], 2u);

  // check iterator constructor
  BOOST_REQUIRE_NO_THROW( Permutation p3(a.begin(), a.end()) );
  Permutation p3(a.begin(), a.end());
  BOOST_CHECK_EQUAL(p3.data()[0], 0u);
  BOOST_CHECK_EQUAL(p3.data()[1], 1u);
  BOOST_CHECK_EQUAL(p3.data()[2], 2u);
}

BOOST_AUTO_TEST_CASE( iteration )
{
  std::array<std::size_t,3> a = {{2, 0, 1}};
  std::array<std::size_t,3>::const_iterator a_it = a.begin();
  for(Permutation::const_iterator it = p.begin(); it != p.end(); ++it, ++a_it)
    BOOST_CHECK_EQUAL(*it, *a_it); // check that basic iteration is correct
}

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL(p[0], 2u); // check that accessor is readable
  BOOST_CHECK_EQUAL(p[1], 0u);
  BOOST_CHECK_EQUAL(p[2], 1u);
  // no write access.
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << p;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( 18, false ) );
  BOOST_CHECK( output.is_equal( "{0->2, 1->0, 2->1}" ) );
}

BOOST_AUTO_TEST_CASE( equality )
{
  Permutation p1({0,2,1});
  Permutation p2({0,2,1});
  Permutation p3({0,2,1,3});

  // Check that identical permutations are equal
  BOOST_CHECK( p1 == p2 );

  // Check that a permutation is equal to itself
  BOOST_CHECK( p1 == p1 );

  // Check that permutations of equal size with different elements are not equal
  BOOST_CHECK( ! (p1 == p) );
  BOOST_CHECK( ! (p == p1) );

  // Check that permutations of different sizes with the same leading elements
  // are not equal
  BOOST_CHECK( ! (p1 == p3) );
  BOOST_CHECK( ! (p3 == p1) );
}

BOOST_AUTO_TEST_CASE( inequality )
{

  Permutation p1({0,2,1});
  Permutation p2({0,2,1});
  Permutation p3({0,2,1,3});

  // Check that different permutations are equal
  BOOST_CHECK( p1 != p );

  // Check that identical permutations are equal
  BOOST_CHECK( ! (p1 != p2) );

  // Check that a permutation is not, not-equal to itself
  BOOST_CHECK(! (p1 != p1) );

  // Check that permutations of equal size with different elements are not equal
  BOOST_CHECK( p1 != p);
  BOOST_CHECK( p != p1);

  // Check that permutations of different sizes with the same leading elements
  // are not equal
  BOOST_CHECK( p1 != p3);
  BOOST_CHECK( p3 != p1);
}

BOOST_AUTO_TEST_CASE( less_than )
{
  Permutation p1({0,2,1});
  Permutation p2({0,2,1});
  Permutation p3({0,2,1,3});

  // Check that a lexicographically smaller permutation is less than a larger
  // permutation
  BOOST_CHECK( p1 < p );

  BOOST_CHECK( ! (p < p1) );
}

BOOST_AUTO_TEST_CASE( permute_helper )
{
  {
    std::vector<int> a1({1, 2, 3});
    std::vector<int> ar({2, 3, 1});
    std::vector<int> a2(3);

    // check permutation applied via detail::permute_array()
    BOOST_CHECK_NO_THROW(detail::permute_array(p, a1, a2));
    BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end());
  }
  {
    std::array<int, 3> a1 = {{1, 2, 3}};
    std::array<int, 3> ar = {{2, 3, 1}};
    std::array<int, 3> a2;

    // check permutation applied via detail::permute()
    BOOST_CHECK_NO_THROW(detail::permute_array(p, a1, a2));
    BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end());
  }
}

BOOST_AUTO_TEST_CASE( identity )
{
  Permutation reference({0,1,2});
  BOOST_CHECK_EQUAL(Permutation::identity(3), reference);
  BOOST_CHECK_EQUAL(p.identity(), reference);
}

BOOST_AUTO_TEST_CASE( mult )
{
  Permutation expected({1,2,0});
  Permutation p0({0,1,2});

  // check permutation multiplication function
  BOOST_CHECK_EQUAL(p.mult(p0), expected);

  // check permutation multiplication function
  BOOST_CHECK_EQUAL(p * p0, expected);

  // check in-place permutation multiplication
  BOOST_CHECK_NO_THROW(p *= p0);
  BOOST_CHECK_EQUAL(p, expected); // check in-place permutation permutation.

  // Check that multiplication by the identity gives the original
  const Permutation p_12({0,2,1});
  BOOST_CHECK_EQUAL(p_12 * I, p_12);
  BOOST_CHECK_EQUAL(I * p_12, p_12);
}


BOOST_AUTO_TEST_CASE( pow )
{
  const Permutation p0231({0,2,3,1});

  // Check that powers of permutations are computed correctly
  BOOST_CHECK_EQUAL(p0231.pow(0), p0231.identity());
  BOOST_CHECK_EQUAL(p0231.pow(1), p0231);
  BOOST_CHECK_EQUAL(p0231.pow(2), p0231 * p0231);
  BOOST_CHECK_EQUAL(p0231.pow(3), p0231 * p0231 * p0231);
  BOOST_CHECK_EQUAL(p0231.pow(4), p0231 * p0231 * p0231 * p0231);
  BOOST_CHECK_EQUAL(p0231.pow(5), p0231 * p0231 * p0231 * p0231 * p0231);
  BOOST_CHECK_EQUAL(p0231.pow(6), p0231 * p0231 * p0231 * p0231 * p0231 * p0231);
  BOOST_CHECK_EQUAL(p0231.pow(7), p0231 * p0231 * p0231 * p0231 * p0231 * p0231 * p0231);
  BOOST_CHECK_EQUAL(p0231.pow(8), p0231 * p0231 * p0231 * p0231 * p0231 * p0231 * p0231 * p0231);

  // Check that inverse powers of permutations are computed correctly
  Permutation I0231 = p0231.inv();
  BOOST_CHECK_EQUAL(p0231.pow(-1), I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-2), I0231 * I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-3), I0231 * I0231 * I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-4), I0231 * I0231 * I0231 * I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-5), I0231 * I0231 * I0231 * I0231 * I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-6), I0231 * I0231 * I0231 * I0231 * I0231 * I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-7), I0231 * I0231 * I0231 * I0231 * I0231 * I0231 * I0231);
  BOOST_CHECK_EQUAL(p0231.pow(-8), I0231 * I0231 * I0231 * I0231 * I0231 * I0231 * I0231 * I0231);

}

BOOST_AUTO_TEST_CASE( inverse )
{
  const Permutation reference({1,2,0});

  // Check that the inverse function does not throw
  Permutation p_inv;
  BOOST_CHECK_NO_THROW(p_inv = p.inv());

  // Check the result of the inverse
  BOOST_CHECK_EQUAL(p.inv(), reference);

  // Check that the p * p_12^-1 == p_12^-1 * p == I
  const Permutation p_12({0,2,1});
  BOOST_CHECK_EQUAL(p_12 * p_12.inv(), I);
  BOOST_CHECK_EQUAL(p_12.inv() * p_12, I);

  // Check that inverse of the identity is the identity
  BOOST_CHECK_EQUAL(I.inv(), I);
}

BOOST_AUTO_TEST_CASE( array_permutation )
{
  std::array<int, 3> a1 = {{1, 2, 3}};
  std::array<int, 3> ar = {{2, 3, 1}};
  std::array<int, 3> a2 = p * a1;
  std::array<int, 3> a3 = a1;
  a3 *= p;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end()); // check assignment permutation
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), ar.begin(), ar.end()); // check in-place permutation
}

BOOST_AUTO_TEST_CASE( vector_permutation )
{
  std::vector<int> a1(3); int a1v[3] = {1, 2, 3}; std::copy(a1v, a1v+3, a1.begin());
  std::vector<int> ar(3); int arv[3] = {2, 3, 1}; std::copy(arv, arv+3, ar.begin());
  std::vector<int> a2 = p * a1;
  std::vector<int> a3 = a1;
  a3 *= p;
  BOOST_CHECK(a2 == ar); // check assignment permutation
  BOOST_CHECK(a3 == ar); // check in-place permutation
}

BOOST_AUTO_TEST_SUITE_END()
