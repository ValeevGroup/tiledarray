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
  const Permutation p102 = Permutation({1,0,2});
  const Permutation p021 = Permutation({0,2,1});
  const Permutation p120 = Permutation({1,2,0});
  const Permutation p201 = Permutation({2,0,1});
  const Permutation p210 = Permutation({2,1,0});
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
  BOOST_CHECK_EQUAL(p201[0], 2u); // check that accessor is readable
  BOOST_CHECK_EQUAL(p201[1], 0u);
  BOOST_CHECK_EQUAL(p201[2], 1u);
  // no write access.
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << p201;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( 18, false ) );
  BOOST_CHECK( output.is_equal( "{0->2, 1->0, 2->1}" ) );
}

BOOST_AUTO_TEST_CASE( equality )
{
  Permutation p0213({0,2,1,3});

  // Check that identical permutations are equal
  BOOST_CHECK( p == p201 );

  // Check that a permutation is equal to itself
  BOOST_CHECK( p201 == p201 );

  // Check that permutations of equal size with different elements are not equal
  BOOST_CHECK( ! (p0213 == p201) );
  BOOST_CHECK( ! (p201 == p0213) );

  // Check that permutations of different sizes with the same leading elements
  // are not equal
  BOOST_CHECK( ! (p0213 == p021) );
  BOOST_CHECK( ! (p021 == p0213) );
}

BOOST_AUTO_TEST_CASE( inequality )
{
  Permutation p0213({0,2,1,3});

  // Check that different permutations are not equal
  BOOST_CHECK( p102 != p201 );

  // Check that identical permutations are not, not equal
  BOOST_CHECK( ! (p102 != p102) );
  BOOST_CHECK( ! (Permutation({1,0,2}) != p102) );

  // Check that permutations of equal size with different elements are not equal
  BOOST_CHECK( p120 != p021);
  BOOST_CHECK( p021 != p120);

  // Check that permutations of different sizes with the same leading elements
  // are not equal
  BOOST_CHECK( p021 != p0213);
  BOOST_CHECK( p0213 != p021);
}

BOOST_AUTO_TEST_CASE( less_than )
{
  Permutation p0213({0,2,1,3});

  // Check that a lexicographically smaller permutation is less than a larger
  // permutation
  BOOST_CHECK( p120 < p210 );
  BOOST_CHECK( p0213 < p120 );

  BOOST_CHECK( ! (p210 < p120) );
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

  // check permutation multiplication function
  BOOST_CHECK_EQUAL(   I.mult(   I),    I);
  BOOST_CHECK_EQUAL(p102.mult(   I), p102);
  BOOST_CHECK_EQUAL(p021.mult(   I), p021);
  BOOST_CHECK_EQUAL(p120.mult(   I), p120);
  BOOST_CHECK_EQUAL(p201.mult(   I), p201);
  BOOST_CHECK_EQUAL(p210.mult(   I), p210);

  BOOST_CHECK_EQUAL(   I.mult(p102), p102);
  BOOST_CHECK_EQUAL(p102.mult(p102),    I);
  BOOST_CHECK_EQUAL(p021.mult(p102), p120);
  BOOST_CHECK_EQUAL(p120.mult(p102), p021);
  BOOST_CHECK_EQUAL(p201.mult(p102), p210);
  BOOST_CHECK_EQUAL(p210.mult(p102), p201);

  BOOST_CHECK_EQUAL(   I.mult(p021), p021);
  BOOST_CHECK_EQUAL(p102.mult(p021), p201);
  BOOST_CHECK_EQUAL(p021.mult(p021),    I);
  BOOST_CHECK_EQUAL(p120.mult(p021), p210);
  BOOST_CHECK_EQUAL(p201.mult(p021), p102);
  BOOST_CHECK_EQUAL(p210.mult(p021), p120);

  BOOST_CHECK_EQUAL(   I.mult(p120), p120);
  BOOST_CHECK_EQUAL(p102.mult(p120), p210);
  BOOST_CHECK_EQUAL(p021.mult(p120), p102);
  BOOST_CHECK_EQUAL(p120.mult(p120), p201);
  BOOST_CHECK_EQUAL(p201.mult(p120),    I);
  BOOST_CHECK_EQUAL(p210.mult(p120), p021);

  BOOST_CHECK_EQUAL(   I.mult(p201), p201);
  BOOST_CHECK_EQUAL(p102.mult(p201), p021);
  BOOST_CHECK_EQUAL(p021.mult(p201), p210);
  BOOST_CHECK_EQUAL(p120.mult(p201),    I);
  BOOST_CHECK_EQUAL(p201.mult(p201), p120);
  BOOST_CHECK_EQUAL(p210.mult(p201), p102);

  BOOST_CHECK_EQUAL(   I.mult(p210), p210);
  BOOST_CHECK_EQUAL(p102.mult(p210), p120);
  BOOST_CHECK_EQUAL(p021.mult(p210), p201);
  BOOST_CHECK_EQUAL(p120.mult(p210), p102);
  BOOST_CHECK_EQUAL(p201.mult(p210), p021);
  BOOST_CHECK_EQUAL(p210.mult(p210),    I);

  // check permutation multiplication operator
  BOOST_CHECK_EQUAL(   I * I,    I);
  BOOST_CHECK_EQUAL(p102 * I, p102);
  BOOST_CHECK_EQUAL(p021 * I, p021);
  BOOST_CHECK_EQUAL(p120 * I, p120);
  BOOST_CHECK_EQUAL(p201 * I, p201);
  BOOST_CHECK_EQUAL(p210 * I, p210);

  BOOST_CHECK_EQUAL(   I * p102, p102);
  BOOST_CHECK_EQUAL(p102 * p102,    I);
  BOOST_CHECK_EQUAL(p021 * p102, p120);
  BOOST_CHECK_EQUAL(p120 * p102, p021);
  BOOST_CHECK_EQUAL(p201 * p102, p210);
  BOOST_CHECK_EQUAL(p210 * p102, p201);

  BOOST_CHECK_EQUAL(   I * p021, p021);
  BOOST_CHECK_EQUAL(p102 * p021, p201);
  BOOST_CHECK_EQUAL(p021 * p021,    I);
  BOOST_CHECK_EQUAL(p120 * p021, p210);
  BOOST_CHECK_EQUAL(p201 * p021, p102);
  BOOST_CHECK_EQUAL(p210 * p021, p120);

  BOOST_CHECK_EQUAL(   I * p120, p120);
  BOOST_CHECK_EQUAL(p102 * p120, p210);
  BOOST_CHECK_EQUAL(p021 * p120, p102);
  BOOST_CHECK_EQUAL(p120 * p120, p201);
  BOOST_CHECK_EQUAL(p201 * p120,    I);
  BOOST_CHECK_EQUAL(p210 * p120, p021);

  BOOST_CHECK_EQUAL(   I * p201, p201);
  BOOST_CHECK_EQUAL(p102 * p201, p021);
  BOOST_CHECK_EQUAL(p021 * p201, p210);
  BOOST_CHECK_EQUAL(p120 * p201,    I);
  BOOST_CHECK_EQUAL(p201 * p201, p120);
  BOOST_CHECK_EQUAL(p210 * p201, p102);

  BOOST_CHECK_EQUAL(   I * p210, p210);
  BOOST_CHECK_EQUAL(p102 * p210, p120);
  BOOST_CHECK_EQUAL(p021 * p210, p201);
  BOOST_CHECK_EQUAL(p120 * p210, p102);
  BOOST_CHECK_EQUAL(p201 * p210, p021);
  BOOST_CHECK_EQUAL(p210 * p210,    I);


  // check permutation multiply-assign operator
  Permutation x = I;
  BOOST_CHECK_EQUAL(x *= I,    I);
  x = p102;
  BOOST_CHECK_EQUAL(x *= I, p102);
  x = p021;
  BOOST_CHECK_EQUAL(x *= I, p021);
  x = p120;
  BOOST_CHECK_EQUAL(x *= I, p120);
  x = p201;
  BOOST_CHECK_EQUAL(x *= I, p201);
  x = p210;
  BOOST_CHECK_EQUAL(x *= I, p210);

  x = I;
  BOOST_CHECK_EQUAL(x *= p102, p102);
  x = p102;
  BOOST_CHECK_EQUAL(x *= p102,    I);
  x = p021;
  BOOST_CHECK_EQUAL(x *= p102, p120);
  x = p120;
  BOOST_CHECK_EQUAL(x *= p102, p021);
  x = p201;
  BOOST_CHECK_EQUAL(x *= p102, p210);
  x = p210;
  BOOST_CHECK_EQUAL(x *= p102, p201);

  x = I;
  BOOST_CHECK_EQUAL(x *= p021, p021);
  x = p102;
  BOOST_CHECK_EQUAL(x *= p021, p201);
  x = p021;
  BOOST_CHECK_EQUAL(x *= p021,    I);
  x = p120;
  BOOST_CHECK_EQUAL(x *= p021, p210);
  x = p201;
  BOOST_CHECK_EQUAL(x *= p021, p102);
  x = p210;
  BOOST_CHECK_EQUAL(x *= p021, p120);

  x = I;
  BOOST_CHECK_EQUAL(x *= p120, p120);
  x = p102;
  BOOST_CHECK_EQUAL(x *= p120, p210);
  x = p021;
  BOOST_CHECK_EQUAL(x *= p120, p102);
  x = p120;
  BOOST_CHECK_EQUAL(x *= p120, p201);
  x = p201;
  BOOST_CHECK_EQUAL(x *= p120,    I);
  x = p210;
  BOOST_CHECK_EQUAL(x *= p120, p021);

  x = I;
  BOOST_CHECK_EQUAL(x *= p201, p201);
  x = p102;
  BOOST_CHECK_EQUAL(x *= p201, p021);
  x = p021;
  BOOST_CHECK_EQUAL(x *= p201, p210);
  x = p120;
  BOOST_CHECK_EQUAL(x *= p201,    I);
  x = p201;
  BOOST_CHECK_EQUAL(x *= p201, p120);
  x = p210;
  BOOST_CHECK_EQUAL(x *= p201, p102);

  x = I;
  BOOST_CHECK_EQUAL(x *= p210, p210);
  x = p102;
  BOOST_CHECK_EQUAL(x *= p210, p120);
  x = p021;
  BOOST_CHECK_EQUAL(x *= p210, p201);
  x = p120;
  BOOST_CHECK_EQUAL(x *= p210, p102);
  x = p201;
  BOOST_CHECK_EQUAL(x *= p210, p021);
  x = p210;
  BOOST_CHECK_EQUAL(x *= p210,    I);
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
  // check permutation inverse function
  BOOST_CHECK_EQUAL(   I.inv(),    I);
  BOOST_CHECK_EQUAL(p102.inv(), p102);
  BOOST_CHECK_EQUAL(p021.inv(), p021);
  BOOST_CHECK_EQUAL(p120.inv(), p201);
  BOOST_CHECK_EQUAL(p201.inv(), p120);
  BOOST_CHECK_EQUAL(p210.inv(), p210);

  // check permutation inverse power
  BOOST_CHECK_EQUAL(   I ^ -1,    I);
  BOOST_CHECK_EQUAL(p102 ^ -1, p102);
  BOOST_CHECK_EQUAL(p021 ^ -1, p021);
  BOOST_CHECK_EQUAL(p120 ^ -1, p201);
  BOOST_CHECK_EQUAL(p201 ^ -1, p120);
  BOOST_CHECK_EQUAL(p210 ^ -1, p210);
}

BOOST_AUTO_TEST_CASE( array_permutation )
{
  std::array<int, 3> a1{{1, 2, 3}};
  std::array<int, 3> ar{{2, 3, 1}};
  std::array<int, 3> a2 = p * a1;
  std::array<int, 3> a3 = a1;
  a3 *= p;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end()); // check assignment permutation
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), ar.begin(), ar.end()); // check in-place permutation
}

BOOST_AUTO_TEST_CASE( vector_permutation )
{
  std::vector<int> a1{1, 2, 3};
  std::vector<int> ar{2, 3, 1};
  std::vector<int> a2 = p * a1;
  std::vector<int> a3 = a1;
  a3 *= p;
  BOOST_CHECK(a2 == ar); // check assignment permutation
  BOOST_CHECK(a3 == ar); // check in-place permutation
}

BOOST_AUTO_TEST_SUITE_END()
