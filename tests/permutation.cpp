#include "TiledArray/permutation.h"
#include "TiledArray/coordinates.h" // for boost array output
#include "unit_test_config.h"

using namespace TiledArray;

struct PermutationFixture {
  PermutationFixture() : p(2,0,1) {}
  ~PermutationFixture() {}
  Permutation p;
};

BOOST_FIXTURE_TEST_SUITE( permutation_suite, PermutationFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW( Permutation p0 ); // check default constructor
  Permutation p0;
  BOOST_CHECK_EQUAL(p0.data().size(), 0ul);

  BOOST_REQUIRE_NO_THROW( Permutation p1(0,1,2) ); // check variable list constructor
  Permutation p1(0,1,2);
  BOOST_CHECK_EQUAL(p1.data()[0], 0u);
  BOOST_CHECK_EQUAL(p1.data()[1], 1u);
  BOOST_CHECK_EQUAL(p1.data()[2], 2u);

  std::array<std::size_t,3> a = {{0, 1, 2}};
  BOOST_REQUIRE_NO_THROW( Permutation p2(a) ); // check boost array constructor
  Permutation p2(a);
  BOOST_CHECK_EQUAL(p2.data()[0], 0u);
  BOOST_CHECK_EQUAL(p2.data()[1], 1u);
  BOOST_CHECK_EQUAL(p2.data()[2], 2u);

  BOOST_REQUIRE_NO_THROW( Permutation p3(a.begin(), a.end()) ); // check iterator constructor
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

BOOST_AUTO_TEST_CASE( comparision )
{
  Permutation p0(0,1,2);
  Permutation p1(p);
  BOOST_CHECK( p1 == p );    // check operator==()
  BOOST_CHECK( ! (p1 == p0) );
  BOOST_CHECK( p1 != p0 );   // check operator!=()
  BOOST_CHECK( ! (p1 != p) );
}

BOOST_AUTO_TEST_CASE( permute_function )
{
  {
    std::vector<int> a1(3);
    a1[0] = 1; a1[1] = 2; a1[2] = 3;
    std::vector<int> ar(3);
    ar[0] = 2; ar[1] = 3; ar[2] = 1;
    std::vector<int> a2(3);
    detail::permute_array(p.begin(), p.end(), a1.begin(), a2.begin());
    BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end()); // check permutation applied via detail::permute_array()
  }
  {
    std::array<int, 3> a1 = {{1, 2, 3}};
    std::array<int, 3> ar = {{2, 3, 1}};
    std::array<int, 3> a2;
    detail::permute_array(p.begin(), p.end(), a1.begin(), a2.begin());
    BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end()); // check permutation applied via detail::permute()
  }
}

BOOST_AUTO_TEST_CASE( permute_permutation )
{
  Permutation pr(1,2,0);
  Permutation p0(0,1,2);
  Permutation p1 = p ^ p0;
  BOOST_CHECK_EQUAL(p1, pr); // check assignment permutation permutation
  Permutation p2(0,1,2);
  p2 ^= p;
  BOOST_CHECK_EQUAL(p2, pr); // check in-place permutation permutation.
}

BOOST_AUTO_TEST_CASE( reverse_permutation )
{
  Permutation p0(p);
  Permutation pr(1,2,0);
  BOOST_CHECK_EQUAL(-p0, pr);
  BOOST_CHECK_EQUAL((p0 ^ (p0)), Permutation(0,1,2));
  Permutation p1(1,2,0);
  p0 ^= p1;
  BOOST_CHECK_NE(p0, p);
  p0 ^= -p1;
  BOOST_CHECK_EQUAL(p0, p);
}

BOOST_AUTO_TEST_CASE( boost_array_permutation )
{
  std::array<int, 3> a1 = {{1, 2, 3}};
  std::array<int, 3> ar = {{2, 3, 1}};
  std::array<int, 3> a2 = p ^ a1;
  std::array<int, 3> a3 = a1;
  a3 ^= p;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), ar.begin(), ar.end()); // check assignment permutation
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), ar.begin(), ar.end()); // check in-place permutation
}

BOOST_AUTO_TEST_CASE( vector_permutation )
{
  std::vector<int> a1(3); int a1v[3] = {1, 2, 3}; std::copy(a1v, a1v+3, a1.begin());
  std::vector<int> ar(3); int arv[3] = {2, 3, 1}; std::copy(arv, arv+3, ar.begin());
  std::vector<int> a2 = p ^ a1;
  std::vector<int> a3 = a1;
  a3 ^= p;
  BOOST_CHECK(a2 == ar); // check assignment permutation
  BOOST_CHECK(a3 == ar); // check in-place permutation
}

BOOST_AUTO_TEST_SUITE_END()
