#include "TiledArray/coordinates.h"
#include "TiledArray/permutation.h"
#include <iostream>
#include "unit_test_config.h"
#include <world/bufar.h>

using namespace TiledArray;
using TiledArray::detail::LevelTag;

struct ArrayCoordinateFixture {
  typedef ArrayCoordinate<std::size_t, 3, LevelTag<0> > Point3;
  ArrayCoordinateFixture() : p(1,2,3) {
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
  }
  ~ArrayCoordinateFixture() {}

  std::array<Point3::value_type, 3> a;
  Point3 p;
};

BOOST_FIXTURE_TEST_SUITE( array_coordinate_suite, ArrayCoordinateFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW( Point3 p1() );     // construct without exception
  Point3 p2;                               // default construction
  BOOST_CHECK_EQUAL( p2.data().size(), 3u); // correct size
  BOOST_CHECK_EQUAL( p2.data()[0], 0u);     // correct element initialization
  BOOST_CHECK_EQUAL( p2.data()[1], 0u);
  BOOST_CHECK_EQUAL( p2.data()[2], 0u);

  BOOST_REQUIRE_NO_THROW(Point3 p3(a.begin())); // Iterator Constructor
  Point3 p3(a.begin());
  BOOST_CHECK_EQUAL_COLLECTIONS(p3.data().begin(), p3.data().end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(Point3 p4(a)); // Boost array constructor
  Point3 p4(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(p3.data().begin(), p3.data().end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(Point3 p5(p));  // Copy constructor
  Point3 p5(p);
  BOOST_CHECK_EQUAL_COLLECTIONS(p5.data().begin(), p5.data().end(), p.begin(), p.end());

  BOOST_REQUIRE_NO_THROW(Point3 p6());  // Assign constant constuctor
  Point3 p6;
  BOOST_CHECK_EQUAL( p6.data()[0], 0u);     // correct element initialization
  BOOST_CHECK_EQUAL( p6.data()[1], 0u);
  BOOST_CHECK_EQUAL( p6.data()[2], 0u);

  BOOST_REQUIRE_NO_THROW(Point3 p7(1,2,3)); // variable argument list constructor
  Point3 p7(1,2,3);
  BOOST_CHECK_EQUAL_COLLECTIONS(p7.data().begin(), p7.data().end(), a.begin(), a.end());
}


BOOST_AUTO_TEST_CASE( size )
{
  BOOST_CHECK_EQUAL( p.size() , 3ul );
}

BOOST_AUTO_TEST_CASE( element_access )
{
  BOOST_CHECK_EQUAL( p[0], 1u);            // correct element access
  BOOST_CHECK_EQUAL( p[1], 2u);
  BOOST_CHECK_EQUAL( p[2], 3u);
  BOOST_CHECK_EQUAL( p.at(0), 1u);         // correct element access
  BOOST_CHECK_EQUAL( p.at(1), 2u);
  BOOST_CHECK_EQUAL( p.at(2), 3u);
  BOOST_CHECK_THROW( p.at(3), Exception);
}

BOOST_AUTO_TEST_CASE( iteration )
{
  BOOST_TEST_MESSAGE("iterator begin, end, and dereferenc");
  std::array<std::size_t, 3> a = {{1, 2, 3}};
  BOOST_CHECK_EQUAL( const_iteration_test(p, a.begin(), a.end()), 3u); // check for basic iteration functionality

  Point3 p1(p);
  std::size_t i = 3;
  BOOST_TEST_MESSAGE("Iterator dereference assignment");
  for(Point3::iterator it = p1.begin(); it != p1.end(); ++it, ++i) {
    *it = i;
    BOOST_CHECK_EQUAL(*it, i); // check iterator assignment.
  }
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Point3 p1;
  p1 = p;
  BOOST_CHECK_EQUAL_COLLECTIONS( p.begin(), p.end(), p1.begin(), p1.end()); // check for equality
  p1[0] = 4;
  BOOST_CHECK_EQUAL( p1[0], 4u); // check individual element assignment.
  p1.at(1) = 5;
  BOOST_CHECK_EQUAL( p1.at(1), 5u); // check individual element assignment with range checking.
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << p;
  BOOST_CHECK( !output.is_empty( false ) );       // check that the string was assigned.
  BOOST_CHECK( output.check_length( 7, false ) ); // check for correct length.
  BOOST_CHECK( output.is_equal( "[1,2,3]" ) );  // check for correct output.
}

BOOST_AUTO_TEST_CASE( c_comparisons )
{
  // 2D coordinate constructor?
  typedef ArrayCoordinate<std::size_t, 2, LevelTag<0> > Point2;
  Point2 p0(0,0);
  Point2 p1(0,1);
  Point2 p2(0,2);
  Point2 p3(1,0);
  Point2 p4(1,1);
  Point2 p5(1,2);
  Point2 p6(2,0);
  Point2 p7(2,1);
  Point2 p8(2,2);
  Point2 pp(1,1);

  BOOST_CHECK(p0 < pp);  // check comparison operators for success.
  BOOST_CHECK(p1 < pp);  // check for correct lexicographical comparison order.
  BOOST_CHECK(p2 < pp);
  BOOST_CHECK(p3 < pp);
  BOOST_CHECK(p3 < pp);
  BOOST_CHECK(p4 <= pp);
  BOOST_CHECK(p4 == pp);
  BOOST_CHECK(p4 >= pp);
  BOOST_CHECK(p5 > pp);
  BOOST_CHECK(p5 > pp);
  BOOST_CHECK(p6 > pp);
  BOOST_CHECK(p7 > pp);
  BOOST_CHECK(p8 > pp);
  BOOST_CHECK(p1 != pp);

  BOOST_CHECK( ! (p0 > pp) ); // check for comparison failures.
  BOOST_CHECK( ! (p1 > pp) );
  BOOST_CHECK( ! (p2 > pp) );
  BOOST_CHECK( ! (p3 > pp) );
  BOOST_CHECK( ! (p3 >= pp) );
  BOOST_CHECK( ! (p4 != pp) );
  BOOST_CHECK( ! (p5 <= pp) );
  BOOST_CHECK( ! (p5 < pp) );
  BOOST_CHECK( ! (p6 < pp) );
  BOOST_CHECK( ! (p7 < pp) );
  BOOST_CHECK( ! (p8 < pp) );
  BOOST_CHECK( ! (p8 == pp) );
}

BOOST_AUTO_TEST_CASE( math )
{
  Point3 p1(1, 1, 1);
  Point3 p2(2, 2, 2);
  Point3 p3(3, 3, 3);
  Point3 pa(p1);

  BOOST_CHECK_EQUAL( p1 + p2, p3 );  // check addition operator
  BOOST_CHECK_EQUAL( p3 - p1, p2 );  // check subtraction operator
  BOOST_CHECK_EQUAL( pa += p2, p3 ); // check addition assignment operator
  BOOST_CHECK_EQUAL( pa -= p2, p1 ); // check subtraction assignment operator
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> perm(2, 0, 1);
  Point3 p1(p);
  Point3 p2(p);
  Point3 pr(2,3,1);
  BOOST_CHECK_EQUAL(p1 ^= perm, pr); // check in-place permutation
  BOOST_CHECK_EQUAL(p1, pr);
  BOOST_CHECK_EQUAL(perm ^ p2, pr); // check permutation
  BOOST_CHECK_EQUAL(p2, p);         // check that p2 is not modified
}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = sizeof(Point3) * 2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar & p;
  std::size_t nbyte = oar.size();
  oar.close();

  Point3 ps;
  madness::archive::BufferInputArchive iar(buf,nbyte);
  iar & ps;
  iar.close();

  delete [] buf;

  BOOST_CHECK_EQUAL(ps, p);
}

BOOST_AUTO_TEST_SUITE_END()

