#include "coordinates.h"
#include "permutation.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include "iterationtest.h"

using namespace TiledArray;

struct ArrayCoordinateFixture {
  typedef ArrayCoordinate<std::size_t, 3, LevelTag<0> > Point3;
  typedef ArrayCoordinate<std::size_t, 3, LevelTag<0>,
      CoordinateSystem<2, detail::increasing_dimension_order> > FPoint3;
  ArrayCoordinateFixture() : p(1,2,3) {
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
  }
  ~ArrayCoordinateFixture() {}

  boost::array<Point3::index, 3> a;
  Point3 p;
};

BOOST_FIXTURE_TEST_SUITE( array_coordinate_suite, ArrayCoordinateFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW( Point3 p1() );     // construct without exception
  Point3 p2;                               // default construction
  BOOST_CHECK_EQUAL( p2.data().size(), 3); // correct size
  BOOST_CHECK_EQUAL( p2.data()[0], 0);     // correct element initialization
  BOOST_CHECK_EQUAL( p2.data()[1], 0);
  BOOST_CHECK_EQUAL( p2.data()[2], 0);

  BOOST_REQUIRE_NO_THROW(Point3 p3(a.begin(), a.end())); // Iterator Constructor
  Point3 p3(a.begin(), a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(p3.data().begin(), p3.data().end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(Point3 p4(a)); // Boost array constructor
  Point3 p4(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(p3.data().begin(), p3.data().end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(Point3 p5(p));  // Copy constructor
  Point3 p5(p);
  BOOST_CHECK_EQUAL_COLLECTIONS(p5.data().begin(), p5.data().end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(Point3 p6(1));  // Assign constant constuctor
  Point3 p6(1);
  BOOST_CHECK_EQUAL( p6.data()[0], 1);     // correct element initialization
  BOOST_CHECK_EQUAL( p6.data()[1], 1);
  BOOST_CHECK_EQUAL( p6.data()[2], 1);

  BOOST_REQUIRE_NO_THROW(Point3 p7(1,2,3)); // variable argument list constructor
  Point3 p7(1,2,3);
  BOOST_CHECK_EQUAL_COLLECTIONS(p7.data().begin(), p7.data().end(), a.begin(), a.end());

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(Point3 p8(std::forward<Point3>(Point3(1)))); // Move constructor.
  Point3 p8(Point3(1));
  BOOST_CHECK_EQUAL( p8.data()[0], 1);     // correct element initialization
  BOOST_CHECK_EQUAL( p8.data()[1], 1);
  BOOST_CHECK_EQUAL( p8.data()[2], 1);
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( make_functions )
{
  Point3 p1 = Point3::make(1,2,3);
  BOOST_TEST_MESSAGE("Class Make Function");
  BOOST_CHECK_EQUAL_COLLECTIONS(p1.data().begin(), p1.data().end(), a.begin(), a.end()); // check for correct creation of point

  Point3 p2 = make_coord<Point3>(1,2,3);
  BOOST_TEST_MESSAGE("Free Make Function");
  BOOST_CHECK_EQUAL_COLLECTIONS(p2.data().begin(), p2.data().end(), a.begin(), a.end()); // check for correct creation of point
}

BOOST_AUTO_TEST_CASE( element_access )
{
  BOOST_CHECK_EQUAL( p[0], 1);            // correct element access
  BOOST_CHECK_EQUAL( p[1], 2);
  BOOST_CHECK_EQUAL( p[2], 3);
  BOOST_CHECK_EQUAL( p.at(0), 1);         // correct element access
  BOOST_CHECK_EQUAL( p.at(1), 2);
  BOOST_CHECK_EQUAL( p.at(2), 3);
#ifdef NEDBUG
  BOOST_CHECK_NO_TRHOW( p[3] );
#endif
  BOOST_CHECK_THROW( p.at(3), std::out_of_range);
}

BOOST_AUTO_TEST_CASE( iteration )
{
  BOOST_TEST_MESSAGE("iterator begin, end, and dereferenc");
  boost::array<std::size_t, 3> a = {{1, 2, 3}};
  BOOST_CHECK_EQUAL( const_iteration_test(p, a.begin(), a.end()), 3); // check for basic iteration functionality

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
  BOOST_CHECK_EQUAL( p1[0], 4); // check individual element assignment.
  p1.at(1) = 5;
  BOOST_CHECK_EQUAL( p1.at(1), 5); // check individual element assignment with range checking.
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  Point3 p2;
  p2 = Point3(1,2,3); // check move assignment.
  BOOST_CHECK_EQUAL_COLLECTIONS( p.begin(), p.end(), p2.begin(), p2.end()); // check for equality
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << p;
  BOOST_CHECK( !output.is_empty( false ) );       // check that the string was assigned.
  BOOST_CHECK( output.check_length( 9, false ) ); // check for correct length.
  BOOST_CHECK( output.is_equal( "(1, 2, 3)" ) );  // check for correct output.
}

BOOST_AUTO_TEST_CASE( c_comparisons )
{
  // TODO: How do we remove the requirement of appending ul to the indices in a
  // 2D coordinate constructor?
  typedef ArrayCoordinate<std::size_t, 2, LevelTag<0> > Point2;
  Point2 p0(0ul,0ul);
  Point2 p1(0ul,1ul);
  Point2 p2(0ul,2ul);
  Point2 p3(1ul,0ul);
  Point2 p4(1ul,1ul);
  Point2 p5(1ul,2ul);
  Point2 p6(2ul,0ul);
  Point2 p7(2ul,1ul);
  Point2 p8(2ul,2ul);
  Point2 pp(1ul,1ul);

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

BOOST_AUTO_TEST_CASE( fortran_comparisons )
{
  typedef ArrayCoordinate<std::size_t, 2, LevelTag<0>,
      CoordinateSystem<2, detail::increasing_dimension_order> > FPoint2;
  FPoint2 p0(0ul,0ul);
  FPoint2 p1(1ul,0ul);
  FPoint2 p2(2ul,0ul);
  FPoint2 p3(0ul,1ul);
  FPoint2 p4(1ul,1ul);
  FPoint2 p5(2ul,1ul);
  FPoint2 p6(0ul,2ul);
  FPoint2 p7(1ul,2ul);
  FPoint2 p8(2ul,2ul);
  FPoint2 pp(1ul,1ul);

  BOOST_CHECK(p0 < pp);    // check for correct lexicographical comparisons
  BOOST_CHECK(p1 < pp);    // for fortran ordering.
  BOOST_CHECK(p2 < pp);
  BOOST_CHECK(p3 < pp);
  BOOST_CHECK(p3 <= pp);
  BOOST_CHECK(p4 <= pp);
  BOOST_CHECK(p4 == pp);
  BOOST_CHECK(p4 >= pp);
  BOOST_CHECK(p5 >= pp);
  BOOST_CHECK(p5 > pp);
  BOOST_CHECK(p6 > pp);
  BOOST_CHECK(p7 > pp);
  BOOST_CHECK(p8 > pp);
  BOOST_CHECK(p1 != pp);

  BOOST_CHECK( ! (p0 > pp) );
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

BOOST_AUTO_TEST_CASE( comparison_functions )
{
  Point3 p111(1,1,1);
  Point3 p222(2,2,2);
  Point3 p123(1,2,3);
  Point3 p012(0,1,2);
  Point3 p002(0,0,2);

  BOOST_CHECK(detail::less(p111.data(),p222.data()));
}

BOOST_AUTO_TEST_CASE( math )
{
  Point3 p1(1);
  Point3 p2(2);
  Point3 p3(3);
  Point3 pa(p1);

  BOOST_CHECK_EQUAL( p1 + p2, p3 );  // check addition operator
  BOOST_CHECK_EQUAL( p3 - p1, p2 );  // check subtraction operator
  BOOST_CHECK_EQUAL( pa += p2, p3 ); // check addition assignment operator
  BOOST_CHECK_EQUAL( pa -= p2, p1 ); // check subtraction assignment operator
}

BOOST_AUTO_TEST_CASE( c_incrmentation )
{
  Point3 p1(1);
  Point3 pa(p1);
  ++pa;
  BOOST_CHECK_EQUAL( pa[0], 1 );  // check that the point was incremented
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 2 );

  BOOST_CHECK_EQUAL( (--pa), p1 ); // check that decrement returns the decremented object.
  BOOST_CHECK_EQUAL( pa[0], 1 );   // check that the point was decremented.
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );

  BOOST_CHECK_EQUAL( (pa++), p1 ); // check that increment returns the initial value.
  BOOST_CHECK_EQUAL( pa[0], 1 );   // check that the point was incremented.
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 2 );

  --pa;
  --pa;
  BOOST_CHECK_EQUAL( (++pa), p1); // check that increment returns the incremented object.
  BOOST_CHECK_EQUAL( pa[0], 1 );  // check that the
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );

  BOOST_CHECK_EQUAL( (pa--), p1); // check that decrement returns the initial value.
  BOOST_CHECK_EQUAL( pa[0], 1 );  // check that the point was decremented.
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 0 );
}

BOOST_AUTO_TEST_CASE( fortran_incrmentation )
{
  FPoint3 p1(1);
  FPoint3 pa(p1);

  ++pa;
  BOOST_CHECK_EQUAL( pa[0], 2 );
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );

  BOOST_CHECK_EQUAL( (--pa), p1 );
  BOOST_CHECK_EQUAL( pa[0], 1 );
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );

  BOOST_CHECK_EQUAL( (pa++), p1 );
  BOOST_CHECK_EQUAL( pa[0], 2 );
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );

  --pa;
  --pa;
  BOOST_CHECK_EQUAL( (++pa), p1);
  BOOST_CHECK_EQUAL( pa[0], 1 );
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );

  BOOST_CHECK_EQUAL( (pa--), p1);
  BOOST_CHECK_EQUAL( pa[0], 0 );
  BOOST_CHECK_EQUAL( pa[1], 1 );
  BOOST_CHECK_EQUAL( pa[2], 1 );
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

BOOST_AUTO_TEST_SUITE_END()

