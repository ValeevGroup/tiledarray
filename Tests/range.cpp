#include "TiledArray/range.h"
#include "TiledArray/permutation.h"
#include "unit_test_config.h"
#include "range_fixture.h"

using namespace TiledArray;

RangeFixture::RangeFixture()
{
}

const RangeFixture::index RangeFixture::start         = fill_index<RangeFixture::index>(0);
const RangeFixture::index RangeFixture::finish        = fill_index<RangeFixture::index>(5);
const RangeFixture::size_array RangeFixture::size     = RangeFixture::finish.data();
const RangeFixture::size_array RangeFixture::weight   = GlobalFixture::coordinate_system::calc_weight(RangeFixture::size);
const RangeFixture::volume_type RangeFixture::volume  = GlobalFixture::coordinate_system::calc_volume(RangeFixture::size);
const RangeFixture::index RangeFixture::p0            = fill_index<RangeFixture::index>(0);
const RangeFixture::index RangeFixture::p1            = fill_index<RangeFixture::index>(1);
const RangeFixture::index RangeFixture::p2            = fill_index<RangeFixture::index>(2);
const RangeFixture::index RangeFixture::p3            = fill_index<RangeFixture::index>(3);
const RangeFixture::index RangeFixture::p4            = fill_index<RangeFixture::index>(4);
const RangeFixture::index RangeFixture::p5            = fill_index<RangeFixture::index>(5);
const RangeFixture::index RangeFixture::p6            = fill_index<RangeFixture::index>(6);


BOOST_FIXTURE_TEST_SUITE( range_suite, RangeFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL(r.start(), start);   // check start()
  BOOST_CHECK_EQUAL(r.finish(), finish);  // check finish()
  BOOST_CHECK_EQUAL(r.size(), size); // check size()
  BOOST_CHECK_EQUAL(r.weight(), weight); // check weight()
  BOOST_CHECK_EQUAL(r.volume(), volume);  // check volume()
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(RangeN r0); // Default Constructor
  RangeN r0;
  BOOST_CHECK_EQUAL(r0.start(), start);
  BOOST_CHECK_EQUAL(r0.finish(), start);
  BOOST_CHECK_EQUAL(r0.size(), start.data());
  BOOST_CHECK_EQUAL(r0.volume(), 0u);

  BOOST_REQUIRE_NO_THROW(RangeN r2(p2, p2 + finish)); // Start/Finish Constructor
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(RangeN r2(p2 + finish, p2), std::runtime_error);
#endif
  RangeN r2(p2, p2 + finish);
  BOOST_CHECK_EQUAL(r2.start(), p2);
  BOOST_CHECK_EQUAL(r2.finish(), p2 + finish);
  BOOST_CHECK_EQUAL(r2.size(), size);
  BOOST_CHECK_EQUAL(r2.volume(), volume);

  BOOST_REQUIRE_NO_THROW(RangeN r4(r)); // Copy Constructor
  RangeN r4(r);
  BOOST_CHECK_EQUAL(r4.start(), start);
  BOOST_CHECK_EQUAL(r4.finish(), finish);
  BOOST_CHECK_EQUAL(r4.size(), size);
  BOOST_CHECK_EQUAL(r4.volume(), volume);

  BOOST_REQUIRE_NO_THROW(RangeN r5(p2, p2)); // Zero Size Construction
  RangeN r5(p2, p2);
  BOOST_CHECK_EQUAL(r5.start(), p2);
  BOOST_CHECK_EQUAL(r5.finish(), p2);
  BOOST_CHECK_EQUAL(r5.size(), start.data());
  BOOST_CHECK_EQUAL(r5.volume(), 0u);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(RangeN r6(std::forward<RangeN>(RangeN(start, finish)))); // move constructor
  RangeN r6(std::forward<RangeN>(RangeN(start, finish)));
  BOOST_CHECK_EQUAL(r6.start(), start);
  BOOST_CHECK_EQUAL(r6.finish(), finish);
  BOOST_CHECK_EQUAL(r6.size(), size);
  BOOST_CHECK_EQUAL(r6.volume(), volume);
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << r;
  BOOST_CHECK( !output.is_empty( false ) ); // check for correct output.
  BOOST_CHECK( output.check_length( 24, false ) );
  BOOST_CHECK( output.is_equal("[ (0, 0, 0), (1, 2, 3) )") );
}

BOOST_AUTO_TEST_CASE( comparision )
{
  RangeN r1(r);
  RangeN r2(p0, p1);
  BOOST_CHECK(r1 == r); // check operator==
  BOOST_CHECK( ! (r2 == r) ); // check for failure
  BOOST_CHECK(r2 != r); // check operator!=
  BOOST_CHECK( ! (r1 != r) ); // check for failure
}

BOOST_AUTO_TEST_CASE( assignment )
{
  RangeN r1;
  BOOST_CHECK_EQUAL( (r1 = r), r); // check that assignment returns itself.
  BOOST_CHECK_EQUAL(r1, r);        // check that assignment is correct.

  RangeN r2 = r;
  BOOST_CHECK_EQUAL(r2, r); // check construction assignment.

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  RangeN r3;
  r3 = RangeN(start, finish);
  BOOST_CHECK_EQUAL(r3, r); // check move assignment.
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( resize )
{
  RangeN r1;

  // Check initial conditions
  BOOST_CHECK_EQUAL(r1.volume(), 0ul);

  // check resize and return value
  BOOST_CHECK_EQUAL(r1.resize(start, finish), r);
  // check that size was correctly recalculated
  BOOST_CHECK_EQUAL(r1.size(), r.size());
  // Check that weight was correctly recalculated
  BOOST_CHECK_EQUAL(r1.size(), r.weight());
  // Check that volume was correctly recalculated
  BOOST_CHECK_EQUAL(r1.volume(), r.volume());
}

BOOST_AUTO_TEST_CASE( permutation )
{
  index p1(4,6,8);
  RangeN r1(size, p1);
  Permutation<3> p(2,0,1);
  RangeN r2 = p ^ r1;
  RangeN r3 = r1;

  // check for correct start permutation
  BOOST_CHECK_EQUAL(r1.start()[0], r2.start()[2]);
  BOOST_CHECK_EQUAL(r1.start()[1], r2.start()[0]);
  BOOST_CHECK_EQUAL(r1.start()[2], r2.start()[1]);

  // check for correct finish permutation
  BOOST_CHECK_EQUAL(r1.finish()[0], r2.finish()[2]);
  BOOST_CHECK_EQUAL(r1.finish()[1], r2.finish()[0]);
  BOOST_CHECK_EQUAL(r1.finish()[2], r2.finish()[1]);
  BOOST_CHECK_EQUAL(r3 ^= p, r2);
  BOOST_CHECK_EQUAL(r3, r2);
}

BOOST_AUTO_TEST_CASE( include )
{
  RangeN r1(p1, p5);
  index t1(0,3,3);
  index t2(1,3,3);
  index t3(2,3,3);
  index t4(4,3,3);
  index t5(5,3,3);
  index t6(6,3,3);
  index t7(0,0,3);
  index t8(1,1,3);
  index t9(2,2,3);
  index t10(4,4,3);
  index t11(5,5,3);
  index t12(6,6,3);
  index t13(0,6,3);
  index t14(1,5,3);
  index t15(2,4,3);
  index t16(4,2,3);
  index t17(5,1,3);
  index t18(6,0,3);
  index t19(1,4,3);
  index t20(4,1,3);

  BOOST_CHECK(! r1.includes(t1)); // check side include
  BOOST_CHECK(r1.includes(t2));
  BOOST_CHECK(r1.includes(t3));
  BOOST_CHECK(r1.includes(t4));
  BOOST_CHECK(!r1.includes(t5));
  BOOST_CHECK(!r1.includes(t6));
  BOOST_CHECK(!r1.includes(t7)); // check diagonal include
  BOOST_CHECK(r1.includes(t8));
  BOOST_CHECK(r1.includes(t9));
  BOOST_CHECK(r1.includes(t10));
  BOOST_CHECK(!r1.includes(t11));
  BOOST_CHECK(!r1.includes(t12));
  BOOST_CHECK(!r1.includes(t13)); // check other diagonal include
  BOOST_CHECK(!r1.includes(t14));
  BOOST_CHECK(r1.includes(t15));
  BOOST_CHECK(r1.includes(t16));
  BOOST_CHECK(!r1.includes(t17));
  BOOST_CHECK(!r1.includes(t18));
  BOOST_CHECK(r1.includes(t19));  // check corners
  BOOST_CHECK(r1.includes(t20));

  RangeN::ordinal_index o = 0;
  for(; o < r.volume(); ++o)
    BOOST_CHECK(r.includes(o));

  BOOST_CHECK(! r.includes(o));
}

BOOST_AUTO_TEST_CASE( unions )
{
  RangeN r1(p1, p3);
  RangeN r2(p2, p4);

  RangeN ru1 = r1 & r2;
  RangeN ru2 = r2 & r1;

  BOOST_CHECK_EQUAL(ru1.start(), p2);  // check with r2 start inside r1
  BOOST_CHECK_EQUAL(ru1.finish(), p3);
  BOOST_CHECK_EQUAL(ru1, ru2);         // and vis versa

  RangeN r3(p1, p2);
  RangeN r4(p3, p4);
  RangeN ru3 = r3 & r4;
  BOOST_CHECK_EQUAL(ru3.start(), start);  // no over lap
  BOOST_CHECK_EQUAL(ru3.finish(), start);
  BOOST_CHECK_EQUAL(ru3.volume(), 0u);

  RangeN r5(p1, p4);
  RangeN r6(p2, p3);
  RangeN ru4 = r5 & r6;
  RangeN ru5 = r6 & r5;
  BOOST_CHECK_EQUAL(ru4, r6); // check contained block
  BOOST_CHECK_EQUAL(ru5, r6);

  index p5(2, 1, 1);
  index p6(6, 5, 2);
  index p7(1, 2, 1);
  index p8(5, 6, 2);
  index p9(2, 2, 1);
  index p10(5, 5, 2);
  RangeN r7(p5,p6);
  RangeN r8(p7,p8);
  RangeN ru6 = r7 & r8;
  RangeN ru7 = r8 & r7;
  BOOST_CHECK_EQUAL(ru6.start(), p9);  // check union when start & finish are
  BOOST_CHECK_EQUAL(ru6.finish(), p10);// not inside each other.
  BOOST_CHECK_EQUAL(ru6, ru7);

  index p11(2, 1, 1);
  index p12(5, 6, 2);
  index p13(1, 2, 1);
  index p14(6, 5, 2);
  index p15(2, 2, 1);
  index p16(5, 5, 2);
  RangeN r9(p11,p12);
  RangeN r10(p13,p14);
  RangeN ru8 = r9 & r10;
  RangeN ru9 = r10 & r9;
  BOOST_CHECK_EQUAL(ru8.start(), p15);  // check union when start & finish are
  BOOST_CHECK_EQUAL(ru8.finish(), p16);// not inside each other.
  BOOST_CHECK_EQUAL(ru8, ru9);
  BOOST_CHECK_EQUAL(ru8, ru6);
}


BOOST_AUTO_TEST_CASE( iteration )
{
  typedef CoordinateSystem<3> cs3;
  std::vector<index> t(8);

#ifdef TEST_C_STYLE_CS
  t[0] = index(1,1,1);
  t[1] = index(1,1,2);
  t[2] = index(1,2,1);
  t[3] = index(1,2,2);
  t[4] = index(2,1,1);
  t[5] = index(2,1,2);
  t[6] = index(2,2,1);
  t[7] = index(2,2,2);
#else // TEST_FORTRAN_CS
  t[0] = FRangeN::index(1,1,1);
  t[1] = FRangeN::index(2,1,1);
  t[2] = FRangeN::index(1,2,1);
  t[3] = FRangeN::index(2,2,1);
  t[4] = FRangeN::index(1,1,2);
  t[5] = FRangeN::index(2,1,2);
  t[6] = FRangeN::index(1,2,2);
  t[7] = FRangeN::index(2,2,2);
#endif

  RangeN r1(index(1,1,1),index(3,3,3));
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_SUITE_END()
