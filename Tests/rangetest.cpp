#include "range.h"
#include "permutation.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include "iterationtest.h"

using namespace TiledArray;

struct RangeFixture {
  typedef Range<std::size_t, 3> Range3;
  typedef Range<std::size_t, 3, LevelTag<1>,
      CoordinateSystem<3, detail::increasing_dimension_order> > FRange3;
  typedef Range3::size_array size_array;
  typedef Range3::index_type index_type;
  typedef Range3::volume_type volume_type;

  RangeFixture() : p000(0,0,0), p111(1,1,1), p222(2,2,2), p333(3,3,3),
      p444(4,4,4), p555(5,5,5), p666(6,6,6)
  {
    size[0] = 1;
    size[1] = 2;
    size[2] = 3;
    r.resize(size);
    f = size;
    v = 6;
  }

  ~RangeFixture() { }

  Range3 r;
  size_array size;
  index_type s;
  index_type f;
  volume_type v;
  const index_type p000;
  const index_type p111;
  const index_type p222;
  const index_type p333;
  const index_type p444;
  const index_type p555;
  const index_type p666;
};


BOOST_FIXTURE_TEST_SUITE( block_suite, RangeFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL(r.start(), s);   // check start()
  BOOST_CHECK_EQUAL(r.finish(), f);  // check finish()
  BOOST_CHECK_EQUAL(r.size(), size); // check size()
  BOOST_CHECK_EQUAL(r.volume(), v);  // check volume()
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(Range3 r0); // Default Constructor
  Range3 r0;
  BOOST_CHECK_EQUAL(r0.start(), s);
  BOOST_CHECK_EQUAL(r0.finish(), s);
  BOOST_CHECK_EQUAL(r0.size(), s.data());
  BOOST_CHECK_EQUAL(r0.volume(), 0);

  BOOST_REQUIRE_NO_THROW(Range3 b1(size)); // Size Constructor
  Range3 r1(size);
  BOOST_CHECK_EQUAL(r1.start(), s);
  BOOST_CHECK_EQUAL(r1.finish(), f);
  BOOST_CHECK_EQUAL(r1.size(), size);
  BOOST_CHECK_EQUAL(r1.volume(), v);

  BOOST_REQUIRE_NO_THROW(Range3 r10(size, p222)); // Size Constructor (with offset)
  Range3 r10(size, p222);
  BOOST_CHECK_EQUAL(r10.start(), p222);
  BOOST_CHECK_EQUAL(r10.finish(), p222 + size);
  BOOST_CHECK_EQUAL(r10.size(), size);
  BOOST_CHECK_EQUAL(r10.volume(), v);

  BOOST_REQUIRE_NO_THROW(Range3 r2(p222, p222 + f)); // Start/Finish Constructor
  Range3 r2(p222, p222 + f);
  BOOST_CHECK_EQUAL(r2.start(), p222);
  BOOST_CHECK_EQUAL(r2.finish(), p222 + f);
  BOOST_CHECK_EQUAL(r2.size(), size);
  BOOST_CHECK_EQUAL(r2.volume(), v);

  BOOST_REQUIRE_NO_THROW(Range3 r4(r)); // Copy Constructor
  Range3 r4(r);
  BOOST_CHECK_EQUAL(r4.start(), s);
  BOOST_CHECK_EQUAL(r4.finish(), f);
  BOOST_CHECK_EQUAL(r4.size(), size);
  BOOST_CHECK_EQUAL(r4.volume(), v);

  BOOST_REQUIRE_NO_THROW(Range3 r5(p222, p222)); // Zero Size Construction
  Range3 r5(p222, p222);
  BOOST_CHECK_EQUAL(r5.start(), p222);
  BOOST_CHECK_EQUAL(r5.finish(), p222);
  BOOST_CHECK_EQUAL(r5.size(), s.data());
  BOOST_CHECK_EQUAL(r5.volume(), 0);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(Range3 r6(std::forward<Range3>(Range3(size)))); // move constructor
  Range3 r6(std::forward<Range3>(Range3(size)));
  BOOST_CHECK_EQUAL(r6.start(), s);
  BOOST_CHECK_EQUAL(r6.finish(), f);
  BOOST_CHECK_EQUAL(r6.size(), size);
  BOOST_CHECK_EQUAL(r6.volume(), v);
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
  Range3 r1(r);
  Range3 r2(p000, p111);
  BOOST_CHECK(r1 == r); // check operator==
  BOOST_CHECK( ! (r2 == r) ); // check for failure
  BOOST_CHECK(r2 != r); // check operator!=
  BOOST_CHECK( ! (r1 != r) ); // check for failure
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Range3 r1;
  BOOST_CHECK_EQUAL( (r1 = r), r); // check that assignment returns itself.
  BOOST_CHECK_EQUAL(r1, r);        // check that assignment is correct.

  Range3 r2 = r;
  BOOST_CHECK_EQUAL(r2, r); // check construction assignment.

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  Range3 r3;
  r3 = Range3(size);
  BOOST_CHECK_EQUAL(r3, r); // check move assignment.
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( resize )
{
  Range3 r1;
  BOOST_CHECK_EQUAL(r1.resize(size), r); // check resize with size_array
  Range3 r2;
  BOOST_CHECK_EQUAL(r2.resize(s, f), r); // check resize with start and finish
  BOOST_CHECK_EQUAL(r2.size(), r.size());// check that size was correctly recalculated
}

BOOST_AUTO_TEST_CASE( permutation )
{
  index_type p1(4,6,8);
  Range3 r1(size, p1);
  Permutation<3> p(2,0,1);
  Range3 r2 = p ^ r1;
  Range3 r3 = r1;
  BOOST_CHECK_EQUAL(r1.start()[0], r2.start()[2]);  // check for correct start
  BOOST_CHECK_EQUAL(r1.start()[1], r2.start()[0]);  // permutation
  BOOST_CHECK_EQUAL(r1.start()[2], r2.start()[1]);
  BOOST_CHECK_EQUAL(r1.finish()[0], r2.finish()[2]);// check for correct finish
  BOOST_CHECK_EQUAL(r1.finish()[1], r2.finish()[0]);// permutation
  BOOST_CHECK_EQUAL(r1.finish()[2], r2.finish()[1]);
  BOOST_CHECK_EQUAL(r3 ^= p, r2);
  BOOST_CHECK_EQUAL(r3, r2);
}

BOOST_AUTO_TEST_CASE( include )
{
  Range3 r1(p111, p555);
  index_type t1(0,3,3);
  index_type t2(1,3,3);
  index_type t3(2,3,3);
  index_type t4(4,3,3);
  index_type t5(5,3,3);
  index_type t6(6,3,3);
  index_type t7(0,0,3);
  index_type t8(1,1,3);
  index_type t9(2,2,3);
  index_type t10(4,4,3);
  index_type t11(5,5,3);
  index_type t12(6,6,3);
  index_type t13(0,6,3);
  index_type t14(1,5,3);
  index_type t15(2,4,3);
  index_type t16(4,2,3);
  index_type t17(5,1,3);
  index_type t18(6,0,3);
  index_type t19(1,4,3);
  index_type t20(4,1,3);

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
}

BOOST_AUTO_TEST_CASE( unions )
{
  Range3 r1(p111, p333);
  Range3 r2(p222, p444);

  Range3 ru1 = r1 & r2;
  Range3 ru2 = r2 & r1;

  BOOST_CHECK_EQUAL(ru1.start(), p222);  // check with r2 start inside r1
  BOOST_CHECK_EQUAL(ru1.finish(), p333);
  BOOST_CHECK_EQUAL(ru1, ru2);         // and vis versa

  Range3 r3(p111, p222);
  Range3 r4(p333, p444);
  Range3 ru3 = r3 & r4;
  BOOST_CHECK_EQUAL(ru3.start(), s);  // no over lap
  BOOST_CHECK_EQUAL(ru3.finish(), s);
  BOOST_CHECK_EQUAL(ru3.volume(), 0);

  Range3 r5(p111, p444);
  Range3 r6(p222, p333);
  Range3 ru4 = r5 & r6;
  Range3 ru5 = r6 & r5;
  BOOST_CHECK_EQUAL(ru4, r6); // check contained block
  BOOST_CHECK_EQUAL(ru5, r6);

  index_type p5(2, 1, 1);
  index_type p6(6, 5, 2);
  index_type p7(1, 2, 1);
  index_type p8(5, 6, 2);
  index_type p9(2, 2, 1);
  index_type p10(5, 5, 2);
  Range3 r7(p5,p6);
  Range3 r8(p7,p8);
  Range3 ru6 = r7 & r8;
  Range3 ru7 = r8 & r7;
  BOOST_CHECK_EQUAL(ru6.start(), p9);  // check union when start & finish are
  BOOST_CHECK_EQUAL(ru6.finish(), p10);// not inside each other.
  BOOST_CHECK_EQUAL(ru6, ru7);

  index_type p11(2, 1, 1);
  index_type p12(5, 6, 2);
  index_type p13(1, 2, 1);
  index_type p14(6, 5, 2);
  index_type p15(2, 2, 1);
  index_type p16(5, 5, 2);
  Range3 r9(p11,p12);
  Range3 r10(p13,p14);
  Range3 ru8 = r9 & r10;
  Range3 ru9 = r10 & r9;
  BOOST_CHECK_EQUAL(ru8.start(), p15);  // check union when start & finish are
  BOOST_CHECK_EQUAL(ru8.finish(), p16);// not inside each other.
  BOOST_CHECK_EQUAL(ru8, ru9);
  BOOST_CHECK_EQUAL(ru8, ru6);
}

BOOST_AUTO_TEST_CASE( c_iteration )
{
  std::vector<index_type> t(8);
  t[0] = index_type(1,1,1);
  t[1] = index_type(1,1,2);
  t[2] = index_type(1,2,1);
  t[3] = index_type(1,2,2);
  t[4] = index_type(2,1,1);
  t[5] = index_type(2,1,2);
  t[6] = index_type(2,2,1);
  t[7] = index_type(2,2,2);

  Range3 r1(p111,p333);
  BOOST_CHECK_EQUAL(const_iteration_test(r1, t.begin(), t.end()), 8);
                                              // check basic iteration operation
}

BOOST_AUTO_TEST_CASE( fortran_iteration )
{
  FRange3::index_type p1(1,1,1);
  FRange3::index_type p3(3,3,3);
  std::vector<FRange3::index_type> t(8);
  t[0] = FRange3::index_type(1,1,1);
  t[1] = FRange3::index_type(2,1,1);
  t[2] = FRange3::index_type(1,2,1);
  t[3] = FRange3::index_type(2,2,1);
  t[4] = FRange3::index_type(1,1,2);
  t[5] = FRange3::index_type(2,1,2);
  t[6] = FRange3::index_type(1,2,2);
  t[7] = FRange3::index_type(2,2,2);

  FRange3 r1(p1,p3);
  BOOST_CHECK_EQUAL(const_iteration_test(r1, t.begin(), t.end()), 8);
                              // check basic fortran ordered iteration operation
}
BOOST_AUTO_TEST_SUITE_END()
