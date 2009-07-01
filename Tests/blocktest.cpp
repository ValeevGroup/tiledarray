#define BOOST_TEST_DYN_LINK

#include <block.h>
#include <permutation.h>
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include <vector>
#include "iterationtest.h"

using namespace TiledArray;

struct BlockFixture {
  typedef Block<std::size_t, 3> Block3;
  typedef Block<std::size_t, 3, LevelTag<1>,
      CoordinateSystem<3, detail::increasing_dimension_order> > FBlock3;
  typedef Block3::size_array size_array;
  typedef Block3::index_type index_type;
  typedef Block3::volume_type volume_type;

  BlockFixture() : p000(0,0,0), p111(1,1,1), p222(2,2,2), p333(3,3,3),
      p444(4,4,4), p555(5,5,5), p666(6,6,6)
  {
    size[0] = 1;
    size[1] = 2;
    size[2] = 3;
    b.resize(size);
    f = size;
    v = 6;
  }

  ~BlockFixture() { }

  Block3 b;
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


BOOST_FIXTURE_TEST_SUITE( block_suite, BlockFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL(b.start(), s);   // check start()
  BOOST_CHECK_EQUAL(b.finish(), f);  // check finish()
  BOOST_CHECK_EQUAL(b.size(), size); // check size()
  BOOST_CHECK_EQUAL(b.volume(), v);  // check volume()
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_TEST_MESSAGE("Default Constructor");
  BOOST_REQUIRE_NO_THROW(Block3 b0);
  Block3 b0;
  BOOST_CHECK_EQUAL(b0.start(), s);
  BOOST_CHECK_EQUAL(b0.finish(), s);
  BOOST_CHECK_EQUAL(b0.size(), s.data());
  BOOST_CHECK_EQUAL(b0.volume(), 0);

  BOOST_TEST_MESSAGE("Size Constructor");
  BOOST_REQUIRE_NO_THROW(Block3 b1(size));
  Block3 b1(size);
  BOOST_CHECK_EQUAL(b1.start(), s);
  BOOST_CHECK_EQUAL(b1.finish(), f);
  BOOST_CHECK_EQUAL(b1.size(), size);
  BOOST_CHECK_EQUAL(b1.volume(), v);

  BOOST_TEST_MESSAGE("Size Constructor (with offset)");
  BOOST_REQUIRE_NO_THROW(Block3 b10(size, p222));
  Block3 b10(size, p222);
  BOOST_CHECK_EQUAL(b10.start(), p222);
  BOOST_CHECK_EQUAL(b10.finish(), size + p222);
  BOOST_CHECK_EQUAL(b10.size(), size);
  BOOST_CHECK_EQUAL(b10.volume(), v);

  BOOST_TEST_MESSAGE("Start/Finish Constructor");
  BOOST_REQUIRE_NO_THROW(Block3 b2(p222, p222 + f));
  Block3 b2(p222, p222 + f);
  BOOST_CHECK_EQUAL(b2.start(), p222);
  BOOST_CHECK_EQUAL(b2.finish(), p222 + f);
  BOOST_CHECK_EQUAL(b2.size(), size);
  BOOST_CHECK_EQUAL(b2.volume(), v);

  BOOST_TEST_MESSAGE("Copy Constructor");
  BOOST_REQUIRE_NO_THROW(Block3 b4(b));
  Block3 b4(b);
  BOOST_CHECK_EQUAL(b4.start(), s);
  BOOST_CHECK_EQUAL(b4.finish(), f);
  BOOST_CHECK_EQUAL(b4.size(), size);
  BOOST_CHECK_EQUAL(b4.volume(), v);

  BOOST_TEST_MESSAGE("Zero Size Construction");
  BOOST_REQUIRE_NO_THROW(Block3 b5(p222, p222));
  Block3 b5(p222, p222);
  BOOST_CHECK_EQUAL(b5.start(), p222);
  BOOST_CHECK_EQUAL(b5.finish(), p222);
  BOOST_CHECK_EQUAL(b5.size(), s.data());
  BOOST_CHECK_EQUAL(b5.volume(), 0);
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << b;
  BOOST_CHECK( !output.is_empty( false ) ); // check for correct output.
  BOOST_CHECK( output.check_length( 24, false ) );
  BOOST_CHECK( output.is_equal("[ (0, 0, 0), (1, 2, 3) )") );
}

BOOST_AUTO_TEST_CASE( comparision )
{
  Block3 b1(b);
  Block3 b2(p000, p111);
  BOOST_CHECK(b1 == b); // check operator==
  BOOST_CHECK( ! (b2 == b) );
  BOOST_CHECK(b2 != b); // check operator!=
  BOOST_CHECK( ! (b1 != b) );
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Block3 b1;
  BOOST_CHECK_EQUAL( (b1 = b), b); // check that assignment returns itself.
  BOOST_CHECK_EQUAL(b1, b);        // check that assignment is correct.

  Block3 b2 = b;
  BOOST_CHECK_EQUAL(b2, b); // check construction assignment.
}

BOOST_AUTO_TEST_CASE( resize )
{
  Block3 b1;
  BOOST_CHECK_EQUAL(b1.resize(size), b); // check resize with size_array
  Block3 b2;
  BOOST_CHECK_EQUAL(b2.resize(s, f), b); // check resize with start and finish
  BOOST_CHECK_EQUAL(b2.size(), b.size());// check that size was correctly recalculated
}

BOOST_AUTO_TEST_CASE( permutation )
{
  index_type p1(4,6,8);
  Block3 b1(size, p1);
  Permutation<3> p(2,0,1);
  Block3 b2 = p ^ b1;
  Block3 b3 = b1;
  BOOST_CHECK_EQUAL(b1.start()[0], b2.start()[2]);  // check for correct start
  BOOST_CHECK_EQUAL(b1.start()[1], b2.start()[0]);  // permutation
  BOOST_CHECK_EQUAL(b1.start()[2], b2.start()[1]);
  BOOST_CHECK_EQUAL(b1.finish()[0], b2.finish()[2]);// check for correct finish
  BOOST_CHECK_EQUAL(b1.finish()[1], b2.finish()[0]);// permutation
  BOOST_CHECK_EQUAL(b1.finish()[2], b2.finish()[1]);
  BOOST_CHECK_EQUAL(b3 ^= p, b2);
  BOOST_CHECK_EQUAL(b3, b2);
}

BOOST_AUTO_TEST_CASE( include )
{
  Block3 b1(p111, p555);
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

  BOOST_CHECK(! b1.includes(t1)); // check side include
  BOOST_CHECK(b1.includes(t2));
  BOOST_CHECK(b1.includes(t3));
  BOOST_CHECK(b1.includes(t4));
  BOOST_CHECK(!b1.includes(t5));
  BOOST_CHECK(!b1.includes(t6));
  BOOST_CHECK(!b1.includes(t7)); // check diagonal include
  BOOST_CHECK(b1.includes(t8));
  BOOST_CHECK(b1.includes(t9));
  BOOST_CHECK(b1.includes(t10));
  BOOST_CHECK(!b1.includes(t11));
  BOOST_CHECK(!b1.includes(t12));
  BOOST_CHECK(!b1.includes(t13)); // check other diagonal include
  BOOST_CHECK(!b1.includes(t14));
  BOOST_CHECK(b1.includes(t15));
  BOOST_CHECK(b1.includes(t16));
  BOOST_CHECK(!b1.includes(t17));
  BOOST_CHECK(!b1.includes(t18));
  BOOST_CHECK(b1.includes(t19));  // check corners
  BOOST_CHECK(b1.includes(t20));
}

BOOST_AUTO_TEST_CASE( unions )
{
  Block3 b1(p111, p333);
  Block3 b2(p222, p444);

  Block3 bu1 = b1 & b2;
  Block3 bu2 = b2 & b1;

  BOOST_CHECK_EQUAL(bu1.start(), p222);  // check with b2 start inside b1
  BOOST_CHECK_EQUAL(bu1.finish(), p333);
  BOOST_CHECK_EQUAL(bu1, bu2);         // and vis versa

  Block3 b3(p111, p222);
  Block3 b4(p333, p444);
  Block3 bu3 = b3 & b4;
  BOOST_CHECK_EQUAL(bu3.start(), s);  // no over lap
  BOOST_CHECK_EQUAL(bu3.finish(), s);
  BOOST_CHECK_EQUAL(bu3.volume(), 0);

  Block3 b5(p111, p444);
  Block3 b6(p222, p333);
  Block3 bu4 = b5 & b6;
  Block3 bu5 = b6 & b5;
  BOOST_CHECK_EQUAL(bu4, b6); // check contained block
  BOOST_CHECK_EQUAL(bu5, b6);

  index_type p5(2, 1, 1);
  index_type p6(6, 5, 2);
  index_type p7(1, 2, 1);
  index_type p8(5, 6, 2);
  index_type p9(2, 2, 1);
  index_type p10(5, 5, 2);
  Block3 b7(p5,p6);
  Block3 b8(p7,p8);
  Block3 bu6 = b7 & b8;
  Block3 bu7 = b8 & b7;
  BOOST_CHECK_EQUAL(bu6.start(), p9);  // check union when start & finish are
  BOOST_CHECK_EQUAL(bu6.finish(), p10);// not inside each other.
  BOOST_CHECK_EQUAL(bu6, bu7);

  index_type p11(2, 1, 1);
  index_type p12(5, 6, 2);
  index_type p13(1, 2, 1);
  index_type p14(6, 5, 2);
  index_type p15(2, 2, 1);
  index_type p16(5, 5, 2);
  Block3 b9(p11,p12);
  Block3 b10(p13,p14);
  Block3 bu8 = b9 & b10;
  Block3 bu9 = b10 & b9;
  BOOST_CHECK_EQUAL(bu8.start(), p15);  // check union when start & finish are
  BOOST_CHECK_EQUAL(bu8.finish(), p16);// not inside each other.
  BOOST_CHECK_EQUAL(bu8, bu9);
  BOOST_CHECK_EQUAL(bu8, bu6);
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

  Block3 b1(p111,p333);
  BOOST_CHECK_EQUAL(const_iteration_test(b1, t.begin(), t.end()), 8);
                                              // check basic iteration operation
}

BOOST_AUTO_TEST_CASE( fortran_iteration )
{
  FBlock3::index_type p1(1,1,1);
  FBlock3::index_type p3(3,3,3);
  std::vector<FBlock3::index_type> t(8);
  t[0] = FBlock3::index_type(1,1,1);
  t[1] = FBlock3::index_type(2,1,1);
  t[2] = FBlock3::index_type(1,2,1);
  t[3] = FBlock3::index_type(2,2,1);
  t[4] = FBlock3::index_type(1,1,2);
  t[5] = FBlock3::index_type(2,1,2);
  t[6] = FBlock3::index_type(1,2,2);
  t[7] = FBlock3::index_type(2,2,2);

  FBlock3 b1(p1,p3);
  BOOST_CHECK_EQUAL(const_iteration_test(b1, t.begin(), t.end()), 8);
                              // check basic fortran ordered iteration operation
}
BOOST_AUTO_TEST_SUITE_END()
