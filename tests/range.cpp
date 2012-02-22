#include "TiledArray/range.h"
#include "TiledArray/permutation.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include <sstream>
#include <world/bufar.h>

using namespace TiledArray;

StaticRangeFixture::StaticRangeFixture() : r(RangeFixture::start, RangeFixture::finish) { }


DynamicRangeFixture::DynamicRangeFixture() :
    r(RangeFixture::start, RangeFixture::finish)
{ }

const RangeFixture::index RangeFixture::start(0);
const RangeFixture::index RangeFixture::finish(5);
const RangeFixture::size_array RangeFixture::size     = RangeFixture::finish.data();
const RangeFixture::size_array RangeFixture::weight   = RangeFixture::calc_weight(RangeFixture::size);
const RangeFixture::size_type RangeFixture::volume  = detail::calc_volume(RangeFixture::size);
const RangeFixture::index RangeFixture::p0(0);
const RangeFixture::index RangeFixture::p1(1);
const RangeFixture::index RangeFixture::p2(2);
const RangeFixture::index RangeFixture::p3(3);
const RangeFixture::index RangeFixture::p4(4);
const RangeFixture::index RangeFixture::p5(5);
const RangeFixture::index RangeFixture::p6(6);


BOOST_FIXTURE_TEST_SUITE( static_range_suite, StaticRangeFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(r.start().begin(), r.start().end(), start.begin(), start.end());   // check start()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.finish().begin(), r.finish().end(), finish.begin(), finish.end());  // check finish()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.size().begin(), r.size().end(), size.begin(), size.end()); // check size()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.weight().begin(), r.weight().end(), weight.begin(), weight.end()); // check weight()
  BOOST_CHECK_EQUAL(r.volume(), volume);  // check volume()
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(RangeN r0); // Default Constructor
  RangeN r0;
  BOOST_CHECK_EQUAL_COLLECTIONS(r0.start().begin(), r0.start().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r0.finish().begin(), r0.finish().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r0.size().begin(), r0.size().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r0.weight().begin(), r0.weight().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL(r0.volume(), 0u);

  BOOST_REQUIRE_NO_THROW(RangeN r2(p2, p2 + finish)); // Start/Finish Constructor
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(RangeN r2(p2 + finish, p2), Exception);
#endif // TA_EXCEPTION_ERROR
  index f2 = p2 + finish;
  RangeN r2(p2, f2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.start().begin(), r2.start().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.finish().begin(), r2.finish().end(), f2.begin(), f2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.size().begin(), r2.size().end(), size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.weight().begin(), r2.weight().end(), weight.begin(), weight.end());
  BOOST_CHECK_EQUAL(r2.volume(), volume);

  BOOST_REQUIRE_NO_THROW(RangeN r4(r)); // Copy Constructor
  RangeN r4(r);
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.start().begin(), r4.start().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.finish().begin(), r4.finish().end(), finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.size().begin(), r4.size().end(), size.begin(), size.end());
  BOOST_CHECK_EQUAL(r4.volume(), volume);

  BOOST_REQUIRE_NO_THROW(RangeN r5(p2, p2)); // Zero Size Construction
  RangeN r5(p2, p2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.start().begin(), r5.start().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.finish().begin(), r5.finish().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.size().begin(), r5.size().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.weight().begin(), r5.weight().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL(r5.volume(), 0u);
}

BOOST_AUTO_TEST_CASE( ostream )
{
  std::stringstream stm;
  stm << "[ " << start << ", " << finish << " )";

  boost::test_tools::output_test_stream output;
  output << r;
  BOOST_CHECK( !output.is_empty( false ) ); // check for correct output.
  BOOST_CHECK( output.check_length( stm.str().size(), false ) );
  BOOST_CHECK( output.is_equal(stm.str().c_str()) );
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
}

BOOST_AUTO_TEST_CASE( resize )
{
  RangeN r1;

  // Check initial conditions
  BOOST_CHECK_EQUAL(r1.volume(), 0ul);

  // check resize and return value
  BOOST_CHECK_EQUAL(r1.resize(start, finish), r);
  // check that size was correctly recalculated
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.size().begin(), r1.size().end(), r.size().begin(), r.size().end());
  // Check that weight was correctly recalculated
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.weight().begin(), r1.weight().end(), r.weight().begin(), r.weight().end());
  // Check that volume was correctly recalculated
  BOOST_CHECK_EQUAL(r1.volume(), r.volume());
}

BOOST_AUTO_TEST_CASE( permutation )
{
  index s;
  index f;
  size_array a;
  for(unsigned int d = 0; d < GlobalFixture::coordinate_system::dim; ++d) {
    s[d] = d;
    f[d] = d + d + 5;
    a[GlobalFixture::coordinate_system::dim - d - 1] = d;
  }
  RangeN r1(s, f);
  // create a reverse order permutation
  Permutation p(a);
  RangeN r2 = p ^ r1;
  RangeN r3 = r1;

  // check start, finish, size, volume, and weight of permuted range
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.start().rbegin(), r1.start().rend(),
                                r2.start().begin(),  r2.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.finish().rbegin(), r1.finish().rend(),
                                r2.finish().begin(),  r2.finish().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.size().rbegin(), r1.size().rend(),
                                r2.size().begin(),  r2.size().end());
  BOOST_CHECK_EQUAL(r2.volume(), r1.volume());

  GlobalFixture::coordinate_system::size_array w =
      RangeFixture::calc_weight(r2.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.weight().begin(), r2.weight().end(), w.begin(), w.end());

  // check for correct finish permutation
  BOOST_CHECK_EQUAL(r3 ^= p, r2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.start().begin(), r3.start().end(), r2.start().begin(), r2.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.finish().begin(), r3.finish().end(), r2.finish().begin(), r2.finish().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.size().begin(), r3.size().end(), r2.size().begin(), r2.size().end());
  BOOST_CHECK_EQUAL(r3.volume(), r2.volume());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.weight().begin(), r3.weight().end(), r2.weight().begin(), r2.weight().end());
  BOOST_CHECK_EQUAL(r3, r2);
}

BOOST_AUTO_TEST_CASE( include )
{
  typedef StaticRange<CoordinateSystem<3> > Range3;
  typedef Range3::index index;

  index s(1,1,1);
  index f(5,5,5);
  Range3 r1(s, f);
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

  Range3::size_type o = 0;
  for(; o < r.volume(); ++o) {
    BOOST_CHECK(r.includes(o));
  }

  BOOST_CHECK(! r.includes(o));
}

BOOST_AUTO_TEST_CASE( iteration )
{
  typedef StaticRange<CoordinateSystem<3> > Range3C;
  std::vector<Range3C::index> tc(8);

  tc[0] = Range3C::index(1,1,1);
  tc[1] = Range3C::index(1,1,2);
  tc[2] = Range3C::index(1,2,1);
  tc[3] = Range3C::index(1,2,2);
  tc[4] = Range3C::index(2,1,1);
  tc[5] = Range3C::index(2,1,2);
  tc[6] = Range3C::index(2,2,1);
  tc[7] = Range3C::index(2,2,2);

  Range3C rc(Range3C::index(1,1,1),Range3C::index(3,3,3));
  BOOST_CHECK_EQUAL_COLLECTIONS(rc.begin(), rc.end(), tc.begin(), tc.end());
}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = sizeof(RangeN) * 2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar & r;
  std::size_t nbyte = oar.size();
  oar.close();

  RangeN rs;
  madness::archive::BufferInputArchive iar(buf,nbyte);
  iar & rs;
  iar.close();

  delete [] buf;

  BOOST_CHECK_EQUAL_COLLECTIONS(rs.start().begin(), rs.start().end(), r.start().begin(), r.start().end());   // check start()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.finish().begin(), rs.finish().end(), r.finish().begin(), r.finish().end()); // check finish()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.size().begin(), rs.size().end(), r.size().begin(), r.size().end());     // check size()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.weight().begin(), rs.weight().end(), r.weight().begin(), r.weight().end()); // check weight()
  BOOST_CHECK_EQUAL(rs.volume(), r.volume()); // check volume()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.begin(), rs.end(), r.begin(), r.end());
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( dynamic_range_suite, DynamicRangeFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(r.start().begin(), r.start().end(), start.begin(), start.end());   // check start()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.finish().begin(), r.finish().end(), finish.begin(), finish.end());  // check finish()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.size().begin(), r.size().end(), size.begin(), size.end()); // check size()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.weight().begin(), r.weight().end(), weight.begin(), weight.end()); // check weight()
  BOOST_CHECK_EQUAL(r.volume(), volume);  // check volume()
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(DynamicRange r0); // Default Constructor
  DynamicRange r0;
  BOOST_CHECK_EQUAL(r0.dim(), 0u);
  BOOST_CHECK_EQUAL(r0.start().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.finish().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.size().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.weight().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.volume(), 0ul);

  BOOST_CHECK_NO_THROW(DynamicRange r00(r0));
  DynamicRange r00(r0);
  BOOST_CHECK_EQUAL(r00.dim(), 0u);
  BOOST_CHECK_EQUAL(r00.start().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.finish().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.size().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.weight().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.volume(), 0ul);

  BOOST_REQUIRE_NO_THROW(DynamicRange r2(p2, p2 + finish)); // Start/Finish Constructor
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(DynamicRange r2(p2 + finish, p2), Exception);
#endif // TA_EXCEPTION_ERROR
  index f2 = p2 + finish;
  DynamicRange r2(p2, f2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.start().begin(), r2.start().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.finish().begin(), r2.finish().end(), f2.begin(), f2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.size().begin(), r2.size().end(), size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.weight().begin(), r2.weight().end(), weight.begin(), weight.end());
  BOOST_CHECK_EQUAL(r2.volume(), volume);

  BOOST_REQUIRE_NO_THROW(DynamicRange r4(r)); // Copy Constructor
  DynamicRange r4(r);
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.start().begin(), r4.start().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.finish().begin(), r4.finish().end(), finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.size().begin(), r4.size().end(), size.begin(), size.end());
  BOOST_CHECK_EQUAL(r4.volume(), volume);

  BOOST_REQUIRE_NO_THROW(DynamicRange r5(p2, p2)); // Zero Size Construction
  DynamicRange r5(p2, p2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.start().begin(), r5.start().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.finish().begin(), r5.finish().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.size().begin(), r5.size().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.weight().begin(), r5.weight().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL(r5.volume(), 0u);
}

BOOST_AUTO_TEST_CASE( ostream )
{
  std::stringstream stm;
  stm << "[ " << start << ", " << finish << " )";

  boost::test_tools::output_test_stream output;
  output << r;
  BOOST_CHECK( !output.is_empty( false ) ); // check for correct output.
  BOOST_CHECK( output.check_length( stm.str().size(), false ) );
  BOOST_CHECK( output.is_equal(stm.str().c_str()) );
}

BOOST_AUTO_TEST_CASE( comparision )
{
  DynamicRange r1(r);
  DynamicRange r2(p0, p1);
  BOOST_CHECK(r1 == r); // check operator==
  BOOST_CHECK( ! (r2 == r) ); // check for failure
  BOOST_CHECK(r2 != r); // check operator!=
  BOOST_CHECK( ! (r1 != r) ); // check for failure
}

BOOST_AUTO_TEST_CASE( assignment )
{
  DynamicRange r1;
  BOOST_CHECK_EQUAL( (r1 = r), r); // check that assignment returns itself.
  BOOST_CHECK_EQUAL(r1, r);        // check that assignment is correct.

  DynamicRange r2 = r;
  BOOST_CHECK_EQUAL(r2, r); // check construction assignment.
}

BOOST_AUTO_TEST_CASE( resize )
{
  DynamicRange r1;

  // Check initial conditions
  BOOST_CHECK_EQUAL(r1.volume(), 0ul);

  // check resize and return value
  BOOST_CHECK_EQUAL(r1.resize(start, finish), r);
  // check that size was correctly recalculated
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.size().begin(), r1.size().end(), r.size().begin(), r.size().end());
  // Check that weight was correctly recalculated
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.weight().begin(), r1.weight().end(), r.weight().begin(), r.weight().end());
  // Check that volume was correctly recalculated
  BOOST_CHECK_EQUAL(r1.volume(), r.volume());
}

BOOST_AUTO_TEST_CASE( permutation )
{
  index s;
  index f;
  size_array a(GlobalFixture::coordinate_system::dim, 0);
  for(unsigned int d = 0; d < GlobalFixture::coordinate_system::dim; ++d) {
    s[d] = d;
    f[d] = d + d + 5;
    a[GlobalFixture::coordinate_system::dim - d - 1] = d;
  }
  DynamicRange r1(s, f);
  // create a reverse order permutation
  Permutation p(a);
  DynamicRange r2 = p ^ r1;
  DynamicRange r3 = r1;

  // check start, finish, size, volume, and weight of permuted range
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.start().rbegin(), r1.start().rend(),
                                r2.start().begin(),  r2.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.finish().rbegin(), r1.finish().rend(),
                                r2.finish().begin(),  r2.finish().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.size().rbegin(), r1.size().rend(),
                                r2.size().begin(),  r2.size().end());
  BOOST_CHECK_EQUAL(r2.volume(), r1.volume());

  DynamicRange::size_array w =
      RangeFixture::calc_weight(r2.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.weight().begin(), r2.weight().end(), w.begin(), w.end());

  // check for correct finish permutation
  BOOST_CHECK_EQUAL(r3 ^= p, r2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.start().begin(), r3.start().end(), r2.start().begin(), r2.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.finish().begin(), r3.finish().end(), r2.finish().begin(), r2.finish().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.size().begin(), r3.size().end(), r2.size().begin(), r2.size().end());
  BOOST_CHECK_EQUAL(r3.volume(), r2.volume());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.weight().begin(), r3.weight().end(), r2.weight().begin(), r2.weight().end());
  BOOST_CHECK_EQUAL(r3, r2);
}

BOOST_AUTO_TEST_CASE( include )
{
  typedef StaticRange<CoordinateSystem<3> > Range3;
  typedef Range3::index index;

  index s(1,1,1);
  index f(5,5,5);
  Range3 r1(s, f);
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

  Range3::size_type o = 0;
  for(; o < r.volume(); ++o) {
    BOOST_CHECK(r.includes(o));
  }

  BOOST_CHECK(! r.includes(o));
}

BOOST_AUTO_TEST_CASE( iteration )
{
  std::vector<DynamicRange::index> tc(9, DynamicRange::index(3, 0));

  tc[0][0] = 1; tc[0][1] = 1; tc[0][2] = 1;
  tc[1][0] = 1; tc[1][1] = 1; tc[1][2] = 2;
  tc[2][0] = 1; tc[2][1] = 2; tc[2][2] = 1;
  tc[3][0] = 1; tc[3][1] = 2; tc[3][2] = 2;
  tc[4][0] = 2; tc[4][1] = 1; tc[4][2] = 1;
  tc[5][0] = 2; tc[5][1] = 1; tc[5][2] = 2;
  tc[6][0] = 2; tc[6][1] = 2; tc[6][2] = 1;
  tc[7][0] = 2; tc[7][1] = 2; tc[7][2] = 2;
  tc[8][0] = 3; tc[8][1] = 3; tc[8][2] = 3;


  DynamicRange rc(tc[0],tc[8]);
  BOOST_CHECK_EQUAL_COLLECTIONS(rc.begin(), rc.end(), tc.begin(), tc.end() - 1);
}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = sizeof(DynamicRange) * 2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar & r;
  std::size_t nbyte = oar.size();
  oar.close();

  DynamicRange rs;
  madness::archive::BufferInputArchive iar(buf,nbyte);
  iar & rs;
  iar.close();

  delete [] buf;

  BOOST_CHECK_EQUAL_COLLECTIONS(rs.start().begin(), rs.start().end(), r.start().begin(), r.start().end());   // check start()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.finish().begin(), rs.finish().end(), r.finish().begin(), r.finish().end()); // check finish()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.size().begin(), rs.size().end(), r.size().begin(), r.size().end());     // check size()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.weight().begin(), rs.weight().end(), r.weight().begin(), r.weight().end()); // check weight()
  BOOST_CHECK_EQUAL(rs.volume(), r.volume()); // check volume()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.begin(), rs.end(), r.begin(), r.end());
}

BOOST_AUTO_TEST_SUITE_END()
