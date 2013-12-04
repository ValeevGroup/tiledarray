/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 */

#include "TiledArray/range.h"
#include "TiledArray/permutation.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include <sstream>
#include <world/bufar.h>

template <typename SizeArray>
inline std::size_t calc_volume(const SizeArray& size) {
  const std::size_t n = detail::size(size);
  std::size_t volume = 0ul;
  if(n) {
    volume = 1ul;
    for(std::size_t i = 0ul; i < n; ++i)
      volume *= size[i];
  }

  return volume;
}

using namespace TiledArray;

const RangeFixture::index RangeFixture::start(GlobalFixture::dim, 0);
const RangeFixture::index RangeFixture::finish(GlobalFixture::dim, 5);
const std::vector<std::size_t> RangeFixture::size(GlobalFixture::dim,  5);
const std::vector<std::size_t> RangeFixture::weight =
    RangeFixture::calc_weight(std::vector<std::size_t>(GlobalFixture::dim, 5));
const RangeFixture::size_type RangeFixture::volume =
    calc_volume(std::vector<std::size_t>(GlobalFixture::dim, 5));
const RangeFixture::index RangeFixture::p0(GlobalFixture::dim, 0);
const RangeFixture::index RangeFixture::p1(GlobalFixture::dim, 1);
const RangeFixture::index RangeFixture::p2(GlobalFixture::dim, 2);
const RangeFixture::index RangeFixture::p3(GlobalFixture::dim, 3);
const RangeFixture::index RangeFixture::p4(GlobalFixture::dim, 4);
const RangeFixture::index RangeFixture::p5(GlobalFixture::dim, 5);
const RangeFixture::index RangeFixture::p6(GlobalFixture::dim, 6);



BOOST_FIXTURE_TEST_SUITE( range_suite, RangeFixture )

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
  BOOST_REQUIRE_NO_THROW(Range r0); // Default Constructor
  Range r0;
  BOOST_CHECK_EQUAL(r0.dim(), 0u);
  BOOST_CHECK_EQUAL(r0.start().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.finish().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.size().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.weight().size(), 0ul);
  BOOST_CHECK_EQUAL(r0.volume(), 0ul);

  BOOST_CHECK_NO_THROW(Range r00(r0));
  Range r00(r0);
  BOOST_CHECK_EQUAL(r00.dim(), 0u);
  BOOST_CHECK_EQUAL(r00.start().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.finish().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.size().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.weight().size(), 0ul);
  BOOST_CHECK_EQUAL(r00.volume(), 0ul);

  index f2 = finish;
  for(std::size_t i = 0; i < f2.size(); ++i)
    f2[i] += p2[i];

  BOOST_REQUIRE_NO_THROW(Range r2(p2, f2)); // Start/Finish Constructor
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(Range r2(f2, p2), Exception);
#endif // TA_EXCEPTION_ERROR
  Range r2(p2, f2);
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.start().begin(), r2.start().end(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.finish().begin(), r2.finish().end(), f2.begin(), f2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.size().begin(), r2.size().end(), size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.weight().begin(), r2.weight().end(), weight.begin(), weight.end());
  BOOST_CHECK_EQUAL(r2.volume(), volume);

  BOOST_REQUIRE_NO_THROW(Range r4(r)); // Copy Constructor
  Range r4(r);
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.start().begin(), r4.start().end(), start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.finish().begin(), r4.finish().end(), finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.size().begin(), r4.size().end(), size.begin(), size.end());
  BOOST_CHECK_EQUAL(r4.volume(), volume);

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(Range(p2, p2), Exception); // Zero Size Construction
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( assignment_operator )
{
  Range x;
  x = r;

  // Check that the data in x matches that of r.
  BOOST_CHECK_EQUAL_COLLECTIONS(x.start().begin(), x.start().end(), r.start().begin(), r.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.finish().begin(), x.finish().end(), r.finish().begin(), r.finish().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), r.size().begin(), r.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.weight().begin(), x.weight().end(), r.weight().begin(), r.weight().end());
  BOOST_CHECK_EQUAL(x.volume(), r.volume());
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
  Range r1(r);
  Range r2(p0, p1);
  BOOST_CHECK(r1 == r); // check operator==
  BOOST_CHECK( ! (r2 == r) ); // check for failure
  BOOST_CHECK(r2 != r); // check operator!=
  BOOST_CHECK( ! (r1 != r) ); // check for failure
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Range r1;
  BOOST_CHECK_EQUAL( (r1 = r), r); // check that assignment returns itself.
  BOOST_CHECK_EQUAL(r1, r);        // check that assignment is correct.

  Range r2 = r;
  BOOST_CHECK_EQUAL(r2, r); // check construction assignment.
}

BOOST_AUTO_TEST_CASE( resize )
{
  Range r1;

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
  index s(GlobalFixture::dim);
  index f(GlobalFixture::dim);
  std::vector<std::size_t> a(GlobalFixture::dim, 0);
  for(unsigned int d = 0; d < GlobalFixture::dim; ++d) {
    s[d] = d;
    f[d] = d + d + 5;
    a[GlobalFixture::dim - d - 1] = d;
  }
  Range r1(s, f);
  // create a reverse order permutation
  Permutation p(a);
  Range r2 = p ^ r1;
  Range r3 = r1;

  // check start, finish, size, volume, and weight of permuted range
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.start().rbegin(), r1.start().rend(),
                                r2.start().begin(),  r2.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.finish().rbegin(), r1.finish().rend(),
                                r2.finish().begin(),  r2.finish().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.size().rbegin(), r1.size().rend(),
                                r2.size().begin(),  r2.size().end());
  BOOST_CHECK_EQUAL(r2.volume(), r1.volume());

  std::vector<std::size_t> w =
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
  index s(3,1);
  index f(3,5);
  Range r1(s, f);
  index t1(GlobalFixture::dim); t1[0] = 0; t1[1] = 3; t1[2] = 3;
  index t2(GlobalFixture::dim); t2[0] = 1; t2[1] = 3; t2[2] = 3;
  index t3(GlobalFixture::dim); t3[0] = 2; t3[1] = 3; t3[2] = 3;
  index t4(GlobalFixture::dim); t4[0] = 4; t4[1] = 3; t4[2] = 3;
  index t5(GlobalFixture::dim); t5[0] = 5; t5[1] = 3; t5[2] = 3;
  index t6(GlobalFixture::dim); t6[0] = 6; t6[1] = 3; t6[2] = 3;
  index t7(GlobalFixture::dim); t7[0] = 0; t7[1] = 0; t7[2] = 3;
  index t8(GlobalFixture::dim); t8[0] = 1; t8[1] = 1; t8[2] = 3;
  index t9(GlobalFixture::dim); t9[0] = 2; t9[1] = 2; t9[2] = 3;
  index t10(GlobalFixture::dim); t10[0] = 4; t10[1] = 4; t10[2] = 3;
  index t11(GlobalFixture::dim); t11[0] = 5; t11[1] = 5; t11[2] = 3;
  index t12(GlobalFixture::dim); t12[0] = 6; t12[1] = 6; t12[2] = 3;
  index t13(GlobalFixture::dim); t13[0] = 0; t13[1] = 6; t13[2] = 3;
  index t14(GlobalFixture::dim); t14[0] = 1; t14[1] = 5; t14[2] = 3;
  index t15(GlobalFixture::dim); t15[0] = 2; t15[1] = 4; t15[2] = 3;
  index t16(GlobalFixture::dim); t16[0] = 4; t16[1] = 2; t16[2] = 3;
  index t17(GlobalFixture::dim); t17[0] = 5; t17[1] = 1; t17[2] = 3;
  index t18(GlobalFixture::dim); t18[0] = 6; t18[1] = 0; t18[2] = 3;
  index t19(GlobalFixture::dim); t19[0] = 1; t19[1] = 4; t19[2] = 3;
  index t20(GlobalFixture::dim); t20[0] = 4; t20[1] = 1; t20[2] = 3;

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

  Range::size_type o = 0;
  for(; o < r.volume(); ++o) {
    BOOST_CHECK(r.includes(o));
  }

  BOOST_CHECK(! r.includes(o));
}

BOOST_AUTO_TEST_CASE( iteration )
{
  std::vector<Range::index> tc(9, Range::index(3, 0));

  tc[0][0] = 1; tc[0][1] = 1; tc[0][2] = 1;
  tc[1][0] = 1; tc[1][1] = 1; tc[1][2] = 2;
  tc[2][0] = 1; tc[2][1] = 2; tc[2][2] = 1;
  tc[3][0] = 1; tc[3][1] = 2; tc[3][2] = 2;
  tc[4][0] = 2; tc[4][1] = 1; tc[4][2] = 1;
  tc[5][0] = 2; tc[5][1] = 1; tc[5][2] = 2;
  tc[6][0] = 2; tc[6][1] = 2; tc[6][2] = 1;
  tc[7][0] = 2; tc[7][1] = 2; tc[7][2] = 2;
  tc[8][0] = 3; tc[8][1] = 3; tc[8][2] = 3;


  Range rc(tc[0],tc[8]);
  BOOST_CHECK_EQUAL_COLLECTIONS(rc.begin(), rc.end(), tc.begin(), tc.end() - 1);
}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = 2 * (sizeof(Range) + sizeof(std::size_t) * (4 * GlobalFixture::dim + 1));
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar & r;
  std::size_t nbyte = oar.size();
  oar.close();

  Range rs;
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
