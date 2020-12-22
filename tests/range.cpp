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

#include <TiledArray/util/eigen.h>
#include <boost/range/combine.hpp>
#ifdef TILEDARRAY_HAS_RANGEV3
#include <range/v3/view/zip.hpp>
#endif

#include <numeric>
#include <sstream>
#include "TiledArray/range.h"
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

template <typename SizeArray>
inline std::size_t calc_volume(const SizeArray& sz) {
  using std::size;
  const std::size_t n = size(sz);
  std::size_t volume = 0ul;
  if (n) {
    volume = 1ul;
    for (std::size_t i = 0ul; i < n; ++i) volume *= sz[i];
  }

  return volume;
}

using namespace TiledArray;

const RangeFixture::index RangeFixture::start(GlobalFixture::dim, 0);
const RangeFixture::index RangeFixture::finish(GlobalFixture::dim, 5);
const std::vector<std::size_t> RangeFixture::size(GlobalFixture::dim, 5);
const std::vector<std::size_t> RangeFixture::weight = RangeFixture::calc_weight(
    std::vector<std::size_t>(GlobalFixture::dim, 5), GlobalFixture::dim);
const RangeFixture::size_type RangeFixture::volume =
    calc_volume(std::vector<std::size_t>(GlobalFixture::dim, 5));
const RangeFixture::index RangeFixture::p0(GlobalFixture::dim, 0);
const RangeFixture::index RangeFixture::p1(GlobalFixture::dim, 1);
const RangeFixture::index RangeFixture::p2(GlobalFixture::dim, 2);
const RangeFixture::index RangeFixture::p3(GlobalFixture::dim, 3);
const RangeFixture::index RangeFixture::p4(GlobalFixture::dim, 4);
const RangeFixture::index RangeFixture::p5(GlobalFixture::dim, 5);
const RangeFixture::index RangeFixture::p6(GlobalFixture::dim, 6);

BOOST_FIXTURE_TEST_SUITE(range_suite, RangeFixture, TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(dimension_accessor) {
  BOOST_CHECK_EQUAL_COLLECTIONS(r.lobound_data(), r.lobound_data() + r.rank(),
                                start.begin(), start.end());  // check start()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.upbound_data(), r.upbound_data() + r.rank(),
                                finish.begin(),
                                finish.end());  // check finish()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.extent_data(), r.extent_data() + r.rank(),
                                size.begin(), size.end());  // check size()
  BOOST_CHECK_EQUAL_COLLECTIONS(r.stride_data(), r.stride_data() + r.rank(),
                                weight.begin(),
                                weight.end());  // check weight()
  BOOST_CHECK_EQUAL(r.volume(), volume);        // check volume()
  for (size_t d = 0; d != r.rank(); ++d) {
    auto range_d = r.dim(d);
    BOOST_CHECK_EQUAL(range_d.first, start[d]);
    BOOST_CHECK_EQUAL(range_d.second, finish[d]);
  }
}

BOOST_AUTO_TEST_CASE(constructors) {
  // Default Constructor
  BOOST_REQUIRE_NO_THROW(Range r0);
  Range r0;
  BOOST_CHECK(!r0);
  BOOST_CHECK_EQUAL(r0.rank(), 0u);
  BOOST_CHECK(r0.upbound_data() == r0.lobound_data());
  BOOST_CHECK(r0.extent_data() == r0.lobound_data());
  BOOST_CHECK(r0.stride_data() == r0.lobound_data());
  BOOST_CHECK_EQUAL(r0.volume(), 0ul);

  // Copy of a default-constructed object
  BOOST_CHECK_NO_THROW(Range r00(r0));
  Range r00(r0);
  BOOST_CHECK(!r00);
  BOOST_CHECK_EQUAL(r00.rank(), 0u);
  BOOST_CHECK(r00.upbound_data() == r00.lobound_data());
  BOOST_CHECK(r00.extent_data() == r00.lobound_data());
  BOOST_CHECK(r00.stride_data() == r00.lobound_data());
  BOOST_CHECK_EQUAL(r00.volume(), 0ul);

  // Rank-0 Range *IS* null Range
  BOOST_REQUIRE_NO_THROW(Range r1(std::vector<int>{}));
  Range r1(std::vector<int>{});
  BOOST_CHECK(!r1);
  BOOST_CHECK_EQUAL(r1.rank(), 0u);
  BOOST_CHECK(r1.upbound_data() == r1.lobound_data());
  BOOST_CHECK(r1.extent_data() == r1.lobound_data());
  BOOST_CHECK(r1.stride_data() == r1.lobound_data());
  BOOST_CHECK_EQUAL(r1.volume(), 0ul);

  // another way to make Rank-0 Range
  BOOST_CHECK_NO_THROW(Range r11(std::vector<int>{}, std::vector<int>{}));
  Range r11(std::vector<int>{}, std::vector<int>{});
  BOOST_CHECK(!r11);
  BOOST_CHECK_EQUAL(r11.rank(), 0u);
  BOOST_CHECK(r11.upbound_data() == r11.lobound_data());
  BOOST_CHECK(r11.extent_data() == r11.lobound_data());
  BOOST_CHECK(r11.stride_data() == r11.lobound_data());
  BOOST_CHECK_EQUAL(r11.volume(), 0ul);

  index f2 = finish;
  for (std::size_t i = 0; i < f2.size(); ++i) f2[i] += p2[i];

  // Constructor that takes extents
  BOOST_REQUIRE_NO_THROW(Range r5(p5));              // uses index container
  BOOST_REQUIRE_NO_THROW(Range r5({0, 2, 3}));       // uses initializer_list
  BOOST_REQUIRE_NO_THROW(Range r5({1, 2, 3}));       // uses initializer_list
  BOOST_REQUIRE_NO_THROW(Range r5(1u, 2, 3ul, 4l));  // uses param pack
  BOOST_REQUIRE_NO_THROW(Range r5({0, 0, 0}));       // zero extents are OK
  Range r5(p5);
  BOOST_CHECK(r5);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      r5.lobound_data(), r5.lobound_data() + r5.rank(), p0.begin(), p0.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      r5.upbound_data(), r5.upbound_data() + r5.rank(), p5.begin(), p5.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.extent_data(), r5.extent_data() + r5.rank(),
                                p5.begin(), p5.end());
  BOOST_CHECK_EQUAL(r5.volume(), volume);

  ///////////////////////////////////////////////
  // Constructors that takes start/finish indices
  ///////////////////////////////////////////////
  BOOST_REQUIRE_NO_THROW(Range r2(p2, f2));  // uses index containers
  BOOST_REQUIRE_NO_THROW(
      Range r(boost::combine(p2, f2)));  // uses zipped range of p2 and f2
#ifdef TILEDARRAY_HAS_RANGEV3
  BOOST_REQUIRE_NO_THROW(
      Range r(ranges::views::zip(p2, f2)));  // uses zipped range of p2 and f2
#endif

  BOOST_CHECK_THROW(Range r2(f2, p2), Exception);  // lobound > upbound
  Range r2(p2, f2);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      r2.lobound_data(), r2.lobound_data() + r2.rank(), p2.begin(), p2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      r2.upbound_data(), r2.upbound_data() + r2.rank(), f2.begin(), f2.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.extent_data(), r2.extent_data() + r2.rank(),
                                size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.stride_data(), r2.stride_data() + r2.rank(),
                                weight.begin(), weight.end());
  BOOST_CHECK_EQUAL(r2.volume(), volume);

  // check that zipped bounds ctors work correctly
  Range should_be_copy_of_r2(
      boost::combine(p2, f2));  // uses zipped range of p2 and f2
  BOOST_CHECK_EQUAL(r2, should_be_copy_of_r2);
#ifdef TILEDARRAY_HAS_RANGEV3
  Range should_be_another_copy_of_r2(
      ranges::views::zip(p2, f2));  // uses zipped range of p2 and f2
  BOOST_CHECK_EQUAL(r2, should_be_another_copy_of_r2);
#endif

  // test the rest of bound-based ctors
  {
    Range ref({0, 1, 2}, {4, 6, 8});
    // BOOST_CHECK_THROW(Range ref{{0, 1, 2}, {4, 6, 8}}, Exception);  // mind
    // the parens!

    // uses initializer_list of pairs
    BOOST_REQUIRE_NO_THROW(
        Range r1({std::make_pair(0, 4), std::pair(1, 6), std::pair{2, 8}}));
    Range r1({std::make_pair(0, 4), std::pair(1, 6), std::pair{2, 8}});
    BOOST_CHECK_EQUAL(ref, r1);

    // uses initializer_list of tuples
    BOOST_REQUIRE_NO_THROW(
        Range r1a({std::make_tuple(0, 4), std::tuple(1, 6), std::tuple{2, 8}}));
    Range r1a({std::make_tuple(0, 4), std::tuple(1, 6), std::tuple{2, 8}});
    BOOST_CHECK_EQUAL(ref, r1a);

    std::vector<std::pair<size_t, size_t>> vpbounds{{0, 4}, {1, 6}, {2, 8}};
    std::vector<std::tuple<size_t, size_t>> vtbounds{{0, 4}, {1, 6}, {2, 8}};

    // uses vector of pairs
    BOOST_REQUIRE_NO_THROW(Range r2(vpbounds));
    Range r2(vpbounds);
    BOOST_CHECK_EQUAL(ref, r2);

    // uses vector of tuples
    BOOST_REQUIRE_NO_THROW(Range r3(vtbounds));
    Range r3(vtbounds);
    BOOST_CHECK_EQUAL(ref, r3);

    // uses param pack of pairs
    BOOST_REQUIRE_NO_THROW(
        Range r4(std::make_pair(0, 4), std::pair(1, 6), std::pair{2, 8}));
    Range r4(std::make_pair(0, 4), std::pair(1, 6), std::pair{2, 8});
    BOOST_CHECK_EQUAL(ref, r4);

    // uses initializer_list of 2-element initializer_list's
    BOOST_REQUIRE_NO_THROW(Range r5({{0, 4}, {1, 6}, {2, 8}}));
    Range r5({{0, 4}, {1, 6}, {2, 8}});
    BOOST_CHECK_EQUAL(ref, r5);
    Range r6{{0, 4}, {1, 6}, {2, 8}};  // same but without extra parens
    BOOST_CHECK_EQUAL(ref, r6);

    // uses zipped bounds
    Range r7(boost::combine(std::vector{0, 1, 2}, std::array{4, 6, 8}));
    BOOST_CHECK_EQUAL(ref, r7);
#ifdef TILEDARRAY_HAS_RANGEV3
//    Range r8(ranges::views::zip(std::array{0, 1, 2}, std::vector{4, 6, 8}));
//    BOOST_CHECK_EQUAL(ref, r8);
#endif

    // zipped bounds with Eigen vectors
    {
      Range r9(
          boost::combine(Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(4, 6, 8)));
      BOOST_CHECK_EQUAL(ref, r9);

      // iv = shorthand for Eigen::Vector of ints
      using TiledArray::eigen::iv;
      Range r10(iv({0, 1, 2}),  // from initializer_list -> VectorXi
                iv(4, 6, 8));  // from param pack -> VectorNi (N=3 in this case)
      BOOST_CHECK_EQUAL(ref, r10);

      // can compose indices easier with Eigen vectors, but need to force
      // evaluation (note iv around the sum)
      Range r11(iv({0, 1, 2}), iv(iv(0, 1, 2) + iv(4, 5, 6)));
      BOOST_CHECK_EQUAL(ref, r11);

      // can zip Eigen vectors
      Range r12(boost::combine(iv({0, 1, 2}), iv(4, 6, 8)));
      BOOST_CHECK_EQUAL(ref, r12);

      // can make Eigen vectors out of other ranges
      Range r13(
          boost::combine(iv(std::vector{0, 1, 2}), iv(std::array{4, 6, 8})));
      BOOST_CHECK_EQUAL(ref, r13);

      // etc
      Range r14(boost::combine(iv({0, 1, 2}), iv(iv({0, 1, 2}) + iv(4, 5, 6))));
      BOOST_CHECK_EQUAL(ref, r14);

#ifdef TILEDARRAY_HAS_RANGEV3
// this requires Eigen ~3.4 (3.3.90 docs suggest it should be sufficient)
//    Range r15(ranges::views::zip(iv(0, 1, 2), iv(4, 6, 8)));
//    BOOST_CHECK_EQUAL(ref, r15);
#endif
    }

    // container::svector as bounds
    {
      // iv = shorthand for container::svector of ints
      using TiledArray::container::iv;
      Range r10(iv({0, 1, 2}),  // from initializer_list -> svector<int>
                iv(4, 6, 8));   // from param pack -> svector<int,3>
      BOOST_CHECK_EQUAL(ref, r10);
    }
  }

  // make sure zero extents are OK also with start/finish indices
  {
    BOOST_REQUIRE_NO_THROW(
        Range r2(std::make_pair(2, 2), std::make_pair(4, 5),
                 std::make_pair(5, 6)));  // uses param pack of pairs
    Range r2(std::make_pair(2, 2), std::make_pair(4, 5), std::make_pair(5, 6));
    auto lobound_ref = {2, 4, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.lobound_data(),
                                  r2.lobound_data() + r2.rank(),
                                  lobound_ref.begin(), lobound_ref.end());
    auto upbound_ref = {2, 5, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.upbound_data(),
                                  r2.upbound_data() + r2.rank(),
                                  upbound_ref.begin(), upbound_ref.end());
    auto extent_ref = {0, 1, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.extent_data(),
                                  r2.extent_data() + r2.rank(),
                                  extent_ref.begin(), extent_ref.end());
    auto stride_ref = {1, 1, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.stride_data(),
                                  r2.stride_data() + r2.rank(),
                                  stride_ref.begin(), stride_ref.end());
    BOOST_CHECK_EQUAL(r2.volume(), 0);
  }

  // make sure negative bounds are OK, if TA_SIGNED_1INDEX_TYPE is define
#ifdef TA_SIGNED_1INDEX_TYPE
  {
    BOOST_REQUIRE_NO_THROW(Range r2({{-1, 1}, {-2, 2}, {0, 6}}));
    Range r2{{-1, 1}, {-2, 2}, {0, 6}};
    auto lobound_ref = {-1, -2, 0};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.lobound_data(),
                                  r2.lobound_data() + r2.rank(),
                                  lobound_ref.begin(), lobound_ref.end());
    auto upbound_ref = {1, 2, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.upbound_data(),
                                  r2.upbound_data() + r2.rank(),
                                  upbound_ref.begin(), upbound_ref.end());
    auto extent_ref = {2, 4, 6};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.extent_data(),
                                  r2.extent_data() + r2.rank(),
                                  extent_ref.begin(), extent_ref.end());
    auto stride_ref = {24, 6, 1};
    BOOST_CHECK_EQUAL_COLLECTIONS(r2.stride_data(),
                                  r2.stride_data() + r2.rank(),
                                  stride_ref.begin(), stride_ref.end());
    BOOST_CHECK_EQUAL(r2.volume(), 48);
  }
#else  // TA_SIGNED_1INDEX_TYPE
  BOOST_REQUIRE_THROW(Range r2({{-1, 1}, {-2, 2}, {0, 6}}),
                      TiledArray::Exception);
#endif  // TA_SIGNED_1INDEX_TYPE

  // Copy Constructor
  BOOST_REQUIRE_NO_THROW(Range r4(r));
  Range r4(r);
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.lobound_data(),
                                r4.lobound_data() + r4.rank(), start.begin(),
                                start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.upbound_data(),
                                r4.upbound_data() + r4.rank(), finish.begin(),
                                finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.extent_data(), r4.extent_data() + r4.rank(),
                                size.begin(), size.end());
  BOOST_CHECK_EQUAL(r4.volume(), volume);
}

BOOST_AUTO_TEST_CASE(move_constructor) {
  Range r_copy(r);
  Range x(std::move(r_copy));

  // Check that the data in x matches that of r.
  BOOST_CHECK_EQUAL_COLLECTIONS(x.lobound_data(), x.lobound_data() + x.rank(),
                                r.lobound_data(), r.lobound_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.upbound_data(), x.upbound_data() + x.rank(),
                                r.upbound_data(), r.upbound_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.extent_data(), x.extent_data() + x.rank(),
                                r.extent_data(), r.extent_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.stride_data(), x.stride_data() + x.rank(),
                                r.stride_data(), r.stride_data() + r.rank());
  BOOST_CHECK_EQUAL(x.volume(), r.volume());

  // moved-from object is null
  BOOST_CHECK(!r_copy);
}

BOOST_AUTO_TEST_CASE(assignment_operator) {
  Range x;
  x = r;

  // Check that the data in x matches that of r.
  BOOST_CHECK_EQUAL_COLLECTIONS(x.lobound_data(), x.lobound_data() + x.rank(),
                                r.lobound_data(), r.lobound_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.upbound_data(), x.upbound_data() + x.rank(),
                                r.upbound_data(), r.upbound_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.extent_data(), x.extent_data() + x.rank(),
                                r.extent_data(), r.extent_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.stride_data(), x.stride_data() + x.rank(),
                                r.stride_data(), r.stride_data() + r.rank());
  BOOST_CHECK_EQUAL(x.volume(), r.volume());
}

BOOST_AUTO_TEST_CASE(move_assignment_operator) {
  Range r_copy(r);
  Range x;
  x = std::move(r_copy);

  // Check that the data in x matches that of r.
  BOOST_CHECK_EQUAL_COLLECTIONS(x.lobound_data(), x.lobound_data() + x.rank(),
                                r.lobound_data(), r.lobound_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.upbound_data(), x.upbound_data() + x.rank(),
                                r.upbound_data(), r.upbound_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.extent_data(), x.extent_data() + x.rank(),
                                r.extent_data(), r.extent_data() + r.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.stride_data(), x.stride_data() + x.rank(),
                                r.stride_data(), r.stride_data() + r.rank());
  BOOST_CHECK_EQUAL(x.volume(), r.volume());

  // moved-from object is null
  BOOST_CHECK(!r_copy);
}

BOOST_AUTO_TEST_CASE(ostream) {
  std::stringstream stm;
  stm << "[ " << start << ", " << finish << " )";

  boost::test_tools::output_test_stream output;
  output << r;
  BOOST_CHECK(!output.is_empty(false));  // check for correct output.
  BOOST_CHECK(output.check_length(stm.str().size(), false));
  BOOST_CHECK(output.is_equal(stm.str().c_str()));
}

BOOST_AUTO_TEST_CASE(comparision) {
  Range r1(r);
  Range r2(p0, p1);
  BOOST_CHECK(r1 == r);     // check operator==
  BOOST_CHECK(!(r2 == r));  // check for failure
  BOOST_CHECK(r2 != r);     // check operator!=
  BOOST_CHECK(!(r1 != r));  // check for failure
}

BOOST_AUTO_TEST_CASE(congruency) {
  Range r1(r);
  r1.inplace_shift(index(GlobalFixture::dim, 1));
  BOOST_CHECK(is_congruent(r1, r));
}

BOOST_AUTO_TEST_CASE(assignment) {
  Range r1;
  BOOST_CHECK_EQUAL((r1 = r), r);  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(r1, r);        // check that assignment is correct.

  Range r2 = r;
  BOOST_CHECK_EQUAL(r2, r);  // check construction assignment.
}

BOOST_AUTO_TEST_CASE(resize) {
  Range r1;

  // Check initial conditions
  BOOST_CHECK_EQUAL(r1.volume(), 0ul);

  // check resize and return value
  BOOST_CHECK_EQUAL(r1.resize(start, finish), r);
  // check that size was correctly recalculated
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.extent_data(), r1.extent_data() + r1.rank(),
                                r.extent_data(), r.extent_data() + r.rank());
  // Check that weight was correctly recalculated
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.stride_data(), r1.stride_data() + r1.rank(),
                                r.stride_data(), r.stride_data() + r.rank());
  // Check that volume was correctly recalculated
  BOOST_CHECK_EQUAL(r1.volume(), r.volume());
}

BOOST_AUTO_TEST_CASE(permutation) {
  index s(GlobalFixture::dim);
  index f(GlobalFixture::dim);
  std::vector<unsigned int> a(GlobalFixture::dim, 0);
  for (unsigned int d = 0; d < GlobalFixture::dim; ++d) {
    s[d] = d;
    f[d] = d + d + 5;
    a[GlobalFixture::dim - d - 1] = d;
  }
  Range r1(s, f);
  // create a reverse order permutation
  Permutation p(a);
  Range r2 = p * r1;
  Range r3 = r1;

  // check start, finish, size, volume, and weight of permuted range
  typedef std::reverse_iterator<const Range::index1_type*> riter_type;
  BOOST_CHECK_EQUAL_COLLECTIONS(
      riter_type(r1.lobound_data() + r1.rank()), riter_type(r1.lobound_data()),
      r2.lobound_data(), r2.lobound_data() + r2.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      riter_type(r1.upbound_data() + r1.rank()), riter_type(r1.upbound_data()),
      r2.upbound_data(), r2.upbound_data() + r2.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(riter_type(r1.extent_data() + r1.rank()),
                                riter_type(r1.extent_data()), r2.extent_data(),
                                r2.extent_data() + r2.rank());
  BOOST_CHECK_EQUAL(r2.volume(), r1.volume());

  std::vector<std::size_t> w =
      RangeFixture::calc_weight(r2.extent_data(), r2.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.stride_data(), r2.stride_data() + r2.rank(),
                                w.begin(), w.end());

  // check for correct finish permutation
  BOOST_CHECK_EQUAL(r3 *= p, r2);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      r3.lobound_data(), r3.lobound_data() + r3.rank(), r2.lobound_data(),
      r2.lobound_data() + r2.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      r3.upbound_data(), r3.upbound_data() + r3.rank(), r2.upbound_data(),
      r2.upbound_data() + r2.rank());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.extent_data(), r3.extent_data() + r3.rank(),
                                r2.extent_data(), r2.extent_data() + r2.rank());
  BOOST_CHECK_EQUAL(r3.volume(), r2.volume());
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.stride_data(), r3.stride_data() + r3.rank(),
                                r2.stride_data(), r2.stride_data() + r2.rank());
  BOOST_CHECK_EQUAL(r3, r2);
}

BOOST_AUTO_TEST_CASE(include) {
  index s(3, 1);
  index f(3, 5);
  Range r1(s, f);
  index t1(GlobalFixture::dim);
  t1[0] = 0;
  t1[1] = 3;
  t1[2] = 3;
  index t2(GlobalFixture::dim);
  t2[0] = 1;
  t2[1] = 3;
  t2[2] = 3;
  index t3(GlobalFixture::dim);
  t3[0] = 2;
  t3[1] = 3;
  t3[2] = 3;
  index t4(GlobalFixture::dim);
  t4[0] = 4;
  t4[1] = 3;
  t4[2] = 3;
  index t5(GlobalFixture::dim);
  t5[0] = 5;
  t5[1] = 3;
  t5[2] = 3;
  index t6(GlobalFixture::dim);
  t6[0] = 6;
  t6[1] = 3;
  t6[2] = 3;
  index t7(GlobalFixture::dim);
  t7[0] = 0;
  t7[1] = 0;
  t7[2] = 3;
  index t8(GlobalFixture::dim);
  t8[0] = 1;
  t8[1] = 1;
  t8[2] = 3;
  index t9(GlobalFixture::dim);
  t9[0] = 2;
  t9[1] = 2;
  t9[2] = 3;
  index t10(GlobalFixture::dim);
  t10[0] = 4;
  t10[1] = 4;
  t10[2] = 3;
  index t11(GlobalFixture::dim);
  t11[0] = 5;
  t11[1] = 5;
  t11[2] = 3;
  index t12(GlobalFixture::dim);
  t12[0] = 6;
  t12[1] = 6;
  t12[2] = 3;
  index t13(GlobalFixture::dim);
  t13[0] = 0;
  t13[1] = 6;
  t13[2] = 3;
  index t14(GlobalFixture::dim);
  t14[0] = 1;
  t14[1] = 5;
  t14[2] = 3;
  index t15(GlobalFixture::dim);
  t15[0] = 2;
  t15[1] = 4;
  t15[2] = 3;
  index t16(GlobalFixture::dim);
  t16[0] = 4;
  t16[1] = 2;
  t16[2] = 3;
  index t17(GlobalFixture::dim);
  t17[0] = 5;
  t17[1] = 1;
  t17[2] = 3;
  index t18(GlobalFixture::dim);
  t18[0] = 6;
  t18[1] = 0;
  t18[2] = 3;
  index t19(GlobalFixture::dim);
  t19[0] = 1;
  t19[1] = 4;
  t19[2] = 3;
  index t20(GlobalFixture::dim);
  t20[0] = 4;
  t20[1] = 1;
  t20[2] = 3;

  BOOST_CHECK(!r1.includes(t1));  // check side include
  BOOST_CHECK(r1.includes(t2));
  BOOST_CHECK(r1.includes(t3));
  BOOST_CHECK(r1.includes(t4));
  BOOST_CHECK(!r1.includes(t5));
  BOOST_CHECK(!r1.includes(t6));
  BOOST_CHECK(!r1.includes(t7));  // check diagonal include
  BOOST_CHECK(r1.includes(t8));
  BOOST_CHECK(r1.includes(t9));
  BOOST_CHECK(r1.includes(t10));
  BOOST_CHECK(!r1.includes(t11));
  BOOST_CHECK(!r1.includes(t12));
  BOOST_CHECK(!r1.includes(t13));  // check other diagonal include
  BOOST_CHECK(!r1.includes(t14));
  BOOST_CHECK(r1.includes(t15));
  BOOST_CHECK(r1.includes(t16));
  BOOST_CHECK(!r1.includes(t17));
  BOOST_CHECK(!r1.includes(t18));
  BOOST_CHECK(r1.includes(t19));  // check corners
  BOOST_CHECK(r1.includes(t20));

  Range::size_type o = 0;
  for (; o < r.volume(); ++o) {
    BOOST_CHECK(r.includes(o));
  }

  BOOST_CHECK(!r.includes(o));

  // ensure that includes always fails if any extent is zero
  {
    Range r({0, 1, 2});
    BOOST_CHECK(!r.includes({0, 0, 0}));
  }
}

BOOST_AUTO_TEST_CASE(iteration) {
  std::vector<Range::index> tc(9, Range::index(3, 0));

  tc[0][0] = 1;
  tc[0][1] = 1;
  tc[0][2] = 1;
  tc[1][0] = 1;
  tc[1][1] = 1;
  tc[1][2] = 2;
  tc[2][0] = 1;
  tc[2][1] = 2;
  tc[2][2] = 1;
  tc[3][0] = 1;
  tc[3][1] = 2;
  tc[3][2] = 2;
  tc[4][0] = 2;
  tc[4][1] = 1;
  tc[4][2] = 1;
  tc[5][0] = 2;
  tc[5][1] = 1;
  tc[5][2] = 2;
  tc[6][0] = 2;
  tc[6][1] = 2;
  tc[6][2] = 1;
  tc[7][0] = 2;
  tc[7][1] = 2;
  tc[7][2] = 2;
  tc[8][0] = 3;
  tc[8][1] = 3;
  tc[8][2] = 3;

  Range rc(tc[0], tc[8]);
  BOOST_CHECK_EQUAL_COLLECTIONS(rc.begin(), rc.end(), tc.begin(), tc.end() - 1);
}

BOOST_AUTO_TEST_CASE(serialization) {
  std::size_t buf_size =
      2 * (sizeof(Range) + sizeof(std::size_t) * (4 * GlobalFixture::dim + 1));
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar& r;
  std::size_t nbyte = oar.size();
  oar.close();

  Range rs;
  madness::archive::BufferInputArchive iar(buf, nbyte);
  iar& rs;
  iar.close();

  delete[] buf;

  BOOST_CHECK_EQUAL_COLLECTIONS(rs.lobound_data(),
                                rs.lobound_data() + rs.rank(), r.lobound_data(),
                                r.lobound_data() + r.rank());  // check start()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.upbound_data(),
                                rs.upbound_data() + rs.rank(), r.upbound_data(),
                                r.upbound_data() + r.rank());  // check finish()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.extent_data(), rs.extent_data() + rs.rank(),
                                r.extent_data(),
                                r.extent_data() + r.rank());  // check size()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.stride_data(), rs.stride_data() + rs.rank(),
                                r.stride_data(),
                                r.stride_data() + r.rank());  // check weight()
  BOOST_CHECK_EQUAL(rs.volume(), r.volume());                 // check volume()
  BOOST_CHECK_EQUAL_COLLECTIONS(rs.begin(), rs.end(), r.begin(), r.end());
}

BOOST_AUTO_TEST_CASE(swap) {
  Range empty_range;

  // Check range swap
  BOOST_CHECK_NO_THROW(r.swap(empty_range));

  // Check that r contains the data of empty_range.
  BOOST_CHECK_EQUAL(r.rank(), 0u);
  BOOST_CHECK(r.upbound_data() == r.lobound_data());
  BOOST_CHECK(r.extent_data() == r.lobound_data());
  BOOST_CHECK(r.stride_data() == r.lobound_data());
  BOOST_CHECK_EQUAL(r.volume(), 0ul);

  // Check that empty_range contains the data of r.
  BOOST_CHECK_EQUAL_COLLECTIONS(empty_range.lobound_data(),
                                empty_range.lobound_data() + empty_range.rank(),
                                start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(empty_range.upbound_data(),
                                empty_range.upbound_data() + empty_range.rank(),
                                finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(empty_range.extent_data(),
                                empty_range.extent_data() + empty_range.rank(),
                                size.begin(), size.end());
  BOOST_CHECK_EQUAL(empty_range.volume(), volume);

  // Swap the data back
  BOOST_CHECK_NO_THROW(r.swap(empty_range));

  // Check that empty_range contains its original data.
  BOOST_CHECK_EQUAL(empty_range.rank(), 0u);
  BOOST_CHECK(empty_range.upbound_data() == empty_range.lobound_data());
  BOOST_CHECK(empty_range.extent_data() == empty_range.lobound_data());
  BOOST_CHECK(empty_range.stride_data() == empty_range.lobound_data());
  BOOST_CHECK_EQUAL(empty_range.volume(), 0ul);

  // Check that r its original data.
  BOOST_CHECK_EQUAL_COLLECTIONS(r.lobound_data(), r.lobound_data() + r.rank(),
                                start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r.upbound_data(), r.upbound_data() + r.rank(),
                                finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r.extent_data(), r.extent_data() + r.rank(),
                                size.begin(), size.end());
  BOOST_CHECK_EQUAL(r.volume(), volume);
}

BOOST_AUTO_TEST_SUITE_END()
