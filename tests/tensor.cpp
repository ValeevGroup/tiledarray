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

#include "tensor_fixture.h"
#include "unit_test_config.h"
#include <iterator>


const TensorFixture::range_type TensorFixture::r = make_range(81);


BOOST_FIXTURE_TEST_SUITE( tensor_suite , TensorFixture )

BOOST_AUTO_TEST_CASE( default_constructor ) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x);
  TensorN x;

  BOOST_CHECK(x.empty());

  // Check that range data is correct
  BOOST_CHECK_EQUAL(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), 0ul);
  BOOST_CHECK_EQUAL(x.range().volume(), 0ul);

  // Check the element data
  BOOST_CHECK_EQUAL(x.begin(), x.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end());

  // check for element access error
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x[0], Exception);
#endif // TA_EXCEPTION_ERROR
}


BOOST_AUTO_TEST_CASE( range_constructor )
{
  BOOST_REQUIRE_NO_THROW(TensorN x(r));
  TensorN x(r);

  BOOST_CHECK(! x.empty());

  // Check that range data is correct
  BOOST_CHECK_NE(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  // Do not check values of x because it maybe uninitialized
}

BOOST_AUTO_TEST_CASE( value_constructor )
{
  BOOST_REQUIRE_NO_THROW(TensorN x(r, 8));
  TensorN x(r, 8);

  BOOST_CHECK(! x.empty());

  // Check that range data is correct
  BOOST_CHECK_NE(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(TensorN::const_iterator it = x.begin(); it != x.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 8);
}

BOOST_AUTO_TEST_CASE( iterator_copy_constructor )
{
  BOOST_REQUIRE_NO_THROW(TensorN x(r, t.begin()));
  TensorN x(r, t.begin());

  BOOST_CHECK(! x.empty());

  // Check range data is correct
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], t[i]);
}

BOOST_AUTO_TEST_CASE( copy_constructor ) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN tc(t));
  TensorN tc(t);

  BOOST_CHECK_EQUAL(tc.empty(), t.empty());

  // Check that range data is correct
  BOOST_CHECK_EQUAL(tc.data(), t.data());
  BOOST_CHECK_EQUAL(tc.size(), t.size());
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK_EQUAL(tc.begin(), t.begin());
  BOOST_CHECK_EQUAL(tc.end(), t.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).begin(), const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).end(), const_cast<const TensorN&>(t).end());
  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE( permute_constructor ) {
  Permutation perm = make_perm();

  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, perm));
  TensorN x(t, perm);

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm ^ r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ord(perm ^ t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], t[i]);
  }
}

BOOST_AUTO_TEST_CASE( permute_constructor_tensor ) {
  const std::array<std::size_t, 4> start = {{0ul, 0ul, 0ul, 0ul}};
  const std::array<std::size_t, 4> finish = {{24ul, 42ul, 16ul, 30ul}};
  TensorN x(range_type(start, finish));
  rand_fill(1693, x.size(), x.data());

  std::array<unsigned int, 4> p = {{0,1,2,3}};

  while(std::next_permutation(p.begin(), p.end())) {
    Permutation perm(p.begin(), p.end());


    TensorN px;
    // check constructor
    BOOST_REQUIRE_NO_THROW(px = TensorN(x, perm));
    BOOST_CHECK(! px.empty());

    for(std::size_t i = 0ul; i < x.size(); ++i) {
      std::size_t pi = px.range().ord(perm ^ x.range().idx(i));
      BOOST_CHECK_EQUAL(px[pi], x[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE( unary_constructor ) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, TiledArray::math::Scale<int>(83)));
  TensorN x(t, TiledArray::math::Scale<int>(83));

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], 83 * t[i]);
}

BOOST_AUTO_TEST_CASE( unary_permute_constructor ) {
  Permutation perm = make_perm();

  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, TiledArray::math::Scale<int>(47), perm));
  TensorN x(t, TiledArray::math::Scale<int>(47), perm);

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm ^ r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ord(perm ^ t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], 47 * t[i]);
  }
}

BOOST_AUTO_TEST_CASE( binary_constructor ) {
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check default constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, s, TiledArray::math::Minus<int, int, int>()));
  TensorN x(t, s, TiledArray::math::Minus<int, int, int>());

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], t[i] - s[i]);
}

BOOST_AUTO_TEST_CASE( binary_perm_constructor ) {
  Permutation perm = make_perm();
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check default constructor
  BOOST_REQUIRE_NO_THROW(TensorN x(t, s, TiledArray::math::Minus<int, int, int>(), perm));
  TensorN x(t, s, TiledArray::math::Minus<int, int, int>(), perm);

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm ^ r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ord(perm ^ t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], t[i] - s[i]);
  }
}

BOOST_AUTO_TEST_CASE( clone ) {
  // check default constructor
  TensorN tc;
  BOOST_CHECK(tc.empty());
  BOOST_REQUIRE_NO_THROW(tc = t.clone());

  BOOST_CHECK_EQUAL(tc.empty(), t.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(tc.data(), t.data());
  BOOST_CHECK_EQUAL(tc.size(), t.size());
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().start().begin(), t.range().start().end(),
      r.start().begin(), r.start().end());  // check start accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().finish().begin(), t.range().finish().end(),
      r.finish().begin(), r.finish().end());// check finish accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().size().begin(), t.range().size().end(),
      r.size().begin(), r.size().end());    // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(t.range().weight().begin(), t.range().weight().end(),
      r.weight().begin(), r.weight().end());    // check weight accessor
  BOOST_CHECK_EQUAL(t.range().volume(), r.volume());// check volume accessor
  BOOST_CHECK_EQUAL(t.range(), r);          // check range accessof
}

BOOST_AUTO_TEST_CASE( element_access )
{
  // check operator[] with array coordinate index and ordinal index
  for(std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_LT(t[i], 42);
    BOOST_CHECK_EQUAL(t[r.idx(i)], t[i]);
  }

  // check out of range error
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(t[r.finish()], Exception);
  BOOST_CHECK_THROW(t[r.volume()], Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( iteration )
{
  BOOST_CHECK_EQUAL(t.begin(), const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(t.end(), const_cast<const TensorN&>(t).end());

  for(TensorN::iterator it = t.begin(); it != t.end(); ++it) {
    BOOST_CHECK_LT(*it, 42);
    BOOST_CHECK_EQUAL(*it, t[std::distance(t.begin(), it)]);
  }

  // check iterator assignment
  TensorN::iterator it = t.begin();
  BOOST_CHECK_NE(t[0], 88);
  *it = 88;
  BOOST_CHECK_EQUAL(t[0], 88);

  // Check that the iterators of an empty tensor are equal
  TensorN t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE( element_assignment )
{

  // verify preassignment conditions
  BOOST_CHECK_NE(t[1], 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t[1] = 2, 2) ;
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t[1], 2);
}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = (t.range().volume() * sizeof(int) + sizeof(size_type) * (r.dim() * 4 + 2))*2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  BOOST_REQUIRE_NO_THROW(oar & t);
  std::size_t nbyte = oar.size();
  oar.close();

  TensorN ts;
  madness::archive::BufferInputArchive iar(buf,nbyte);
  BOOST_REQUIRE_NO_THROW(iar & ts);
  iar.close();

  delete [] buf;

  BOOST_CHECK_EQUAL(t.range(), ts.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(t.begin(), t.end(), ts.begin(), ts.end());
}

BOOST_AUTO_TEST_CASE( swap )
{
  TensorN s = make_tensor(79, 1559);
  rand_fill(431, s.size(), s.data());

  // Store a copy of the current state.
  range_type t_range = t.range();
  const int* const t_data = t.data();
  range_type s_range = s.range();
  const int* const s_data = s.data();

  BOOST_REQUIRE_NO_THROW(t.swap(s));

  // Check that the data has been moved correctly
  BOOST_CHECK_EQUAL(t.range(), s_range);
  BOOST_CHECK_EQUAL(t.data(), s_data);
  BOOST_CHECK_EQUAL(s.range(), t_range);
  BOOST_CHECK_EQUAL(s.data(), t_data);
}

BOOST_AUTO_TEST_CASE( unary_op ) {
  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(x = t.unary(TiledArray::math::Scale<int>(83)));

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_EQUAL(x.range(), r);

  // Check that the data pointers are correct
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  // Check that the element values are correct
  for(std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], 83 * t[i]);
}

BOOST_AUTO_TEST_CASE( unary_permute_op ) {
  Permutation perm = make_perm();

  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(x = t.unary(TiledArray::math::Scale<int>(47), perm));

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm ^ r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ord(perm ^ t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], 47 * t[i]);
  }
}

BOOST_AUTO_TEST_CASE( binary_op ) {
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(x = t.binary(s, TiledArray::math::Minus<int, int, int>()));

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i)
    BOOST_CHECK_EQUAL(x[i], t[i] - s[i]);
}

BOOST_AUTO_TEST_CASE( binary_perm_op ) {
  Permutation perm = make_perm();
  TensorN s(r);
  rand_fill(431, s.size(), s.data());

  // check default constructor
  // check operation
  TensorN x;
  BOOST_REQUIRE_NO_THROW(x = t.binary(s, TiledArray::math::Minus<int, int, int>(), perm));

  BOOST_CHECK(! x.empty());

  // Check that range data is correct.
  BOOST_CHECK_NE(x.data(), t.data());
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), perm ^ r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(), const_cast<const TensorN&>(x).end()), r.volume());

  for(std::size_t i = 0ul; i < x.size(); ++i) {
    std::size_t pi = x.range().ord(perm ^ t.range().idx(i));
    BOOST_CHECK_EQUAL(x[pi], t[i] - s[i]);
  }
}


BOOST_AUTO_TEST_SUITE_END()

