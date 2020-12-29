/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#include "TiledArray/transform_iterator.h"
#include <vector>
#include "unit_test_config.h"

using TiledArray::detail::BinaryTransformIterator;
using TiledArray::detail::UnaryTransformIterator;

struct sqr : public std::unary_function<int, int> {
  int operator()(const int& i) const { return (i * i); }
};

struct object {
  int i;
};

struct to_object {
  typedef object result_type;

  result_type operator()(const object& obj) const { return obj; }
};

struct TransformIterFixture {
  typedef BinaryTransformIterator<std::vector<int>::iterator,
                                  std::vector<int>::iterator,
                                  std::multiplies<int> >
      BIter;
  typedef BinaryTransformIterator<std::vector<int>::const_iterator,
                                  std::vector<int>::const_iterator,
                                  std::multiplies<int> >
      BCIter;
  typedef UnaryTransformIterator<std::vector<int>::iterator, std::negate<int> >
      UIter;
  typedef UnaryTransformIterator<std::vector<int>::const_iterator,
                                 std::negate<int> >
      UCIter;

  TransformIterFixture()
      : Bbegin(v1.begin(), v2.begin(), std::multiplies<int>()),
        Bend(v1.end(), v2.end(), std::multiplies<int>()),
        Ubegin(v1.begin(), std::negate<int>()),
        Uend(v1.end(), std::negate<int>()) {}

  ~TransformIterFixture() {}

  static std::vector<int> make_vec(int scale) {
    std::vector<int> result;
    for (int i = 1; i <= 10; ++i) result.push_back(scale * i);

    return result;
  }

  static std::vector<int> v1;
  static std::vector<int> v2;

  BIter Bbegin;
  BIter Bend;
  UIter Ubegin;
  UIter Uend;
};

std::vector<int> TransformIterFixture::v1(make_vec(1));
std::vector<int> TransformIterFixture::v2(make_vec(2));

BOOST_FIXTURE_TEST_SUITE(transform_iterator_suite, TransformIterFixture, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(constructor) {
  // Check binary transform iterator constructor
  BOOST_REQUIRE_NO_THROW(
      BIter it(v1.begin(), v2.begin(), std::multiplies<int>()));
  {
    BIter it(v1.begin(), v2.begin(), std::multiplies<int>());
    BOOST_CHECK(it.base1() == v1.begin());
    BOOST_CHECK(it.base2() == v2.begin());
  }

  // Check binary transform iterator constructor with convertible iterators
  BOOST_REQUIRE_NO_THROW(
      BCIter it(v1.begin(), v2.begin(), std::multiplies<int>()));
  {
    BCIter it(v1.begin(), v2.begin(), std::multiplies<int>());
    BOOST_CHECK(it.base1() == v1.begin());
    BOOST_CHECK(it.base2() == v2.begin());
  }

  // Check binary transform iterator copy constructor
  BOOST_REQUIRE_NO_THROW(BIter it(Bbegin));
  {
    BIter it(Bbegin);
    BOOST_CHECK(it.base1() == Bbegin.base1());
    BOOST_CHECK(it.base2() == Bbegin.base2());
  }

  // Check binary transform iterator copy conversion constructor
  BOOST_REQUIRE_NO_THROW(BCIter it(Bbegin));
  {
    BCIter it(Bbegin);
    BOOST_CHECK(it.base1() == Bbegin.base1());
    BOOST_CHECK(it.base2() == Bbegin.base2());
  }

  // Check binary transform iterator constructor
  BOOST_REQUIRE_NO_THROW(UIter it(v1.begin(), std::negate<int>()));
  {
    UIter it(v1.begin(), std::negate<int>());
    BOOST_CHECK(it.base() == v1.begin());
  }

  // Check binary transform iterator with convertible iterators
  BOOST_REQUIRE_NO_THROW(UCIter it(v1.begin(), std::negate<int>()));
  {
    UCIter it(v1.begin(), std::negate<int>());
    BOOST_CHECK(it.base() == v1.begin());
  }

  // Check binary transform iterator copy constructor
  BOOST_REQUIRE_NO_THROW(UIter it(Ubegin));
  {
    UIter it(Ubegin);
    BOOST_CHECK(it.base() == Ubegin.base());
  }

  // Check binary transform iterator copy conversion constructor
  BOOST_REQUIRE_NO_THROW(UCIter it(Ubegin));
  {
    UCIter it(Ubegin);
    BOOST_CHECK(it.base() == Ubegin.base());
  }
}

BOOST_AUTO_TEST_CASE(prefix_increment) {
  std::vector<int>::const_iterator it1 = v1.begin();
  std::vector<int>::const_iterator it2 = v2.begin();
  for (; it1 != v1.end(); ++it1, ++it2) {
    BOOST_CHECK(Bbegin.base1() == it1);
    BOOST_CHECK(Bbegin.base2() == it2);
    BOOST_CHECK(Ubegin.base() == it1);

    BIter bit = ++Bbegin;
    UIter uit = ++Ubegin;

    BOOST_CHECK(Bbegin.base1() == (it1 + 1));
    BOOST_CHECK(Bbegin.base2() == (it2 + 1));
    BOOST_CHECK(Ubegin.base() == (it1 + 1));
    BOOST_CHECK(bit.base1() == (it1 + 1));
    BOOST_CHECK(bit.base2() == (it2 + 1));
    BOOST_CHECK(uit.base() == (it1 + 1));
  }
}

BOOST_AUTO_TEST_CASE(postfix_increment) {
  std::vector<int>::const_iterator it1 = v1.begin();
  std::vector<int>::const_iterator it2 = v2.begin();
  for (; it1 != v1.end(); ++it1, ++it2) {
    BOOST_CHECK(Bbegin.base1() == it1);
    BOOST_CHECK(Bbegin.base2() == it2);
    BOOST_CHECK(Ubegin.base() == it1);

    BIter bit = Bbegin++;
    UIter uit = Ubegin++;

    BOOST_CHECK(Bbegin.base1() == (it1 + 1));
    BOOST_CHECK(Bbegin.base2() == (it2 + 1));
    BOOST_CHECK(Ubegin.base() == (it1 + 1));
    BOOST_CHECK(bit.base1() == it1);
    BOOST_CHECK(bit.base2() == it2);
    BOOST_CHECK(uit.base() == it1);
  }
}

BOOST_AUTO_TEST_CASE(dereference) {
  // Check that dereference correctly transforms the object
  BOOST_CHECK_EQUAL(*Bbegin, (*v1.begin()) * (*v2.begin()));
  BOOST_CHECK_EQUAL(*Ubegin, -(*v1.begin()));

  // Check that dereferencing objects works correctly
  object obj;
  obj.i = 2;
  UnaryTransformIterator<object*, to_object> it(&obj, to_object());
  BOOST_CHECK_EQUAL((*it).i, obj.i);
  BOOST_CHECK_EQUAL(it->i, obj.i);
}

BOOST_AUTO_TEST_CASE(compare) {
  {
    BIter it(Bbegin);
    BCIter cit(Bbegin);

    BOOST_CHECK(it == Bbegin);
    BOOST_CHECK(cit == Bbegin);
    BOOST_CHECK(it.base1() == Bbegin.base1());
    BOOST_CHECK(it.base2() == Bbegin.base2());
    BOOST_CHECK(cit.base1() == Bbegin.base1());
    BOOST_CHECK(cit.base2() == Bbegin.base2());
    ++it;
    ++cit;
    BOOST_CHECK(it != Bbegin);
    BOOST_CHECK(cit != Bbegin);
  }

  {
    UIter it(Ubegin);
    UCIter cit(Ubegin);

    BOOST_CHECK(it == Ubegin);
    BOOST_CHECK(cit == Ubegin);
    BOOST_CHECK(it.base() == Ubegin.base());
    BOOST_CHECK(cit.base() == Ubegin.base());
    ++it;
    ++cit;
    BOOST_CHECK(it != Ubegin);
    BOOST_CHECK(cit != Ubegin);
  }
}

BOOST_AUTO_TEST_CASE(loop) {
  std::vector<int>::iterator v1it;
  std::vector<int>::iterator v2it;
  v1it = v1.begin();
  v2it = v2.begin();
  for (BIter it = Bbegin; it != Bend; ++it, ++v1it, ++v2it) {
    BOOST_CHECK_EQUAL(*it, (*v1it * *v2it));
  }

  v1it = v1.begin();
  v2it = v2.begin();
  for (BCIter it = Bbegin; it != Bend; ++it, ++v1it, ++v2it) {
    BOOST_CHECK_EQUAL(*it, (*v1it * *v2it));
  }

  v1it = v1.begin();
  for (UIter it = Ubegin; it != Uend; ++it, ++v1it) {
    BOOST_CHECK_EQUAL(*it, -(*v1it));
  }

  v1it = v1.begin();
  for (UCIter it = Ubegin; it != Uend; ++it, ++v1it) {
    BOOST_CHECK_EQUAL(*it, -(*v1it));
  }
}

BOOST_AUTO_TEST_SUITE_END()
