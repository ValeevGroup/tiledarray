#include "TiledArray/transform_iterator.h"
#include "unit_test_config.h"
#include <vector>
#include <list>
#include <stdlib.h>
#include <cstring>

//using TiledArray::detail::PolyTransformIterator;
//
//struct sqr : public std::unary_function<int, double> {
//  double operator()(const int& i) const {
//    return (i * i);
//  }
//};
//
//struct object {
//  int i;
//};
//
//struct to_object : public std::unary_function<int, object> {
//  object operator()(const int& i) const {
//    object o;
//    o.i = i;
//    return o;
//  }
//};
//
//struct TransformIterFixture {
//  TransformIterFixture() : v(10, 0), cv(v),
//      begin(v.begin(), sqr()), end(v.end(), sqr()),
//      const_begin(cv.begin(), sqr()), const_end(cv.end(), sqr())
//  {
//    for(int i = 0; i < 10; ++i) {
//      v[i] = i + 2;
//    }
//  }
//
//  ~TransformIterFixture() { }
//
//  std::vector<int> v;
//  const std::vector<int>& cv;
//  PolyTransformIterator<double> begin;
//  PolyTransformIterator<double> end;
//  PolyTransformIterator<double> const_begin;
//  PolyTransformIterator<double> const_end;
//};
//
//BOOST_FIXTURE_TEST_SUITE( transform_iterator_suite , TransformIterFixture )
//
//BOOST_AUTO_TEST_CASE( constructor )
//{
//  BOOST_REQUIRE_NO_THROW(PolyTransformIterator<double> it1(v.begin(), sqr()));
//  BOOST_REQUIRE_NO_THROW(PolyTransformIterator<double> it2(begin));
//}
//
//BOOST_AUTO_TEST_CASE( dereference )
//{
//  BOOST_CHECK_EQUAL(4.0, *begin);
//  BOOST_CHECK_EQUAL(4.0, *const_begin);
//  BOOST_CHECK(typeid(double) == typeid(*begin));
//  BOOST_CHECK(typeid(double) == typeid(*const_begin));
//
//  PolyTransformIterator<object> ito(v.begin(), to_object());
//  object o = *ito;
//  BOOST_CHECK_EQUAL(2, o.i);
//  BOOST_CHECK_EQUAL(2, ito->i);
//}
//
//BOOST_AUTO_TEST_CASE( increment )
//{
//  BOOST_CHECK_EQUAL(4.0, *begin);
//  BOOST_CHECK_EQUAL(4.0, *(begin++));
//  BOOST_CHECK_EQUAL(9.0, *begin);
//  BOOST_CHECK_EQUAL(16.0, *(++begin));
//}
//
//BOOST_AUTO_TEST_CASE( compare )
//{
//  PolyTransformIterator<double> it(v.begin(), sqr());
//  BOOST_CHECK(it == begin);
//  ++it;
//  BOOST_CHECK(it != begin);
//  BOOST_CHECK(begin == const_begin);
//  BOOST_CHECK(end == const_end);
//}
//
//BOOST_AUTO_TEST_CASE( loop )
//{
//  std::vector<int>::iterator vit;
//  vit = v.begin();
//  for(PolyTransformIterator<double> it(begin); it != end; ++it) {
//    BOOST_CHECK_EQUAL(*it, (*vit * *vit));
//    ++vit;
//  }
//
//  vit = v.begin();
//  for(PolyTransformIterator<double> it(const_begin); it != const_end; ++it) {
//    BOOST_CHECK_EQUAL(*it, (*vit * *vit));
//    ++vit;
//  }
//
//  vit = v.begin();
//  for(PolyTransformIterator<double> it(begin); it != const_end; ++it) {
//    BOOST_CHECK_EQUAL(*it, (*vit * *vit));
//    ++vit;
//  }
//
//  vit = v.begin();
//  for(PolyTransformIterator<double> it(const_begin); it != end; ++it) {
//    BOOST_CHECK_EQUAL(*it, (*vit * *vit));
//    ++vit;
//  }
//}
//
//BOOST_AUTO_TEST_SUITE_END()
