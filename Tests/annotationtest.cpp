/*
 * annotationtest.cpp
 *
 *  Created on: Nov 9, 2009
 *      Author: justus
 */

#include "annotation.h"
#include "tiled_range1.h"
#include "coordinates.h"
#include "permutation.h"
#include <iostream>
#include <math.h>
#include <utility>
#include <boost/test/unit_test.hpp>

using namespace TiledArray;
using TiledArray::expressions::Annotation;
using TiledArray::expressions::VariableList;

struct AnnotationFixture {
  static const size_t ndim = 3;

  typedef Annotation::size_array size_array;
  typedef Annotation::ordinal_type ordinal_type;
  typedef ArrayCoordinate<double,ndim,LevelTag<0>, CoordinateSystem<ndim> > index_type;

  AnnotationFixture() : v("a,b,c"), s(ndim) {
    const size_t primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for(size_t i=0; i<ndim; ++i) s[i] = primes[i];
    Annotation* aa = new Annotation(s.begin(), s.end(), v);
    a = *aa;
    t = *aa;
    delete aa;
  }
  ~AnnotationFixture() { }

  VariableList v;
  size_array s;
  Annotation a, t;
};

BOOST_FIXTURE_TEST_SUITE( annotation_suite , AnnotationFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(a.size().begin(), a.size().end(),
                                t.size().begin(), t.size().end());    // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(a.weight().begin(), a.weight().end(),
                                t.weight().begin(), t.weight().end()); // check weight accessor
  BOOST_CHECK_EQUAL(a.vars(), v); // check variable list accessor
  BOOST_CHECK_EQUAL(a.dim(), t.dim()); // check dimension accessor
  BOOST_CHECK_EQUAL(a.volume(), t.volume());// check volume accessor
  BOOST_CHECK_EQUAL(a.order(), t.order());// check order accessor
}

BOOST_AUTO_TEST_CASE( include )
{
  BOOST_CHECK(a.includes(index_type(0,0,0)));
  BOOST_CHECK(a.includes(index_type(2,4,4)));
  BOOST_CHECK(! a.includes(index_type(4,4,5)));
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(Annotation a0); // check default constructor
  Annotation a0;
  BOOST_CHECK_EQUAL(a0.volume(), 0u);
  BOOST_REQUIRE_THROW(a0.includes(index_type(0,0,0)), std::runtime_error);

  BOOST_REQUIRE_NO_THROW(Annotation ac(a)); // check copy constructor
  Annotation ac(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.size().begin(), ac.size().end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.weight().begin(), ac.weight().end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(ac.vars(), a.vars());
  BOOST_CHECK_EQUAL(ac.dim(), a.dim());
  BOOST_CHECK_EQUAL(ac.volume(), a.volume());
  BOOST_CHECK_EQUAL(ac.order(), a.order());

  BOOST_REQUIRE_NO_THROW(Annotation a1(s.begin(), s.end(), v)); // check construction from sizes and annotation
  Annotation a1(s.begin(), s.end(), v);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.size().begin(), a1.size().end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.weight().begin(), a1.weight().end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(a1.vars(), a.vars());
  BOOST_CHECK_EQUAL(a1.dim(), a.dim());
  BOOST_CHECK_EQUAL(a1.volume(), a.volume());
  BOOST_CHECK_EQUAL(a1.order(), a.order());

  BOOST_REQUIRE_NO_THROW(Annotation a2(s.begin(), s.end(), a.weight().begin(), a.weight().end(), a.volume(), v)); // check construction from sizes, weights, size, and annotation
  Annotation a2(s.begin(), s.end(), a.weight().begin(), a.weight().end(), a.volume(), v);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.size().begin(), a2.size().end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.weight().begin(), a2.weight().end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(a2.vars(), a.vars());
  BOOST_CHECK_EQUAL(a2.dim(), a.dim());
  BOOST_CHECK_EQUAL(a2.volume(), a.volume());
  BOOST_CHECK_EQUAL(a2.order(), a.order());
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Annotation ac;  ac = a;
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.size().begin(), ac.size().end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.weight().begin(), ac.weight().end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(ac.vars(), a.vars());
  BOOST_CHECK_EQUAL(ac.dim(), a.dim());
  BOOST_CHECK_EQUAL(ac.volume(), a.volume());
  BOOST_CHECK_EQUAL(ac.order(), a.order());
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<ndim> p(1,2,0);
  size_array s1 = p ^ s;
  VariableList v1 = p ^ v;
  Annotation a1(s1.begin(), s1.end(), v1);
  Annotation a2 = p ^ a;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.size().begin(), a2.size().end(),
                                a1.size().begin(), a1.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.weight().begin(), a2.weight().end(),
                                a1.weight().begin(), a1.weight().end());
  BOOST_CHECK_EQUAL(a2.vars(), a1.vars());
  BOOST_CHECK_EQUAL(a2.dim(), a1.dim());
  BOOST_CHECK_EQUAL(a2.volume(), a1.volume());
  BOOST_CHECK_EQUAL(a2.order(), a1.order());
}

BOOST_AUTO_TEST_SUITE_END()

