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
  typedef Annotation::size_array size_array;
  typedef Annotation::ordinal_type ordinal_type;
  typedef ArrayCoordinate<double,3,LevelTag<0>, CoordinateSystem<3> > index_type;

  AnnotationFixture() : var("a,b,c"), a(size, size + dim, var) { }
  ~AnnotationFixture() { }

  static const unsigned int dim;
  static const Annotation::volume_type vol;
  static const size_t size[];
  static const size_t weight[];
  static const TiledArray::detail::DimensionOrderType order;
  const VariableList var;
  Annotation a;
};

const unsigned int AnnotationFixture::dim = 3;
const Annotation::volume_type AnnotationFixture::vol = 105;
const size_t AnnotationFixture::size[] = {3, 5, 7};
const size_t AnnotationFixture::weight[] = {35, 7, 1};
const TiledArray::detail::DimensionOrderType AnnotationFixture::order
    = detail::decreasing_dimension_order;

BOOST_FIXTURE_TEST_SUITE( annotation_suite , AnnotationFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(a.size().begin(), a.size().end(),
                                size, size + dim);    // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(a.weight().begin(), a.weight().end(),
                                weight, weight + dim); // check weight accessor
  BOOST_CHECK_EQUAL(a.vars(), var); // check variable list accessor
  BOOST_CHECK_EQUAL(a.dim(), dim); // check dimension accessor
  BOOST_CHECK_EQUAL(a.volume(), vol);// check volume accessor
  BOOST_CHECK_EQUAL(a.order(), order);// check order accessor
}

BOOST_AUTO_TEST_CASE( include )
{
  BOOST_CHECK(a.includes(index_type(0,0,0)));
  BOOST_CHECK(a.includes(index_type(2,4,4)));
  BOOST_CHECK(! a.includes(index_type(4,4,5)));
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(Annotation ac(a)); // check copy constructor
  Annotation ac(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.size().begin(), ac.size().end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.weight().begin(), ac.weight().end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(ac.vars(), var);
  BOOST_CHECK_EQUAL(ac.dim(), 3);
  BOOST_CHECK_EQUAL(ac.volume(), a.volume());
  BOOST_CHECK_EQUAL(ac.order(), a.order());

  // check construction from sizes and annotation
  BOOST_REQUIRE_NO_THROW(Annotation a1(size, size + dim, var));
  Annotation a1(size, size + dim, var);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.size().begin(), a1.size().end(),
                                size, size + dim);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.weight().begin(), a1.weight().end(),
                                weight, weight + dim);
  BOOST_CHECK_EQUAL(a1.vars(), var);
  BOOST_CHECK_EQUAL(a1.dim(), dim);
  BOOST_CHECK_EQUAL(a1.volume(), vol);
  BOOST_CHECK_EQUAL(a1.order(), order);

  // check construction from sizes, weights, size, and annotation
  BOOST_REQUIRE_NO_THROW(Annotation a2(size, size + dim, weight, weight + dim, vol, var));
  Annotation a2(size, size + dim, weight, weight + dim, vol, var);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.size().begin(), a2.size().end(),
                                size, size + dim);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.weight().begin(), a2.weight().end(),
                                weight, weight + dim);
  BOOST_CHECK_EQUAL(a2.vars(), var);
  BOOST_CHECK_EQUAL(a2.dim(), dim);
  BOOST_CHECK_EQUAL(a2.volume(), vol);
  BOOST_CHECK_EQUAL(a2.order(), order);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  const size_t s[] = { 10, 10, 10 };
  Annotation ac(s, s + dim, var);
  ac = a;
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
  Permutation<dim> p(1,2,0);
  size_array s1(size, size +dim);
  s1 ^= p;
  VariableList v1 = p ^ var;
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

