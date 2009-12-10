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

namespace {
  const size_t primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
  const char varlist[] = "a,b,c,d,e,f,g,h,i,j,k";
}

struct AnnotationFixture {
  static const size_t ndim = 4;
  static const size_t __ndim;
  typedef Annotation::size_array size_array;
  typedef Annotation::ordinal_type ordinal_type;
  typedef ArrayCoordinate<double,ndim,LevelTag<0>, CoordinateSystem<ndim> > index_type;

  AnnotationFixture() : var(std::string(varlist, 2*ndim-1)), a(primes, primes+ndim, var) {
    assert(ndim >= 3ul);
    assert(ndim <= 11ul);
    const TiledArray::detail::DimensionOrder<ndim> dimord(order);
    vol = 1;
    for(unsigned int o=0; o<ndim; ++o) {
      const unsigned int d = dimord.order2dim(o);
      const size_t s = primes[d];
      size[d] = s;
      weight[d] = vol;
      vol *= s;
    }
  }
  ~AnnotationFixture() { }

  Annotation::volume_type vol;
  size_t size[ndim];
  size_t weight[ndim];
  static const TiledArray::detail::DimensionOrderType order;
  const VariableList var;
  Annotation a;
};

#if 0
const Annotation::volume_type AnnotationFixture::vol = 105;
const size_t AnnotationFixture::size[] = {3, 5, 7};
const size_t AnnotationFixture::weight[] = {35, 7, 1};
#endif
const TiledArray::detail::DimensionOrderType AnnotationFixture::order
    = detail::decreasing_dimension_order;
const size_t AnnotationFixture::__ndim = AnnotationFixture::ndim;

BOOST_FIXTURE_TEST_SUITE( annotation_suite , AnnotationFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(a.size().begin(), a.size().end(),
                                size, size + __ndim);    // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(a.weight().begin(), a.weight().end(),
                                weight, weight + __ndim); // check weight accessor
  BOOST_CHECK_EQUAL(a.vars(), var); // check variable list accessor
  BOOST_CHECK_EQUAL(a.dim(), __ndim); // check dimension accessor
  BOOST_CHECK_EQUAL(a.volume(), vol);// check volume accessor
  BOOST_CHECK_EQUAL(a.order(), order);// check order accessor
}

BOOST_AUTO_TEST_CASE( include )
{
  index_type in0;
  index_type in1(in0);  ++in1;
  index_type out0(primes);
  BOOST_CHECK(a.includes(in0));
  BOOST_CHECK(a.includes(in1));
  BOOST_CHECK(! a.includes(out0));
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
  BOOST_CHECK_EQUAL(ac.dim(), __ndim);
  BOOST_CHECK_EQUAL(ac.volume(), a.volume());
  BOOST_CHECK_EQUAL(ac.order(), a.order());

  // check construction from sizes and annotation
  BOOST_REQUIRE_NO_THROW(Annotation a1(size, size + __ndim, var));
  Annotation a1(size, size + __ndim, var);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.size().begin(), a1.size().end(),
                                size, size + __ndim);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.weight().begin(), a1.weight().end(),
                                weight, weight + __ndim);
  BOOST_CHECK_EQUAL(a1.vars(), var);
  BOOST_CHECK_EQUAL(a1.dim(), __ndim);
  BOOST_CHECK_EQUAL(a1.volume(), vol);
  BOOST_CHECK_EQUAL(a1.order(), order);

  // check construction from sizes, weights, size, and annotation
  BOOST_REQUIRE_NO_THROW(Annotation a2(size, size + __ndim, weight, weight + __ndim, vol, var));
  Annotation a2(size, size + __ndim, weight, weight + __ndim, vol, var);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.size().begin(), a2.size().end(),
                                size, size + __ndim);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.weight().begin(), a2.weight().end(),
                                weight, weight + __ndim);
  BOOST_CHECK_EQUAL(a2.vars(), var);
  BOOST_CHECK_EQUAL(a2.dim(), __ndim);
  BOOST_CHECK_EQUAL(a2.volume(), vol);
  BOOST_CHECK_EQUAL(a2.order(), order);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  const size_t s[] = { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 };
  Annotation ac(s, s + __ndim, var);
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
  const size_t perm_indices[] = {1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Permutation<ndim> p(perm_indices);
  size_array s1(size, size + __ndim);
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

