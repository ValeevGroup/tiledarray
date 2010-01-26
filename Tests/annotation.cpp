#include "TiledArray/annotation.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/coordinates.h"
#include "TiledArray/permutation.h"
#include "TiledArray/coordinate_system.h"
#include <iostream>
#include <math.h>
#include <utility>
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::expressions::Annotation;
using TiledArray::expressions::VariableList;

namespace {
  const size_t primes[] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
  const char varlist[] = "a,b,c,d,e,f,g,h,i,j,k";
}

template<std::size_t DIM>
struct AnnotationBaseFixture {
  BOOST_STATIC_ASSERT((DIM >= 3ul) && (DIM <= 11ul));


  AnnotationBaseFixture() { }
  ~AnnotationBaseFixture() { }

  static boost::array<std::size_t, DIM> make_size() {
    boost::array<std::size_t, DIM> result;
    std::copy(primes, primes + DIM, result.begin());
    return result;
  }

  static boost::array<std::size_t, DIM> make_weight() {
    boost::array<std::size_t, DIM> result;
    detail::calc_weight(primes, primes + DIM, result.begin());
    return result;
  }

  static const std::size_t dim;
  static const Annotation<std::size_t>::volume_type vol;
  static const boost::array<std::size_t, DIM> size;
  static const boost::array<std::size_t, DIM> weight;
  static const TiledArray::detail::DimensionOrderType order;
  static const VariableList var;
}; // struct AnnotationBaseFixture

template<std::size_t DIM>
const std::size_t AnnotationBaseFixture<DIM>::dim = DIM;
template<std::size_t DIM>
const std::size_t AnnotationBaseFixture<DIM>::vol = detail::volume(primes, primes + DIM);
template<std::size_t DIM>
const boost::array<std::size_t, DIM> AnnotationBaseFixture<DIM>::size = AnnotationBaseFixture<DIM>::make_size();
template<std::size_t DIM>
const boost::array<std::size_t, DIM> AnnotationBaseFixture<DIM>::weight = AnnotationBaseFixture<DIM>::make_weight();
template<std::size_t DIM>
const TiledArray::detail::DimensionOrderType AnnotationBaseFixture<DIM>::order
    = detail::decreasing_dimension_order;
template<std::size_t DIM>
const VariableList AnnotationBaseFixture<DIM>::var(std::string(varlist, varlist + 2*dim-1));

struct AnnotationFixture : public AnnotationBaseFixture<4> {
  // Note: You can change the number of dimensions used by modifying the
  // template parameter of AnnotationBaseFixture.
  typedef Annotation<std::size_t>::size_array size_array;
  typedef Annotation<std::size_t>::ordinal_type ordinal_type;
  typedef ArrayCoordinate<double,dim,LevelTag<0>, CoordinateSystem<dim> > index_type;

  AnnotationFixture() : s(size), w(weight),
      ca(size.begin(), size.end(), weight.begin(), weight.end(), vol, var),
      a(s.begin(), s.end(), w.begin(), w.end(), vol, var)
  { }
  ~AnnotationFixture() { }

  boost::array<std::size_t, dim> s;
  boost::array<std::size_t, dim> w;
  Annotation<const std::size_t> ca;
  Annotation<std::size_t> a;
};

BOOST_FIXTURE_TEST_SUITE( annotation_suite , AnnotationFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(a.size().begin(), a.size().end(),
                                size.begin(), size.end());    // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(a.weight().begin(), a.weight().end(),
                                weight.begin(), weight.end()); // check weight accessor
  BOOST_CHECK_EQUAL(a.vars(), var); // check variable list accessor
  BOOST_CHECK_EQUAL(a.dim(), dim); // check dimension accessor
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
  BOOST_REQUIRE_NO_THROW(Annotation<std::size_t> ac(a)); // check copy constructor
  Annotation<std::size_t> ac(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.size().begin(), ac.size().end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.weight().begin(), ac.weight().end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(ac.vars(), var);
  BOOST_CHECK_EQUAL(ac.dim(), dim);
  BOOST_CHECK_EQUAL(ac.volume(), a.volume());
  BOOST_CHECK_EQUAL(ac.order(), a.order());

  // check construction from sizes, weights, size, and annotation
  BOOST_REQUIRE_NO_THROW(Annotation<std::size_t> a2(s.begin(), s.end(), w.begin(), w.end(), vol, var));
  Annotation<std::size_t> a2(s.begin(), s.end(), w.begin(), w.end(), vol, var);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.size().begin(), a2.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.weight().begin(), a2.weight().end(),
      weight.begin(), weight.end());
  BOOST_CHECK_EQUAL(a2.vars(), var);
  BOOST_CHECK_EQUAL(a2.dim(), dim);
  BOOST_CHECK_EQUAL(a2.volume(), vol);
  BOOST_CHECK_EQUAL(a2.order(), order);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  typedef detail::CoordIterator<boost::array<std::size_t, dim>, order> CI;
  boost::array<std::size_t, dim> ss;
  ss.assign(2);
  boost::array<std::size_t, dim> ww;
  detail::calc_weight(CI::begin(ss), CI::end(ss), CI::begin(ww));
  std::size_t vv = detail::volume(ss.begin(), ss.end());
  Annotation<std::size_t> ac(ss.begin(), ss.end(), w.begin(), w.end(), vv, var);
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
  Permutation<dim> p(perm_indices);
  boost::array<std::size_t, dim> sp = p ^ size;
  boost::array<std::size_t, dim> wp;
  detail::calc_weight(sp.begin(), sp.end(), wp.begin());
  std::size_t volp = detail::volume(sp);
  VariableList varp = p ^ var;
  a ^= p;
  BOOST_CHECK_EQUAL_COLLECTIONS(sp.begin(), sp.end(),
                                a.size().begin(), a.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(wp.begin(), wp.end(),
                                a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(varp, a.vars());
  BOOST_CHECK_EQUAL(dim, a.dim());
  BOOST_CHECK_EQUAL(volp, a.volume());
  BOOST_CHECK_EQUAL(order, a.order());
}

BOOST_AUTO_TEST_SUITE_END()

