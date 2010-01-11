#include "TiledArray/shape.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/predicate.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include <map>
#include "iteration_test.h"

using namespace TiledArray;

struct ShapeFixture {
  typedef Shape<std::size_t, 3> Shape3;
  typedef PredShape<std::size_t, 3, LowerTrianglePred<3> > PShape3;
  typedef PShape3::tiled_range_type TRange3;
  typedef PShape3::index_type index_type;

  ShapeFixture() {
    d0[0] = 0; d0[1] = 10; d0[2] = 20; d0[3] = 30;
    d1[0] = 0; d1[1] = 5; d1[2] = 10; d1[3] = 15; d1[4] = 20;
    d2[0] = 0; d2[1] = 3; d2[2] = 6; d2[3] = 9; d2[4] = 12; d2[5] = 15;
    dims[0] = TRange3::tiled_range1_type(d0.begin(), d0.end());
    dims[1] = TRange3::tiled_range1_type(d1.begin(), d1.end());
    dims[2] = TRange3::tiled_range1_type(d2.begin(), d2.end());

    r.resize(dims.begin(), dims.end());
    s = new PShape3(boost::make_shared<TRange3>(dims.begin(), dims.end()));
  }
  ~ShapeFixture() {
    delete s;
  }

  boost::array<std::size_t, 4> d0;
  boost::array<std::size_t, 5> d1;
  boost::array<std::size_t, 6> d2;
  boost::array<TRange3::tiled_range1_type, 3> dims;
  Shape3* s;
  TRange3 r;
};

// Note: Since we plan use Shape<> pointers to access all shapes and constructing
// PredShape on the stack is a little awkward and tricky, all tests will be done
// via a Shape<> pointer.

BOOST_FIXTURE_TEST_SUITE( shape_suite, ShapeFixture )

BOOST_AUTO_TEST_CASE( range_access )
{
  BOOST_CHECK_EQUAL(* (s->range()), r);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  Shape3* s1 = NULL;
  BOOST_REQUIRE_NO_THROW(s1 = new PShape3(boost::make_shared<TRange3>(dims.begin(), dims.end())));
  BOOST_CHECK_EQUAL(* (s1->range()), r);           // Check primary constructor.

  Shape3* s2 = NULL;
  BOOST_REQUIRE_NO_THROW( (s2 = new PShape3(* dynamic_cast<PShape3*>(s))) );       // check copy constructor.
  BOOST_CHECK_EQUAL(*(s2->range()), *(s->range()));
  BOOST_CHECK_NE(s2->range().get(), s->range().get());// check for deep copy.

  Shape3* s3 = NULL;
  BOOST_REQUIRE_NO_THROW( (s3 = new PShape3(s)) );     // check Shape<>* constructor.
  BOOST_CHECK_EQUAL(*(s3->range()), *(s->range()));
  BOOST_CHECK_NE(s3->range().get(), s->range().get());// check for deep copy.

  delete s1;
  delete s2;
  delete s3;
}

BOOST_AUTO_TEST_CASE( includes )
{
  BOOST_CHECK( (s->range()->tiles().includes(Shape3::index_type(1,2,3))) );
  BOOST_CHECK( (dynamic_cast<PShape3*>(s)->predicate().includes(Shape3::index_type(1,2,3))) );
  BOOST_CHECK( (s->includes(Shape3::index_type(1,2,3))) );  // check index included by predicate and range.

  BOOST_CHECK( (s->range()->tiles().includes(Shape3::index_type(2,1,0))) );
  BOOST_CHECK( (! dynamic_cast<PShape3*>(s)->predicate().includes(Shape3::index_type(2,1,0))) );
  BOOST_CHECK( (! s->includes(Shape3::index_type(2,1,0))) );// check index excluded by predicate only.

  BOOST_CHECK( (! s->range()->tiles().includes(Shape3::index_type(4,5,6))) );
  BOOST_CHECK( (dynamic_cast<PShape3*>(s)->predicate().includes(Shape3::index_type(4,5,6))) );
  BOOST_CHECK( (! s->includes(Shape3::index_type(4,5,6))) );  // check index included by predicate and excluded by range.

  BOOST_CHECK( (! s->range()->tiles().includes(Shape3::index_type(6,5,4))) );
  BOOST_CHECK( (! dynamic_cast<PShape3*>(s)->predicate().includes(Shape3::index_type(6,5,4))) );
  BOOST_CHECK( (! s->includes(Shape3::index_type(6,5,4))) );  // check index included by predicate and excluded by range.

  std::map<Shape3::index_type, bool> m;
  {
    Shape3::tiled_range_type::range_type::const_iterator it = s->range()->tiles().begin();
    bool includes;
    for(; it != s->range()->tiles().end(); ++it) {
      includes = true;
      for(unsigned int d = 1; d < s->dim(); ++d)
        if((*it)[d - 1] > (*it)[d])
          includes = false;

      m.insert(std::make_pair(*it, includes));
    }
  }

  for(std::map<Shape3::index_type, bool>::const_iterator it = m.begin(); it != m.end(); ++it)
    BOOST_CHECK_EQUAL(s->includes(it->first), it->second); // check to make sure the expected values are included/excluded.
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(Shape3::const_iterator it = s->begin(); it != s->end(); ++it) {
    for(unsigned int d = 1; d < s->dim(); ++d)
      BOOST_CHECK((*it)[d - 1] <= (*it)[d]);    // check that only indexes included by the predicate are found.

    BOOST_CHECK(s->range()->tiles().includes( *it )); // check that all tiles are included.
  }
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> p(2,0,1);
  Shape3* s1(s);
  Shape3::index_type i(1,2,3);

  BOOST_CHECK_EQUAL((s1 ^= p), s1); // check that the permutation returns the pointer.
  BOOST_CHECK(! dynamic_cast<PShape3*>(s)->predicate().includes(i)); // check that it is not included in the original.
  BOOST_CHECK(s1->includes(p ^ i));  // check that it is included after the permutation.
}

BOOST_AUTO_TEST_SUITE_END()

