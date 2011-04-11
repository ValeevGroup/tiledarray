#include "unit_test_config.h"
#include "global_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

// =============================================================================
// PredicatedShape Test Suite

BOOST_FIXTURE_TEST_SUITE( pred_shape_suite, PredShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(PredShapeT ps(r, m, p));
}

BOOST_AUTO_TEST_CASE( clone )
{
  std::shared_ptr<ShapeT> s = ps.clone();

  BOOST_CHECK(s->type() == typeid(PredShapeT));
}

BOOST_AUTO_TEST_CASE( is_local )
{
  // For predicate shape, is_local should always be true.

  ordinal_index o = 0ul;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++o) {
    // with an ordinal_index
    BOOST_CHECK(ps.is_local(o));
    // with an index
    BOOST_CHECK(ps.is_local(*it));
    // with a key initialized with an ordianl_index
    BOOST_CHECK(ps.is_local(key_type(o)));
    // with a key initalized with a index
    BOOST_CHECK(ps.is_local(key_type(*it)));
    // with a key initialized with an ordianl_index and an index
    BOOST_CHECK(ps.is_local(key_type(o, *it)));
  }
}

BOOST_AUTO_TEST_CASE( probe )
{
  // For dense shapes, probe should always be true.

  ShapeT* s = &ps;

  ordinal_index o = 0ul;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++o) {
    if((o % 2) == 0) {
      // Check for inclusion

      // with an ordinal_index
      BOOST_CHECK(s->probe(o));
      // with an index
      BOOST_CHECK(s->probe(*it));
      // with a key initialized with an ordianl_index
      BOOST_CHECK(s->probe(key_type(o)));
      // with a key initalized with a index
      BOOST_CHECK(s->probe(key_type(*it)));
      // with a key initialized with an ordianl_index and an index
      BOOST_CHECK(s->probe(key_type(o, *it)));
    } else {
      // Check for exclusion

      // with an ordinal_index
      BOOST_CHECK(! s->probe(o));
      // with an index
      BOOST_CHECK(! s->probe(*it));
      // with a key initialized with an ordianl_index
      BOOST_CHECK(! s->probe(key_type(o)));
      // with a key initalized with a index
      BOOST_CHECK(! s->probe(key_type(*it)));
      // with a key initialized with an ordianl_index and an index
      BOOST_CHECK(! s->probe(key_type(o, *it)));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

