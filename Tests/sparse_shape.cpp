#include "unit_test_config.h"
#include "global_fixture.h"
#include "shape_fixtures.h"

#include <algorithm>

using namespace TiledArray;

// =============================================================================
// SparseShape Test Suite

BOOST_FIXTURE_TEST_SUITE( sparse_shape_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW((SparseShapeT(* GlobalFixture::world, r, m, list.begin(), list.end())));
  SparseShapeT::array_type a(r, 1);
  BOOST_REQUIRE_NO_THROW((SparseShapeT(* GlobalFixture::world, r, m, a)));
  BOOST_REQUIRE_NO_THROW((SparseShapeT(* GlobalFixture::world, r, m, ss, ss)));
  BOOST_REQUIRE_NO_THROW((SparseShapeT(* GlobalFixture::world, r, m, ss)));
}

BOOST_AUTO_TEST_CASE( clone )
{
  std::shared_ptr<ShapeT> s = ss.clone();

  BOOST_CHECK(s->type() == typeid(SparseShapeT));
}

BOOST_AUTO_TEST_CASE( is_local )
{
  // For sparse shape, the data may or may not be local.
  // TODO: At the moment, all sparse shape data is replicated. When that changes
  // these tests will need to change as well.

  ordinal_index o = 0ul;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++o) {
    // with an ordinal_index
    BOOST_CHECK(ss.is_local(o));
    // with an index
    BOOST_CHECK(ss.is_local(*it));
  }
}

BOOST_AUTO_TEST_CASE( probe )
{
  // For dense shapes, probe should always be true.

  ordinal_index o = 0ul;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++o) {
    if(ss.is_local(o)) { // Only some of the data is local and we can only check local data.
      if(std::find(list.begin(), list.end(), o) != list.end()) {
        // Check for inclusion

        // with an ordinal_index
        BOOST_CHECK(ss.probe(o));
        // with an index
        BOOST_CHECK(ss.probe(*it));
      } else {
        // Check for exclusion

        // with an ordinal_index
        BOOST_CHECK(! ss.probe(o));
        // with an index
        BOOST_CHECK(! ss.probe(*it));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( array )
{
  SparseShapeT::array_type a = ss.make_shape_map();

  ordinal_index o = 0ul;
  for(SparseShapeT::array_type::const_iterator it = a.begin(); it != a.end(); ++it, ++o) {
    if(ss.is_local(o)) { // Only some of the data is local and we can only check local data.
      if(ss.probe(o)) {
        BOOST_CHECK_EQUAL(a[o], 1);
      } else {
        BOOST_CHECK_EQUAL(a[o], 0);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

