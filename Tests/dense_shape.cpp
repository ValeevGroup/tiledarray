#include "unit_test_config.h"
#include "global_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

// =============================================================================
// DenseShape Test Suite

BOOST_FIXTURE_TEST_SUITE( dense_shape_suite, DenseShapeFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(DenseShapeT s(r, m));
}

BOOST_AUTO_TEST_CASE( clone )
{
  std::shared_ptr<ShapeT> s = ds.clone();

  BOOST_CHECK(s->type() == typeid(DenseShapeT));
}

BOOST_AUTO_TEST_CASE( is_local )
{
  // For dense shape, is local should always be true.

  ordinal_index o = 0ul;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++o) {
    // with an ordinal_index
    BOOST_CHECK(ds.is_local(o));
    // with an index
    BOOST_CHECK(ds.is_local(*it));
    // with a key initialized with an ordianl_index
    BOOST_CHECK(ds.is_local(key_type(o)));
    // with a key initalized with a index
    BOOST_CHECK(ds.is_local(key_type(*it)));
    // with a key initialized with an ordianl_index and an index
    BOOST_CHECK(ds.is_local(key_type(o, *it)));
  }
}

BOOST_AUTO_TEST_CASE( probe )
{
  // For dense shapes, probe should always be true.

  ordinal_index o = 0ul;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++o) {
    // with an ordinal_index
    BOOST_CHECK(ds.probe(o));
    // with an index
    BOOST_CHECK(ds.probe(*it));
    // with a key initialized with an ordianl_index
    BOOST_CHECK(ds.probe(key_type(o)));
    // with a key initalized with a index
    BOOST_CHECK(ds.probe(key_type(*it)));
    // with a key initialized with an ordianl_index and an index
    BOOST_CHECK(ds.probe(key_type(o, *it)));
  }

}

BOOST_AUTO_TEST_CASE( array )
{
  DenseShapeT::array_type a = ds.make_shape_map();
  for(DenseShapeT::array_type::const_iterator it = a.begin(); it != a.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);
}

BOOST_AUTO_TEST_SUITE_END()

