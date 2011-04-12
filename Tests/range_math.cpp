#include "TiledArray/range_math.h"
#include "unit_test_config.h"
#include "math_fixture.h"

using TiledArray::math::BinaryOp;
using TiledArray::math::UnaryOp;

struct RangeMathFixture : public MathFixture {

  RangeMathFixture() : result() { }

  range_type result;
}; // RangeMathFixture

BOOST_FIXTURE_TEST_SUITE( range_math_suite, RangeMathFixture )

BOOST_AUTO_TEST_CASE( construct )
{
  // Construct a binary range operation object
  BOOST_REQUIRE_NO_THROW((BinaryOp<range_type, array_annotation, array_annotation, std::plus>()));

  // Construct a unary range operation object
  BOOST_REQUIRE_NO_THROW((BinaryOp<range_type, array_annotation, array_annotation, std::multiplies>()));

  // Construct a binary range operation object
  BOOST_REQUIRE_NO_THROW((UnaryOp<range_type, array_annotation, std::negate>()));
}

BOOST_AUTO_TEST_CASE( binary )
{
  // Test binary range operation
}

BOOST_AUTO_TEST_CASE( unary )
{
  // Test unary range operation
}

BOOST_AUTO_TEST_CASE( binary_contract )
{
  // Test contraction range operation
}

BOOST_AUTO_TEST_SUITE_END()
