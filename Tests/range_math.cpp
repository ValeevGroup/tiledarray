#include "TiledArray/range_math.h"
#include "TiledArray/annotated_array.h"
#include "unit_test_config.h"
#include "array_fixtures.h"

using TiledArray::Range;
using TiledArray::expressions::AnnotatedArray;

struct RangeMathFixture {
  typedef FakeArray<int, GlobalFixture::coordinate_system> array_type;
  typedef AnnotatedArray<array_type > fake_annotation;
  typedef fake_annotation::range_type range_type;
  typedef array_type::index index;

  static const range_type r;
  static const array_type a;
  static const VariableList vars;
  static const fake_annotation fa;

  RangeMathFixture() : result() { }

  range_type result;
}; // RangeMathFixture

const RangeMathFixture::range_type RangeMathFixture::r;
const RangeMathFixture::array_type RangeMathFixture::a(r);
const VariableList RangeMathFixture::vars(AnnotatedArrayFixture::make_var_list());
const RangeMathFixture::fake_annotation RangeMathFixture::fa(a, RangeMathFixture::vars);

BOOST_FIXTURE_TEST_SUITE( range_math_suite, RangeMathFixture )

BOOST_AUTO_TEST_CASE( construct )
{
  // Construct a binary range operation object

  // Construct a unary range operation object

  // Construct a binary range operation object
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
