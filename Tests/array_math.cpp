#include "TiledArray/array_math.h"
#include "TiledArray/array.h"
#include "range_fixture.h"
#include "math_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::math;

struct ArrayMathFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::coordinate_system> ArrayN;


  ArrayMathFixture() : world(*GlobalFixture::world), a(world, tr), b(world, tr), c(world, tr) {
    for(ArrayN::range_type::volume_type i = 0; i < a.tiles().volume(); ++i) {
      a.set(i, 3);
      b.set(i, 2);
    }
  }

  madness::World& world;
  ArrayN a;
  ArrayN b;
  ArrayN c;

  static const std::string vars;
};

const std::string ArrayMathFixture::vars = MathFixture::make_var_list();

BOOST_FIXTURE_TEST_SUITE( array_math_suite , ArrayMathFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
//  BOOST_REQUIRE_NO_THROW( (BinaryOp<ArrayN, ArrayN, ArrayN, std::plus<ArrayN::value_type> > x(world, std::plus<ArrayN::value_type>())) );

}

BOOST_AUTO_TEST_CASE( addition )
{
  BinaryOp<ArrayN, ArrayN, ArrayN, std::plus<ArrayN::value_type> >
    op(world, c.version(), std::plus<ArrayN::value_type>());

  c = op(a(vars), b(vars));
}

BOOST_AUTO_TEST_SUITE_END()
