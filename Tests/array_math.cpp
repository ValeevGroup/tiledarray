#include "TiledArray/array_math.h"
#include "TiledArray/array.h"
#include "range_fixture.h"
#include "math_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::math;

struct ArrayMathFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::coordinate_system> ArrayN;

  typedef TiledArray::CoordinateSystem<2,
        GlobalFixture::coordinate_system::level,
        GlobalFixture::coordinate_system::order,
        GlobalFixture::coordinate_system::ordinal_index> coordinate_system2;

  typedef Array<int, coordinate_system2> Array2;


  ArrayMathFixture() : world(*GlobalFixture::world), a(world, tr), b(world, tr), c(world, tr) {
    for(ArrayN::range_type::volume_type i = 0; i < a.tiles().volume(); ++i) {
      if(a.is_local(i))
        a.set(i, 3);
      if(b.is_local(i))
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

  for(ArrayN::range_type::const_iterator it = c.tiles().begin(); it != c.tiles().end(); ++it) {

    if(c.is_local(*it)) {
      madness::Future<ArrayN::value_type> tile = c.find(*it);

      for(ArrayN::value_type::iterator it = tile.get().begin(); it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, 5);
    }
  }
}


BOOST_AUTO_TEST_CASE( contraction )
{
  BinaryOp<Array2, ArrayN, ArrayN, TileContract<Array2::value_type, ArrayN::value_type, ArrayN::value_type> >
    op(world, c.version());

  Array2 c2 = op(a(vars), b(MathFixture::make_var_list(1, GlobalFixture::element_coordinate_system::dim + 1)));

  for(Array2::range_type::const_iterator it = c2.tiles().begin(); it != c2.tiles().end(); ++it) {

    if(c2.is_local(*it) && (! c2.is_zero(*it))) {
      madness::Future<Array2::value_type> tile = c2.find(*it);

      for(Array2::value_type::iterator it = tile.get().begin(); it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, 5);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
