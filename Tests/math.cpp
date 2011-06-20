#include "math_fixture.h"
#include "TiledArray/tile.h"
#include <assert.h>
#include "unit_test_config.h"

using TiledArray::expressions::VariableList;
using namespace TiledArray;
using namespace TiledArray::math;

const VariableList MathFixture::vars(make_var_list());
const MathFixture::range_type MathFixture::r(
    MathFixture::index(0),
    MathFixture::index(5));
const MathFixture::array_type MathFixture::f1(r, 1);
const MathFixture::array_type MathFixture::f2(r, 2);
const MathFixture::array_type MathFixture::f3(r, 3);


const MathFixture::array_annotation MathFixture::a1(f1, VariableList(make_var_list()));
const MathFixture::array_annotation MathFixture::a2(f2, VariableList(make_var_list()));
const MathFixture::array_annotation MathFixture::a3(f3, VariableList(make_var_list(1,
    GlobalFixture::element_coordinate_system::dim + 1)));

std::string MathFixture::make_var_list(std::size_t first, std::size_t last) {
  assert(abs(last - first) <= 24);
  assert(last < 24);

  std::string result;
  result += 'a' + first;
  for(++first; first != last; ++first) {
    result += ",";
    result += 'a' + first;
  }

  return result;
}

BOOST_FIXTURE_TEST_SUITE( tile_math_suite, MathFixture)

BOOST_AUTO_TEST_CASE( construct_tile_plus )
{
  BOOST_CHECK_NO_THROW((TilePlus<array_type>()));
  BOOST_CHECK_NO_THROW((TileMinus<array_type>()));
  BOOST_CHECK_NO_THROW((TileScale<array_type>()));
}


BOOST_AUTO_TEST_CASE( addition )
{
  array_type t(r, 0);

  TilePlus<array_type> plus;


  // Check plus operation
  t = plus(f1, f2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 3);

  t = plus(f1, array_type());
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);

  t = plus(array_type(), f2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);

  t = plus(f1, 2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 3);

  t = plus(2, f2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 4);
}

BOOST_AUTO_TEST_CASE( subtraction )
{
  array_type t(r, 0);

  TileMinus<array_type> minus;


  // Check minus operation
  t = minus(f1, f2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t = minus(f1, array_type());
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);

  t = minus(array_type(), f2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -2);

  t = minus(f1, 2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t = minus(3, f2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);
}

BOOST_AUTO_TEST_CASE( scalar_multiplication )
{
  array_type t(r, 0);

  TileScale<array_type> scale;


  // Check scale operation
  t = scale(f1, 2);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);

  t = scale(3, f1);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 3);
}

BOOST_AUTO_TEST_CASE( negation )
{
  array_type t(r, 0);

  std::negate<array_type> negate;


  // Check scale operation
  t = negate(f1);
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);
}

BOOST_AUTO_TEST_CASE( contract )
{
  typedef array2_type::range_type range2_type;
  typedef array2_type::index index2;
  range2_type r2(index2(0), index2(a1.range().finish()[0],a2.range().finish()[0]));
  array2_type t;
  std::shared_ptr<Contraction<std::size_t> > cont(new Contraction<std::size_t>(a1.vars(), a3.vars()));
  array2_type::range_type r;
  cont->contract_range(r, a1.range(), a3.range());
  TileContract<array2_type, array_type, array_type> cont_op(cont, r);

  // Check scale operation
  t = cont_op(a1.array(), a3.array());
  for(array_type::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 75);
}


BOOST_AUTO_TEST_SUITE_END()
