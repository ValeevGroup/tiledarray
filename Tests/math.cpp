#include "math_fixture.h"
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

