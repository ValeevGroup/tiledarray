#include "math_fixture.h"
#include <assert.h>

using TiledArray::expressions::VariableList;

const VariableList MathFixture::vars(make_var_list());
const MathFixture::range_type MathFixture::r(
    MathFixture::index(0),
    MathFixture::index(5));
const MathFixture::array_type MathFixture::f1(r, 1);
const MathFixture::array_type MathFixture::f2(r, 2);
const MathFixture::array_type MathFixture::f3(MathFixture::range_type(
    MathFixture::index(0), MathFixture::index(6)), 3);


const MathFixture::array_annotation MathFixture::a1(f1, VariableList(make_var_list()));
const MathFixture::array_annotation MathFixture::a2(f2, VariableList(make_var_list()));
const MathFixture::array_annotation MathFixture::a3(f3, VariableList(make_var_list(1,
    GlobalFixture::element_coordinate_system::dim + 1)));

std::string MathFixture::make_var_list(std::size_t first, std::size_t last) {
  assert(abs(last - first) <= 24);
  assert(last < 24);
  static const char temp[26] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j',
      'k','l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

  std::string result;
  result += temp[first];
  for(++first; first != last; ++first) {
    result += ",";
    result += temp[first];
  }

  return result;
}

