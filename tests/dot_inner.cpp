#include "TiledArray/expressions/einsum.h"
#include "tiledarray.h"
#include "tot_array_fixture.h"
#include "unit_test_config.h"

BOOST_AUTO_TEST_SUITE(dot_inner)

namespace {
using T = TA::Tensor<int>;
using ToT = TA::Tensor<T>;
using ArrayToT = TA::DistArray<ToT>;
using ArrayT = TA::DistArray<T>;
}  // namespace

BOOST_AUTO_TEST_CASE(hadamard_outer) {
  TA::TiledRange tr{{0, 2, 4}, {0, 4}};
  auto A = random_array<ArrayToT>(tr, {3, 2});
  auto B = random_array<ArrayToT>(tr, {3, 2});
  ArrayT ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ij", A, B);
  ArrayT out;
  out("i,j") = A("i,j;a,b").dot_inner(B("i,j;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

BOOST_AUTO_TEST_SUITE_END()
