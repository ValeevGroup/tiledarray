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

// outer: ipk x iqk -> ipq  (Hadamard i, external p & q, contracted-outer k);
// inner ab fully contracted. Exercises the Contraction/General outer routing.
BOOST_AUTO_TEST_CASE(hadamard_external_contracted_outer) {
  TA::TiledRange a_tr{{0, 2}, {0, 2, 3}, {0, 2, 4}};  // i, p, k
  TA::TiledRange b_tr{{0, 2}, {0, 3}, {0, 2, 4}};     // i, q, k
  auto A = random_array<ArrayToT>(a_tr, {3, 2});
  auto B = random_array<ArrayToT>(b_tr, {3, 2});

  ArrayT ref = TA::einsum<DeNest::True>("ipk;ab,iqk;ab->ipq", A, B);

  ArrayT out;
  out("i,p,q") = A("i,p,k;a,b").dot_inner(B("i,q,k;a,b"));

  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

BOOST_AUTO_TEST_SUITE_END()
