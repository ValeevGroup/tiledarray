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

// inner alignment a,b,c (A) vs b,c,a (B) -- the flat dot must align them.
BOOST_AUTO_TEST_CASE(permuted_inner) {
  TA::TiledRange a_tr{{0, 2, 4}, {0, 2}, {0, 2, 4}};  // outer i, j, k
  TA::TiledRange b_tr{{0, 2}, {0, 2, 4}};             // outer j, k
  auto A = random_array<ArrayToT>(a_tr, {3, 2, 4});   // inner a,b,c
  auto B = random_array<ArrayToT>(b_tr, {2, 4, 3});   // inner b,c,a
  ArrayT ref = TA::einsum<DeNest::True>("ijk;abc,jk;bca->ki", A, B);
  ArrayT out;
  out("k,i") = A("i,j,k;a,b,c").dot_inner(B("j,k;b,c,a"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

BOOST_AUTO_TEST_CASE(no_hadamard_contraction) {
  TA::TiledRange a_tr{{0, 2, 4}, {0, 2}};  // outer i, j
  TA::TiledRange b_tr{{0, 2}, {0, 2, 4}};  // outer j, l
  auto A = random_array<ArrayToT>(a_tr, {3, 2});
  auto B = random_array<ArrayToT>(b_tr, {3, 2});
  ArrayT ref = TA::einsum<DeNest::True>("ij;ab,jl;ab->il", A, B);
  ArrayT out;
  out("i,l") = A("i,j;a,b").dot_inner(B("j,l;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

BOOST_AUTO_TEST_SUITE_END()
