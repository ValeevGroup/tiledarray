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

// Build an ArrayToT whose inner (a,b) extents are a deterministic function of
// the outer element index (i,j): Range{2 + (i%2), 3 + (j%2)}. This makes the
// inner extents NON-UNIFORM across outer cells. The extents depend ONLY on the
// outer index, never on `seed`, so two arrays built with the same TiledRange
// have congruent inner extents cell-by-cell -- only the element VALUES differ
// (they are offset by `seed`), which is exactly what a per-cell dot needs.
//
// The outer element at the global origin (0,0) is left with a DEFAULT-
// CONSTRUCTED (empty) inner tensor, so that cell's dot contributes nothing
// (result element is 0 there); the einsum oracle does the same.
ArrayToT make_nonuniform_tot(TA::World& world, TA::TiledRange const& tr,
                             int seed) {
  auto make_tile = [seed](ToT& tile, TA::Range const& rng) {
    tile = ToT(rng, [seed](TA::Range::index_type const& oix) {
      const auto i = oix[0];
      const auto j = oix[1];
      if (i == 0 && j == 0) return T{};  // empty inner cell at the origin
      TA::Range inner{2 + (i % 2), 3 + (j % 2)};
      T inner_tile(inner);
      int v = seed + 1;
      for (auto& x : inner_tile) x = v++;  // deterministic, seed-shifted values
      return inner_tile;
    });
    return tile.norm();
  };
  return TA::make_array<ArrayToT>(world, tr, make_tile);
}

// Deterministic ToT fill templated on the inner scalar type, so `int` and
// `double` copies carry IDENTICAL (integer-valued) data. Used to check that a
// mixed-precision dot_inner promotes the result scalar type rather than
// narrowing to the left operand's type. When `frac` is true a 0.5 fractional
// part is added (representable only in a floating result), so a narrowing-to-
// int bug is observable in the values. Inner extents are a uniform {3,2}.
template <typename InnerScalar>
TA::DistArray<TA::Tensor<TA::Tensor<InnerScalar>>> make_uniform_tot_typed(
    TA::World& world, TA::TiledRange const& tr, double base, bool frac) {
  using InT = TA::Tensor<InnerScalar>;
  using InToT = TA::Tensor<InT>;
  using InArr = TA::DistArray<InToT>;
  auto make_tile = [base, frac](InToT& tile, TA::Range const& rng) {
    tile = InToT(rng, [base, frac](TA::Range::index_type const& oix) {
      const auto i = oix[0];
      const auto j = oix[1];
      InT inner_tile(TA::Range{3, 2});
      double v = base + i + 2.0 * j;
      for (auto& x : inner_tile) {
        x = static_cast<InnerScalar>(frac ? (v + 0.5) : v);
        v += 1.0;
      }
      return inner_tile;
    });
    return tile.norm();
  };
  return TA::make_array<InArr>(world, tr, make_tile);
}
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

// Genuine tensor-of-tensors: inner extents depend on the outer element index
// (non-uniform), and the origin cell's inner tensor is empty. Cross-checked
// against the einsum oracle. A and B share per-cell inner extents (so the
// per-cell dot is well-defined) but carry different values.
BOOST_AUTO_TEST_CASE(nonuniform_and_empty_inner) {
  auto& world = TA::get_default_world();
  TA::TiledRange tr{{0, 2, 4}, {0, 4}};
  ArrayToT A = make_nonuniform_tot(world, tr, /*seed=*/1);
  ArrayToT B = make_nonuniform_tot(world, tr, /*seed=*/2);
  ArrayT ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ij", A, B);
  ArrayT out;
  out("i,j") = A("i,j;a,b").dot_inner(B("i,j;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

// SparsePolicy: the result SparseShape (per-tile Frobenius-norm estimates +
// threshold) must match what einsum produces. dot_inner reuses the contraction
// engine's norm-GEMM shape propagation; this checks the denest result-shape
// path (scalar result tiles derived from ToT operand norms). are_equal with
// ShapeComp::True compares the sparse shapes AND the element values; with
// T = Tensor<int> the arithmetic is exact, so the comparison is faithful.
BOOST_AUTO_TEST_CASE(sparse_policy) {
  using SpToT = TA::DistArray<ToT, TA::SparsePolicy>;
  using SpT = TA::DistArray<T, TA::SparsePolicy>;
  TA::TiledRange tr{{0, 2, 4}, {0, 2, 4}};
  auto A = random_array<SpToT>(tr, {3, 2});
  auto B = random_array<SpToT>(tr, {3, 2});

  SpT ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ij", A, B);
  ref.truncate();

  SpT out;
  out("i,j") = A("i,j;a,b").dot_inner(B("i,j;a,b"));
  out.truncate();

  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

// Conjugating inner_product semantics via a lazy .conj() on the LEFT ToT
// operand: A.conj().dot_inner(B) must compute sum_ab conj(A_ab) * B_ab per
// outer cell. Oracle: conjugate A EAGERLY, then run the NON-conjugating einsum
// denest. Both paths perform the identical real arithmetic (conjugate, then
// multiply-accumulate in the same per-cell summation order), so the complex
// results match bit-for-bit and exact are_equal comparison is faithful.
BOOST_AUTO_TEST_CASE(conjugated_left) {
  using Tc = TA::Tensor<std::complex<double>>;
  using ToTc = TA::Tensor<Tc>;
  using ArrayToTc = TA::DistArray<ToTc>;
  using ArrayTc = TA::DistArray<Tc>;
  TA::TiledRange tr{{0, 2, 4}, {0, 4}};
  auto A = random_array<ArrayToTc>(tr, {3, 2});
  auto B = random_array<ArrayToTc>(tr, {3, 2});

  // oracle: conjugate A's inner elements eagerly, then NON-conjugating denest
  // dot
  ArrayToTc Aconj;
  Aconj("i,j;a,b") = A("i,j;a,b").conj();
  ArrayTc ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ij", Aconj, B);

  ArrayTc out;
  out("i,j") = A("i,j;a,b").conj().dot_inner(B("i,j;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

// conj + CONTRACTION outer. The outer index j is contracted (il result), which
// routes the dot through the ContractReduce / element_nonreturn_op_
// ACCUMULATING path (not the Hadamard Mult/element_return_op_ path). This
// confirms the lazy .conj() on the LEFT operand survives all the way into
// flat_dot even when the per-cell scalar results are summed over the contracted
// outer index. Oracle: conjugate A eagerly, then NON-conjugating denest dot.
BOOST_AUTO_TEST_CASE(conjugated_contraction_outer) {
  using Tc = TA::Tensor<std::complex<double>>;
  using ToTc = TA::Tensor<Tc>;
  using ArrayToTc = TA::DistArray<ToTc>;
  using ArrayTc = TA::DistArray<Tc>;
  // outer: contract j -> il ; inner ab fully contracted; LEFT conjugated.
  TA::TiledRange a_tr{{0, 2, 4}, {0, 2}};  // i, j
  TA::TiledRange b_tr{{0, 2}, {0, 2, 4}};  // j, l
  auto A = random_array<ArrayToTc>(a_tr, {3, 2});
  auto B = random_array<ArrayToTc>(b_tr, {3, 2});
  ArrayToTc Aconj;
  Aconj("i,j;a,b") = A("i,j;a,b").conj();
  ArrayTc ref = TA::einsum<DeNest::True>("ij;ab,jl;ab->il", Aconj, B);
  ArrayTc out;
  out("i,l") = A("i,j;a,b").conj().dot_inner(B("j,l;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

// conj + PERMUTED Hadamard outer. The outer regime is Hadamard ij, but the
// result is transposed to ji, which reaches DotInnerEngine::make_tile_op(Perm).
// This confirms .conj() composes faithfully with an outer permutation of the
// plain-T result. Oracle: conjugate A eagerly, then NON-conjugating denest dot
// with the same ji output permutation.
BOOST_AUTO_TEST_CASE(conjugated_permuted_outer) {
  using Tc = TA::Tensor<std::complex<double>>;
  using ToTc = TA::Tensor<Tc>;
  using ArrayToTc = TA::DistArray<ToTc>;
  using ArrayTc = TA::DistArray<Tc>;
  // outer Hadamard ij, result transposed to ji; LEFT conjugated.
  TA::TiledRange tr{{0, 2, 4}, {0, 2, 4}};
  auto A = random_array<ArrayToTc>(tr, {3, 2});
  auto B = random_array<ArrayToTc>(tr, {3, 2});
  ArrayToTc Aconj;
  Aconj("i,j;a,b") = A("i,j;a,b").conj();
  ArrayTc ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ji", Aconj, B);
  ArrayTc out;
  out("j,i") = A("i,j;a,b").conj().dot_inner(B("i,j;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

// PLAIN permuted Hadamard outer (no conj). Closes the make_tile_op(Perm)
// coverage gap with exact integer arithmetic: the outer permutation of the
// plain-T result must match einsum's denest path. T = Tensor<int>, so the
// comparison is exact.
BOOST_AUTO_TEST_CASE(permuted_hadamard_outer) {
  TA::TiledRange tr{{0, 2, 4}, {0, 2, 4}};
  auto A = random_array<ArrayToT>(tr, {3, 2});
  auto B = random_array<ArrayToT>(tr, {3, 2});
  ArrayT ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ji", A, B);
  ArrayT out;
  out("j,i") = A("i,j;a,b").dot_inner(B("i,j;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

// Mixed-precision operands: int-inner dot_inner double-inner must produce a
// DOUBLE result (the promoted product type), not silently narrow to int (the
// left operand's type). Regression test for the result-scalar deduction
// (reported by Copilot review on PR #567).
BOOST_AUTO_TEST_CASE(mixed_inner_numeric_type) {
  using ToTd = TA::Tensor<TA::Tensor<double>>;
  using ArrayToTd = TA::DistArray<ToTd>;
  using ArrayTd = TA::DistArray<TA::Tensor<double>>;
  auto& world = TA::get_default_world();
  TA::TiledRange tr{{0, 2, 4}, {0, 4}};

  ArrayToT A_int = make_uniform_tot_typed<int>(world, tr, 1.0, false);
  ArrayToTd A_dbl =
      make_uniform_tot_typed<double>(world, tr, 1.0, false);  // == A_int values
  ArrayToTd B_dbl =
      make_uniform_tot_typed<double>(world, tr, 2.0, true);  // fractional

  // Type: the result scalar must be the promoted product type (double). This
  // static_assert fails to compile if the deduction narrows to the left
  // operand's int.
  using expr_t =
      std::decay_t<decltype(A_int("i,j;a,b").dot_inner(B_dbl("i,j;a,b")))>;
  static_assert(
      std::is_same_v<typename TA::expressions::ExprTrait<expr_t>::scalar_type,
                     double>,
      "dot_inner must promote the result scalar across mixed-precision "
      "operands");

  // Value: the double oracle (A as double with identical integer values). With
  // a narrowing-to-int bug the fractional contributions would be truncated and
  // the values would differ from this reference.
  ArrayTd ref = TA::einsum<DeNest::True>("ij;ab,ij;ab->ij", A_dbl, B_dbl);
  ArrayTd out;
  out("i,j") = A_int("i,j;a,b").dot_inner(B_dbl("i,j;a,b"));
  BOOST_REQUIRE((ToTArrayFixture::are_equal<ShapeComp::True>(ref, out)));
}

BOOST_AUTO_TEST_SUITE_END()
