/// Unit tests for general-product (fused + contracted + free indices)
/// classification and layout in the expression layer.

#include "TiledArray/einsum/tiledarray.h"
#include "TiledArray/expressions/permopt.h"
#include "TiledArray/expressions/product.h"

#include "tiledarray.h"
#include "unit_test_config.h"

BOOST_AUTO_TEST_SUITE(general_product_suite, TA_UT_LABEL_SERIAL)

namespace TA = TiledArray;
using TA::expressions::compute_product_type;
using TA::expressions::GeneralPermutationOptimizer;
using TA::expressions::IndexList;
using TA::expressions::PermutationType;
using TA::expressions::TensorProduct;

BOOST_AUTO_TEST_CASE(classification_2arg_unchanged) {
  // bottom-up (no target): shared indices are assumed contracted, General is
  // never produced
  BOOST_CHECK(compute_product_type(IndexList("i,j"), IndexList("j,i")) ==
              TensorProduct::Hadamard);
  BOOST_CHECK(compute_product_type(IndexList("i,j"), IndexList("j,k")) ==
              TensorProduct::Contraction);
  BOOST_CHECK(compute_product_type(IndexList("b,i,j"), IndexList("b,j,k")) ==
              TensorProduct::Contraction);
  BOOST_CHECK(compute_product_type(IndexList{}, IndexList("i,j")) ==
              TensorProduct::Scale);
}

BOOST_AUTO_TEST_CASE(classification_3arg) {
  // pure Hadamard: all three related by permutation
  BOOST_CHECK(compute_product_type(IndexList("i,j"), IndexList("j,i"),
                                   IndexList("i,j")) ==
              TensorProduct::Hadamard);
  // pure contraction: no index in all three
  BOOST_CHECK(compute_product_type(IndexList("i,j"), IndexList("j,k"),
                                   IndexList("i,k")) ==
              TensorProduct::Contraction);
  // batched contraction: b fused, j contracted, i/k free
  BOOST_CHECK(compute_product_type(IndexList("b,i,j"), IndexList("b,j,k"),
                                   IndexList("b,i,k")) ==
              TensorProduct::General);
  // the motivating TODO example: ijk * jkl -> ijl (j fused, k contracted)
  BOOST_CHECK(compute_product_type(IndexList("i,j,k"), IndexList("j,k,l"),
                                   IndexList("i,j,l")) ==
              TensorProduct::General);
  // Hadamard-reduction: args are permutations of each other but the target
  // drops an index => fused + contracted
  BOOST_CHECK(compute_product_type(IndexList("i,j"), IndexList("i,j"),
                                   IndexList("i")) == TensorProduct::General);
  // batched outer product: b fused, no contracted, i/k free
  BOOST_CHECK(compute_product_type(IndexList("b,i"), IndexList("b,k"),
                                   IndexList("b,i,k")) ==
              TensorProduct::General);
}

BOOST_AUTO_TEST_CASE(optimizer_canonical_layout) {
  // C("b,i,k") = A("b,i,j") * B("b,j,k")
  GeneralPermutationOptimizer opt(IndexList("b,i,k"), IndexList("b,i,j"),
                                  IndexList("b,j,k"));
  BOOST_CHECK(opt.op_type() == TensorProduct::General);
  BOOST_CHECK_EQUAL(opt.fused_indices().string(), "b");
  BOOST_CHECK_EQUAL(opt.contracted_indices().string(), "j");
  BOOST_CHECK_EQUAL(opt.left_external_indices().string(), "i");
  BOOST_CHECK_EQUAL(opt.right_external_indices().string(), "k");
  // canonical layouts: left (h, eA, c), right (h, c, eB), result (h, eA, eB)
  BOOST_CHECK_EQUAL(opt.target_left_indices().string(), "b,i,j");
  BOOST_CHECK_EQUAL(opt.target_right_indices().string(), "b,j,k");
  BOOST_CHECK_EQUAL(opt.target_result_indices().string(), "b,i,k");
  // both args already canonical
  BOOST_CHECK(opt.left_permtype() == PermutationType::identity);
  BOOST_CHECK(opt.right_permtype() == PermutationType::identity);
}

BOOST_AUTO_TEST_CASE(optimizer_noncanonical_args) {
  // C("k,b,i") = A("i,j,b") * B("k,j,b"): b fused, j contracted, i/k free;
  // neither argument is in canonical layout, so both get general permtypes
  GeneralPermutationOptimizer opt(IndexList("k,b,i"), IndexList("i,j,b"),
                                  IndexList("k,j,b"));
  BOOST_CHECK_EQUAL(opt.fused_indices().string(), "b");
  BOOST_CHECK_EQUAL(opt.contracted_indices().string(), "j");
  BOOST_CHECK_EQUAL(opt.left_external_indices().string(), "i");
  BOOST_CHECK_EQUAL(opt.right_external_indices().string(), "k");
  BOOST_CHECK_EQUAL(opt.target_left_indices().string(), "b,i,j");
  BOOST_CHECK_EQUAL(opt.target_right_indices().string(), "b,j,k");
  BOOST_CHECK_EQUAL(opt.target_result_indices().string(), "b,i,k");
  BOOST_CHECK(opt.left_permtype() == PermutationType::general);
  BOOST_CHECK(opt.right_permtype() == PermutationType::general);
}

BOOST_AUTO_TEST_CASE(optimizer_multiple_fused_target_order) {
  // two fused indices; class order follows the target
  GeneralPermutationOptimizer opt(IndexList("c,b,i,k"), IndexList("b,c,i,j"),
                                  IndexList("c,j,b,k"));
  BOOST_CHECK_EQUAL(opt.fused_indices().string(), "c,b");
  BOOST_CHECK_EQUAL(opt.target_left_indices().string(), "c,b,i,j");
  BOOST_CHECK_EQUAL(opt.target_right_indices().string(), "c,b,j,k");
  BOOST_CHECK_EQUAL(opt.target_result_indices().string(), "c,b,i,k");
}

BOOST_AUTO_TEST_CASE(optimizer_hadamard_reduction) {
  // "i,j" * "i,j" -> "i": i fused, j contracted, no externals
  GeneralPermutationOptimizer opt(IndexList("i"), IndexList("i,j"),
                                  IndexList("i,j"));
  BOOST_CHECK_EQUAL(opt.fused_indices().string(), "i");
  BOOST_CHECK_EQUAL(opt.contracted_indices().string(), "j");
  BOOST_CHECK(!opt.left_external_indices());
  BOOST_CHECK(!opt.right_external_indices());
  BOOST_CHECK_EQUAL(opt.target_left_indices().string(), "i,j");
  BOOST_CHECK_EQUAL(opt.target_right_indices().string(), "i,j");
  BOOST_CHECK_EQUAL(opt.target_result_indices().string(), "i");
}

BOOST_AUTO_TEST_CASE(optimizer_requires_target) {
  // bottom-up construction must throw: fused-vs-contracted undecidable
  BOOST_CHECK_THROW(
      GeneralPermutationOptimizer(IndexList("b,i,j"), IndexList("b,j,k")),
      TiledArray::Exception);
}

BOOST_AUTO_TEST_CASE(optimizer_rejects_implicit_reduction) {
  // left index "j" appears in neither right nor target
  BOOST_CHECK_THROW(GeneralPermutationOptimizer(
                        IndexList("b,i"), IndexList("b,i,j"), IndexList("b,i")),
                    TiledArray::Exception);
}

namespace {

/// sets the einsum legacy-subworld toggle within a scope; the previous value
/// is restored on scope exit (also on exceptions)
struct ScopedEinsumRoute {
  bool prev_;
  explicit ScopedEinsumRoute(const bool legacy)
      : prev_(TA::detail::einsum_legacy_subworld()) {
    TA::detail::einsum_legacy_subworld() = legacy;
  }
  ~ScopedEinsumRoute() { TA::detail::einsum_legacy_subworld() = prev_; }
};

/// forces the legacy sub-World einsum within a scope, so einsum-based
/// reference values remain an *independent* oracle for the expression route
/// (einsum itself routes general products through the expression layer now)
struct ForceLegacyEinsum : ScopedEinsumRoute {
  ForceLegacyEinsum() : ScopedEinsumRoute(true) {}
};

/// makes a dense array over \p tr filled with an index-dependent pattern
TA::TArrayD make_patterned_array(TA::World& world, const TA::TiledRange& tr,
                                 const double seed) {
  TA::TArrayD result(world, tr);
  for (auto it = result.begin(); it != result.end(); ++it) {
    auto tile =
        TA::TArrayD::value_type(result.trange().make_tile_range(it.index()));
    for (auto&& ix : tile.range()) {
      double v = seed;
      double scale = 1.0;
      for (auto x : ix) {
        v += scale * static_cast<double>(x + 1);
        scale *= 0.1;
      }
      tile[ix] = v;
    }
    *it = tile;
  }
  return result;
}

/// \return the Frobenius norm of `lhs - rhs`
double diff_norm(TA::TArrayD& lhs, TA::TArrayD& rhs, const std::string& annot) {
  TA::TArrayD diff;
  diff(annot) = lhs(annot) - rhs(annot);
  return diff(annot).norm().get();
}

}  // namespace

BOOST_AUTO_TEST_CASE(expression_general_product_dense) {
  // dense general products evaluate via the batched Summa; differential-test
  // against the einsum free function (the established implementation)
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  // C("b,i,k") = A("b,i,j") * B("b,j,k"), uneven multi-tile dimensions
  TA::TiledRange tr_a{{0, 2, 5}, {0, 3, 4}, {0, 2, 6, 7}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 5}, {0, 2, 6, 7}, {0, 4, 5}};  // b, j, k
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  TA::TArrayD c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k") = a("b,i,j") * b("b,j,k"));
  auto c_ref = TA::einsum(a("b,i,j"), b("b,j,k"), "b,i,k");
  BOOST_CHECK_SMALL(diff_norm(c, c_ref, "b,i,k"), 1e-10);

  // pure contraction and pure Hadamard still work
  TA::TArrayD d;
  BOOST_CHECK_NO_THROW(d("i,k") = a("b,i,j") * b("b,j,k"));
  TA::TArrayD e;
  BOOST_CHECK_NO_THROW(e("b,i,j") = a("b,i,j") * a("b,i,j"));
}

BOOST_AUTO_TEST_CASE(expression_general_product_dense_permuted_args) {
  // non-canonical argument layouts: the engine permutes the args into the
  // canonical (h, e_A, c) / (h, c, e_B) layouts
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  TA::TiledRange tr_a{{0, 3, 4}, {0, 2, 6, 7}, {0, 2, 5}};  // i, j, b
  TA::TiledRange tr_b{{0, 4, 5}, {0, 2, 6, 7}, {0, 2, 5}};  // k, j, b
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  TA::TArrayD c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k") = a("i,j,b") * b("k,j,b"));
  auto c_ref = TA::einsum(a("i,j,b"), b("k,j,b"), "b,i,k");
  BOOST_CHECK_SMALL(diff_norm(c, c_ref, "b,i,k"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_dense_batched_outer) {
  // batched outer product: fused + free, no contracted indices
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  TA::TiledRange tr_a{{0, 2, 5}, {0, 3, 4}};  // b, i
  TA::TiledRange tr_b{{0, 2, 5}, {0, 4, 5}};  // b, k
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  TA::TArrayD c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k") = a("b,i") * b("b,k"));
  auto c_ref = TA::einsum(a("b,i"), b("b,k"), "b,i,k");
  BOOST_CHECK_SMALL(diff_norm(c, c_ref, "b,i,k"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_noncanonical_root_target) {
  // a root-level general product with a NON-canonical target layout: the
  // product evaluates canonically (r1,p,q) and is re-permuted to the target
  // by the streaming unary eval
  auto& world = TA::get_default_world();
  TA::TiledRange tr_x{{0, 2, 4}, {0, 3, 5}};  // orbital x auxiliary
  auto x = make_patterned_array(world, tr_x, 1.0);

  TA::TArrayD w, i1, w_ref;
  BOOST_REQUIRE_NO_THROW(w("p,q,r1") = x("p,r1") * x("q,r1"));
  i1("r1,p,q") = x("p,r1") * x("q,r1");  // canonical evaluation
  w_ref("p,q,r1") = i1("r1,p,q");        // plain permute assignment

  BOOST_CHECK_SMALL(diff_norm(w, w_ref, "p,q,r1"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_inner_node_depth2) {
  // minimal inner-node case: a general product (fused r1) feeding a
  // contraction over r1 -- the general child evaluates canonically
  // (r1,p,q) and is re-permuted on the fly to the consumer's GEMM layout
  auto& world = TA::get_default_world();
  TA::TiledRange tr_x{{0, 2, 4}, {0, 3, 5}};  // orbital x auxiliary
  TA::TiledRange tr_z{{0, 3, 5}, {0, 3, 5}};  // auxiliary x auxiliary
  auto x = make_patterned_array(world, tr_x, 1.0);
  auto z = make_patterned_array(world, tr_z, 2.0);

  TA::TArrayD w;
  BOOST_REQUIRE_NO_THROW(w("p,q,r2") = (x("p,r1") * x("q,r1")) * z("r1,r2"));

  TA::TArrayD i1, w_ref;
  i1("r1,p,q") = x("p,r1") * x("q,r1");
  w_ref("p,q,r2") = i1("r1,p,q") * z("r1,r2");

  BOOST_CHECK_SMALL(diff_norm(w, w_ref, "p,q,r2"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_inner_node_thc) {
  // THC-style reconstruction in ONE expression:
  //   g("p,q,r,s") = X("p,r1") * X("q,r1") * Z("r1,r2") * X("r,r2") * X("s,r2")
  // r1 is fused in X("p,r1") * X("q,r1") but contracted downstream, so the
  // first product is a general product at an INNER node of the (left-deep)
  // expression tree. The top-down index-set deduction demands r1 of the X*X
  // node (its consumer carries it) and contracts it where Z meets that
  // subtree; higher up, r1 is dropped from the demand (consumed entirely
  // within). Verified against the same chain staged through explicit
  // intermediates (the pre-deduction recipe, itself differential-tested
  // against einsum in expression_general_product_thc_intermediates).
  auto& world = TA::get_default_world();
  TA::TiledRange tr_x{{0, 2, 4}, {0, 3, 5}};  // orbital x auxiliary
  TA::TiledRange tr_z{{0, 3, 5}, {0, 3, 5}};  // auxiliary x auxiliary
  auto x = make_patterned_array(world, tr_x, 1.0);
  auto z = make_patterned_array(world, tr_z, 2.0);

  TA::TArrayD g;
  BOOST_REQUIRE_NO_THROW(g("p,q,r,s") = x("p,r1") * x("q,r1") * z("r1,r2") *
                                        x("r,r2") * x("s,r2"));

  TA::TArrayD i1, i2, i3, g_ref;
  i1("r1,p,q") = x("p,r1") * x("q,r1");
  i2("p,q,r2") = i1("r1,p,q") * z("r1,r2");
  i3("r2,p,q,r") = i2("p,q,r2") * x("r,r2");
  g_ref("p,q,r,s") = i3("r2,p,q,r") * x("s,r2");

  BOOST_CHECK_SMALL(diff_norm(g, g_ref, "p,q,r,s"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_thc_intermediates) {
  // the supported way to evaluate the THC factorization today: materialize
  // each general product as the root of its own assignment (fused indices
  // leading in the result annotation), differential-tested against einsum
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_x{{0, 2, 4}, {0, 3, 5}};  // orbital x auxiliary
  TA::TiledRange tr_z{{0, 3, 5}, {0, 3, 5}};  // auxiliary x auxiliary
  auto x = make_patterned_array(world, tr_x, 1.0);
  auto z = make_patterned_array(world, tr_z, 2.0);

  TA::TArrayD i1, i2, i3, g;
  BOOST_REQUIRE_NO_THROW(i1("r1,p,q") = x("p,r1") * x("q,r1"));  // general
  BOOST_REQUIRE_NO_THROW(i2("p,q,r2") = i1("r1,p,q") * z("r1,r2"));
  BOOST_REQUIRE_NO_THROW(i3("r2,p,q,r") = i2("p,q,r2") * x("r,r2"));  // general
  BOOST_REQUIRE_NO_THROW(g("p,q,r,s") = i3("r2,p,q,r") * x("s,r2"));

  // oracle: the same chain with the general products evaluated by einsum
  auto i1_ref = TA::einsum(x("p,r1"), x("q,r1"), "r1,p,q");
  TA::TArrayD i2_ref;
  i2_ref("p,q,r2") = i1_ref("r1,p,q") * z("r1,r2");
  auto i3_ref = TA::einsum(i2_ref("p,q,r2"), x("r,r2"), "r2,p,q,r");
  TA::TArrayD g_ref;
  g_ref("p,q,r,s") = i3_ref("r2,p,q,r") * x("s,r2");

  BOOST_CHECK_SMALL(diff_norm(g, g_ref, "p,q,r,s"), 1e-10);
}

namespace {

/// makes a block-sparse array over \p tr with an index-dependent fill and a
/// deterministic block-sparsity pattern (every \p zero_stride -th tile zero)
TA::TSpArrayD make_patterned_sparse_array(TA::World& world,
                                          const TA::TiledRange& tr,
                                          const double seed,
                                          const std::size_t zero_stride) {
  // shape: unit norm except every zero_stride-th tile
  TA::Tensor<float> norms(tr.tiles_range(), 1.0f);
  for (std::size_t ord = 0; ord < norms.size(); ord += zero_stride)
    norms.data()[ord] = 0.0f;
  TA::SparseShape<float> shape(norms, tr);

  TA::TSpArrayD result(world, tr, shape);
  // iteration visits only local non-zero tiles
  for (auto it = result.begin(); it != result.end(); ++it) {
    auto tile =
        TA::TSpArrayD::value_type(result.trange().make_tile_range(it.index()));
    for (auto&& ix : tile.range()) {
      double v = seed;
      double scale = 1.0;
      for (auto x : ix) {
        v += scale * static_cast<double>(x + 1);
        scale *= 0.1;
      }
      tile[ix] = v;
    }
    *it = tile;
  }
  return result;
}

/// \return the Frobenius norm of `lhs - rhs` (block-sparse)
double diff_norm_sp(TA::TSpArrayD& lhs, TA::TSpArrayD& rhs,
                    const std::string& annot) {
  TA::TSpArrayD diff;
  diff(annot) = lhs(annot) - rhs(annot);
  return diff(annot).norm().get();
}

}  // namespace

BOOST_AUTO_TEST_CASE(expression_general_product_sparse) {
  // block-sparse general products: the result shape is computed slab-by-slab
  // (SparseShape::gemm_batched) and the batched Summa runs its sparse path
  // ((h,k)-keyed masks/groups); differential-test against einsum
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  TA::TiledRange tr_a{{0, 2, 5}, {0, 3, 4}, {0, 2, 6, 7}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 5}, {0, 2, 6, 7}, {0, 4, 5}};  // b, j, k
  auto a = make_patterned_sparse_array(world, tr_a, 1.0, 3);
  auto b = make_patterned_sparse_array(world, tr_b, 2.0, 4);

  TA::TSpArrayD c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k") = a("b,i,j") * b("b,j,k"));
  auto c_ref = TA::einsum(a("b,i,j"), b("b,j,k"), "b,i,k");
  BOOST_CHECK_SMALL(diff_norm_sp(c, c_ref, "b,i,k"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_sparse_batched_outer) {
  // block-sparse batched outer product (no contracted indices)
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  TA::TiledRange tr_a{{0, 2, 5}, {0, 3, 4}};  // b, i
  TA::TiledRange tr_b{{0, 2, 5}, {0, 4, 5}};  // b, k
  auto a = make_patterned_sparse_array(world, tr_a, 1.0, 3);
  auto b = make_patterned_sparse_array(world, tr_b, 2.0, 2);

  TA::TSpArrayD c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k") = a("b,i") * b("b,k"));
  auto c_ref = TA::einsum(a("b,i"), b("b,k"), "b,i,k");
  BOOST_CHECK_SMALL(diff_norm_sp(c, c_ref, "b,i,k"), 1e-10);
}

namespace {

using TArrayToT =
    TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::DensePolicy>;

/// makes a dense ToT array over \p tr with cells of extents \p inner_extents
/// filled with an index-dependent pattern
TArrayToT make_patterned_tot_array(
    TA::World& world, const TA::TiledRange& tr,
    const std::vector<std::size_t>& inner_extents, const double seed) {
  TArrayToT result(world, tr);
  for (auto it = result.begin(); it != result.end(); ++it) {
    auto outer_range = result.trange().make_tile_range(it.index());
    TA::Tensor<TA::Tensor<double>> tile(outer_range);
    for (auto&& oix : outer_range) {
      TA::Tensor<double> cell{TA::Range(inner_extents)};
      for (auto&& iix : cell.range()) {
        double v = seed;
        double scale = 1.0;
        for (auto x : oix) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        for (auto x : iix) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        cell[iix] = v;
      }
      tile[oix] = cell;
    }
    *it = tile;
  }
  return result;
}

/// \return the max abs elementwise difference between two congruent dense ToT
/// arrays (replicated check: every rank fetches every tile)
double tot_max_abs_diff(const TArrayToT& lhs, const TArrayToT& rhs) {
  double max_diff = 0.0;
  const auto n = lhs.trange().tiles_range().volume();
  for (std::size_t ord = 0; ord < n; ++ord) {
    auto lt = lhs.find(ord).get();
    auto rt = rhs.find(ord).get();
    for (std::size_t c = 0; c < lt.range().volume(); ++c) {
      const auto& lc = lt.data()[c];
      const auto& rc = rt.data()[c];
      for (std::size_t e = 0; e < lc.range().volume(); ++e)
        max_diff = std::max(max_diff, std::abs(lc.data()[e] - rc.data()[e]));
    }
  }
  return max_diff;
}

}  // namespace

BOOST_AUTO_TEST_CASE(expression_general_product_tot_inner_hadamard) {
  // ToT general product, inner Hadamard:
  //   C("b,i,k;m") = A("b,i,j;m") * B("b,j,k;m")
  // outer: b fused, j contracted, i/k free; inner: m fused
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 5}, {0, 3, 4}};  // b, j, k
  auto a = make_patterned_tot_array(world, tr_a, {3}, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {3}, 2.0);

  TArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k;m") = a("b,i,j;m") * b("b,j,k;m"));
  auto c_ref = TA::einsum(a("b,i,j;m"), b("b,j,k;m"), "b,i,k;m");
  BOOST_CHECK_SMALL(tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_tot_inner_contraction) {
  // ToT general product, inner contraction:
  //   C("b,i,k;m,n") = A("b,i,j;m,c") * B("b,j,k;c,n")
  // outer: b fused, j contracted, i/k free; inner: c contracted, m/n free
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 5}, {0, 3, 4}};  // b, j, k
  auto a = make_patterned_tot_array(world, tr_a, {3, 2}, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {2, 4}, 2.0);

  TArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k;m,n") = a("b,i,j;m,c") * b("b,j,k;c,n"));
  auto c_ref = TA::einsum(a("b,i,j;m,c"), b("b,j,k;c,n"), "b,i,k;m,n");
  BOOST_CHECK_SMALL(tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_tot_times_t) {
  // mixed ToT x T general product (inner Scale):
  //   C("b,i,k;m") = A("b,i,j;m") * B("b,j,k")
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 5}, {0, 3, 4}};  // b, j, k
  auto a = make_patterned_tot_array(world, tr_a, {3}, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  TArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k;m") = a("b,i,j;m") * b("b,j,k"));
  auto c_ref = TA::einsum(a("b,i,j;m"), b("b,j,k"), "b,i,k;m");
  BOOST_CHECK_SMALL(tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_t_times_tot) {
  // mixed T x ToT general product (inner Scale):
  //   C("b,i,k;m") = A("b,i,j") * B("b,j,k;m")
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 5}, {0, 3, 4}};  // b, j, k
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {3}, 2.0);

  TArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k;m") = a("b,i,j") * b("b,j,k;m"));
  auto c_ref = TA::einsum(a("b,i,j"), b("b,j,k;m"), "b,i,k;m");
  BOOST_CHECK_SMALL(tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_tot_inner_outer_product) {
  // the PNO-CC PPL building-block shape: ToT x ToT with an EMPTY right
  // outer-external set and an inner OUTER-product:
  //   C("b1,b2,K;x,y") = A("b1,b2,m,K;x") * B("b1,b2,m;y")
  // outer: b1,b2 fused, m contracted, K left-free, no right-free;
  // inner: x (left) (x) y (right), no inner contraction
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}, {0, 3, 6}};  // b1,b2,m,K
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};             // b1,b2,m
  auto a = make_patterned_tot_array(world, tr_a, {3}, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {2}, 2.0);

  TArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b1,b2,K;x,y") = a("b1,b2,m,K;x") * b("b1,b2,m;y"));
  auto c_ref = TA::einsum(a("b1,b2,m,K;x"), b("b1,b2,m;y"), "b1,b2,K;x,y");
  BOOST_CHECK_SMALL(tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_arena_inner_outer_product) {
  // same shape as expression_general_product_tot_inner_outer_product but
  // with arena-backed (ArenaTensor view) inner cells -- the PNO-CC CSV
  // representation; exercises the arena plans + strided ce+e kernel under
  // the batched Summa
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}, {0, 3, 6}};  // b1,b2,m,K
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};             // b1,b2,m
  constexpr long X = 3, Y = 2;

  auto fill = [](ArenaArr& arr, const TA::TiledRange& tr, const long n_in,
                 const double seed) {
    arr = ArenaArr(arr.world(), tr);
    arr.init_tiles([n_in, seed](const TA::Range& outer_range) {
      ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
          outer_range, 1,
          [n_in](std::size_t /*ord*/) { return TA::Range{n_in}; });
      std::size_t o = 0;
      for (auto&& oix : t.range()) {
        ArenaInner& cell = t.data()[o++];
        if (!cell) continue;
        for (long e = 0; e < n_in; ++e) {
          double v = seed + 0.01 * static_cast<double>(e + 1);
          double scale = 1.0;
          for (auto x : oix) {
            v += scale * static_cast<double>(x + 1);
            scale *= 0.1;
          }
          cell.data()[e] = v;
        }
      }
      return t;
    });
  };

  ArenaArr a(world, tr_a), b(world, tr_b);
  fill(a, tr_a, X, 1.0);
  fill(b, tr_b, Y, 2.0);
  world.gop.fence();

  ArenaArr c;
  BOOST_REQUIRE_NO_THROW(c("b1,b2,K;x,y") = a("b1,b2,m,K;x") * b("b1,b2,m;y"));
  auto c_ref = TA::einsum(a("b1,b2,m,K;x"), b("b1,b2,m;y"), "b1,b2,K;x,y");

  // elementwise comparison (replicated)
  double max_diff = 0.0;
  const auto n = c_ref.trange().tiles_range().volume();
  for (std::size_t ord = 0; ord < n; ++ord) {
    auto lt = c.find(ord).get();
    auto rt = c_ref.find(ord).get();
    BOOST_REQUIRE_EQUAL(lt.range().volume(), rt.range().volume());
    for (std::size_t cc = 0; cc < lt.range().volume(); ++cc) {
      const auto& lc = lt.data()[cc];
      const auto& rc = rt.data()[cc];
      BOOST_REQUIRE_EQUAL(lc.range().volume(), rc.range().volume());
      for (std::size_t e = 0; e < lc.range().volume(); ++e)
        max_diff = std::max(max_diff, std::abs(lc.data()[e] - rc.data()[e]));
    }
  }
  BOOST_CHECK_SMALL(max_diff, 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_tot_batched_outer) {
  // batched outer product (no contracted outer index) of ToTs with an inner
  // contraction, an interleaved target, and an empty right-external set:
  //   C("i2,i1,b1,b2;y") = A("b1,b2,i2,i1;x") * B("b1,b2;x,y")
  // (the canonical layout is (b1,b2,i2,i1); einsum reaches the interleaved
  // target by the final permutation assignment)
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}, {0, 2, 3}};
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 3}};
  auto a = make_patterned_tot_array(world, tr_a, {3}, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {3, 2}, 2.0);

  auto c = TA::einsum(a("b1,b2,i2,i1;x"), b("b1,b2;x,y"), "i2,i1,b1,b2;y");
  decltype(c) c_new;
  {
    ScopedEinsumRoute expression_route(false);
    c_new = TA::einsum(a("b1,b2,i2,i1;x"), b("b1,b2;x,y"), "i2,i1,b1,b2;y");
  }
  BOOST_CHECK_SMALL(tot_max_abs_diff(c_new, c), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_t_tot_batched_outer) {
  // batched outer product of a plain tensor with a ToT (inner Scale), with
  // an EMPTY left-external set:
  //   C("i2,i3,i1;x") = A("i3,i1") * B("i3,i1,i2;x")
  // (fused i3,i1; no contracted; right-external i2)
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}};
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {3}, 2.0);

  auto c = TA::einsum(a("i3,i1"), b("i3,i1,i2;x"), "i2,i3,i1;x");
  decltype(c) c_new;
  {
    ScopedEinsumRoute expression_route(false);
    c_new = TA::einsum(a("i3,i1"), b("i3,i1,i2;x"), "i2,i3,i1;x");
  }
  BOOST_CHECK_SMALL(tot_max_abs_diff(c_new, c), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_tot_t_nonleading_fused) {
  // reproduction of a CSV-CC mismatch shape: ToT x T (inner Scale) with a
  // NON-leading fused index and an interleaved target:
  //   C("i1,i4,K;x") = A("i4,i1,m;x") * B("m,i4,K")
  // (fused i4; contracted m; eA = i1; eB = K; canonical = i4,i1,K)
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // i4, i1, m
  TA::TiledRange tr_b{{0, 2, 5}, {0, 2, 4}, {0, 3, 6}};  // m, i4, K
  auto a = make_patterned_tot_array(world, tr_a, {3}, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  auto c_ref = TA::einsum(a("i4,i1,m;x"), b("m,i4,K"), "i1,i4,K;x");
  decltype(c_ref) c_new;
  {
    ScopedEinsumRoute expression_route(false);
    c_new = TA::einsum(a("i4,i1,m;x"), b("m,i4,K"), "i1,i4,K;x");
  }
  BOOST_CHECK_SMALL(tot_max_abs_diff(c_new, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(arena_outer_permute_assignment) {
  // hypothesis probe for the CSV mismatches: a pure OUTER permutation
  // assignment of an arena-backed (ArenaTensor view cell) ToT array
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaArr = TA::DistArray<ArenaOuter, TA::DensePolicy>;

  auto& world = TA::get_default_world();
  TA::TiledRange tr{{0, 2, 4}, {0, 2, 3}, {0, 3, 6}};  // b1, b2, k
  constexpr long X = 3;

  ArenaArr a(world, tr);
  a.init_tiles([](const TA::Range& outer_range) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        outer_range, 1, [](std::size_t) { return TA::Range{X}; });
    std::size_t o = 0;
    for (auto&& oix : t.range()) {
      ArenaInner& cell = t.data()[o++];
      if (!cell) continue;
      for (long e = 0; e < X; ++e) {
        double v = 0.01 * static_cast<double>(e + 1);
        double scale = 1.0;
        for (auto x : oix) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        cell.data()[e] = v;
      }
    }
    return t;
  });
  world.gop.fence();

  ArenaArr b;
  b("k,b1,b2;x") = a("b1,b2,k;x");

  // verify element-by-element against the source
  double max_diff = 0.0;
  const auto& trb = b.trange();
  for (std::size_t ord = 0; ord < trb.tiles_range().volume(); ++ord) {
    auto bt = b.find(ord).get();
    for (auto&& oix : bt.range()) {
      const auto& bc = bt[oix];
      // source element index: (b1,b2,k) from (k,b1,b2)
      std::array<long, 3> six{static_cast<long>(oix[1]),
                              static_cast<long>(oix[2]),
                              static_cast<long>(oix[0])};
      double v0 = 0.0;
      for (long e = 0; e < X; ++e) {
        double v = 0.01 * static_cast<double>(e + 1);
        double scale = 1.0;
        for (auto x : six) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        max_diff = std::max(max_diff, std::abs(bc.data()[e] - v));
        v0 = v;
      }
      (void)v0;
    }
  }
  BOOST_CHECK_SMALL(max_diff, 1e-12);
}

namespace {

using TSpArrayToT =
    TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>;

/// block-sparse ToT array with index-dependent fill, every zero_stride-th
/// tile zero
TSpArrayToT make_patterned_sparse_tot_array(
    TA::World& world, const TA::TiledRange& tr,
    const std::vector<std::size_t>& inner_extents, const double seed,
    const std::size_t zero_stride) {
  TA::Tensor<float> norms(tr.tiles_range(), 1.0f);
  for (std::size_t ord = 0; ord < norms.size(); ord += zero_stride)
    norms.data()[ord] = 0.0f;
  TA::SparseShape<float> shape(norms, tr);

  TSpArrayToT result(world, tr, shape);
  for (auto it = result.begin(); it != result.end(); ++it) {
    auto outer_range = result.trange().make_tile_range(it.index());
    TA::Tensor<TA::Tensor<double>> tile(outer_range);
    for (auto&& oix : outer_range) {
      TA::Tensor<double> cell{TA::Range(inner_extents)};
      for (auto&& iix : cell.range()) {
        double v = seed;
        double scale = 1.0;
        for (auto x : oix) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        for (auto x : iix) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        cell[iix] = v;
      }
      tile[oix] = cell;
    }
    *it = tile;
  }
  return result;
}

/// max abs elementwise diff of two congruent sparse ToT arrays
double sparse_tot_max_abs_diff(const TSpArrayToT& lhs, const TSpArrayToT& rhs) {
  double max_diff = 0.0;
  const auto n = lhs.trange().tiles_range().volume();
  for (std::size_t ord = 0; ord < n; ++ord) {
    const bool lz = lhs.is_zero(ord);
    const bool rz = rhs.is_zero(ord);
    if (lz && rz) continue;
    if (lz != rz) {
      // one is a zero tile: the other must be numerically zero
      auto t = (lz ? rhs : lhs).find(ord).get();
      for (std::size_t c = 0; c < t.range().volume(); ++c) {
        const auto& cell = t.data()[c];
        if (cell.empty()) continue;
        for (std::size_t e = 0; e < cell.range().volume(); ++e)
          max_diff = std::max(max_diff, std::abs(cell.data()[e]));
      }
      continue;
    }
    auto lt = lhs.find(ord).get();
    auto rt = rhs.find(ord).get();
    for (std::size_t c = 0; c < lt.range().volume(); ++c) {
      const auto& lc = lt.data()[c];
      const auto& rc = rt.data()[c];
      if (lc.empty() && rc.empty()) continue;
      for (std::size_t e = 0; e < lc.range().volume(); ++e)
        max_diff = std::max(max_diff, std::abs(lc.data()[e] - rc.data()[e]));
    }
  }
  return max_diff;
}

}  // namespace

BOOST_AUTO_TEST_CASE(expression_general_product_sparse_tot) {
  // SPARSE-policy ToT general product -- the CSV-CC case (all other ToT
  // tests are dense): exercises the batched Summa's sparse step iteration
  // ((h,k) skipping) and sparse reducer gating with ToT tiles
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 5}, {0, 3, 4}};  // b, j, k
  auto a = make_patterned_sparse_tot_array(world, tr_a, {3}, 1.0, 3);
  auto b = make_patterned_sparse_tot_array(world, tr_b, {2}, 2.0, 4);

  TSpArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k;x,y") = a("b,i,j;x") * b("b,j,k;y"));
  auto c_ref = TA::einsum(a("b,i,j;x"), b("b,j,k;y"), "b,i,k;x,y");
  BOOST_CHECK_SMALL(sparse_tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_sparse_tot_skipped_steps) {
  // STRUCTURED block-sparsity that zeroes entire (slab, k) panels, forcing
  // the batched Summa's sparse step iteration to SKIP steps (and discard
  // the skipped panels' tiles) -- the suspected CSV-CC failure mode
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 4, 6}, {0, 2, 3}, {0, 2, 5, 7}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4, 6}, {0, 2, 5, 7}, {0, 3, 4}};  // b, j, k

  // left: zero the whole (b=0, j=1) and (b=2, j=0) panels (all i)
  TA::Tensor<float> norms_a(tr_a.tiles_range(), 1.0f);
  for (std::size_t i = 0; i < 2; ++i) {
    norms_a[std::array<std::size_t, 3>{0, i, 1}] = 0.0f;
    norms_a[std::array<std::size_t, 3>{2, i, 0}] = 0.0f;
  }
  // right: zero the whole (b=1, j=2) panel (all k) and (b=0, j=1)
  TA::Tensor<float> norms_b(tr_b.tiles_range(), 1.0f);
  for (std::size_t k = 0; k < 2; ++k) {
    norms_b[std::array<std::size_t, 3>{1, 2, k}] = 0.0f;
    norms_b[std::array<std::size_t, 3>{0, 1, k}] = 0.0f;
  }
  TA::SparseShape<float> shape_a(norms_a, tr_a);
  TA::SparseShape<float> shape_b(norms_b, tr_b);

  auto fill = [](TSpArrayToT& arr, const std::size_t n_in, const double seed) {
    for (auto it = arr.begin(); it != arr.end(); ++it) {
      auto outer_range = arr.trange().make_tile_range(it.index());
      TA::Tensor<TA::Tensor<double>> tile(outer_range);
      for (auto&& oix : outer_range) {
        TA::Tensor<double> cell{TA::Range(std::vector<std::size_t>{n_in})};
        for (auto&& iix : cell.range()) {
          double v = seed;
          double scale = 1.0;
          for (auto x : oix) {
            v += scale * static_cast<double>(x + 1);
            scale *= 0.1;
          }
          for (auto x : iix) v += 0.001 * static_cast<double>(x + 1);
          cell[iix] = v;
        }
        tile[oix] = cell;
      }
      *it = tile;
    }
  };

  TSpArrayToT a(world, tr_a, shape_a);
  TSpArrayToT b(world, tr_b, shape_b);
  fill(a, 3, 1.0);
  fill(b, 2, 2.0);
  world.gop.fence();

  // run the expression route TWICE to also detect non-determinism
  TSpArrayToT c1, c2;
  BOOST_REQUIRE_NO_THROW(c1("b,i,k;x,y") = a("b,i,j;x") * b("b,j,k;y"));
  BOOST_REQUIRE_NO_THROW(c2("b,i,k;x,y") = a("b,i,j;x") * b("b,j,k;y"));
  auto c_ref = TA::einsum(a("b,i,j;x"), b("b,j,k;y"), "b,i,k;x,y");
  const double d_rep = sparse_tot_max_abs_diff(c1, c2);
  const double d_ref = sparse_tot_max_abs_diff(c1, c_ref);
  BOOST_CHECK_SMALL(d_rep, 1e-12);
  BOOST_CHECK_SMALL(d_ref, 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_csv_like) {
  // kitchen-sink reproduction of the CSV-CC mismatch shape
  //   (i2,i1,m;a) * (m,i2,K) -> (i1,i2,K;a):
  // arena (view) inner cells, SparsePolicy, inner extent VARYING with the
  // fused+external outer pair, EMPTY (screened) cells, non-leading fused
  // index, interleaved target
  using ArenaInner = TA::ArenaTensor<double, TA::Range>;
  using ArenaOuter = TA::Tensor<ArenaInner>;
  using ArenaSpArr = TA::DistArray<ArenaOuter, TA::SparsePolicy>;
  using PlainSpArr = TA::TSpArrayD;

  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent

  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5, 7}};  // i2, i1, m
  TA::TiledRange tr_b{{0, 2, 5, 7}, {0, 2, 4}, {0, 3, 6}};  // m, i2, K

  // left: arena ToT, sparse (zero the (i2=1, i1=0, m=1) tile); inner extent
  // depends on (i2 + i1); the (i2+i1+m) % 5 == 0 cells are empty (screened)
  TA::Tensor<float> norms_a(tr_a.tiles_range(), 1.0f);
  norms_a[std::array<std::size_t, 3>{1, 0, 1}] = 0.0f;
  ArenaSpArr a(world, tr_a, TA::SparseShape<float>(norms_a, tr_a));
  a.init_tiles([](const TA::Range& outer_range) {
    auto lo = outer_range.lobound_data();
    auto cell_range = [&outer_range, lo](std::size_t ord) {
      auto oix = outer_range.idx(ord);
      const long ext = 2 + (oix[0] + oix[1]) % 3;  // varies with pair
      if ((oix[0] + oix[1] + oix[2]) % 5 == 0)     // screened cells
        return TA::Range{};
      return TA::Range{ext};
    };
    ArenaOuter t =
        TA::detail::arena_outer_init<ArenaOuter>(outer_range, 1, cell_range);
    std::size_t o = 0;
    for (auto&& oix : t.range()) {
      ArenaInner& cell = t.data()[o++];
      if (!cell) continue;
      for (std::size_t e = 0; e < cell.range().volume(); ++e) {
        double v = 1.0 + 0.01 * static_cast<double>(e + 1);
        double scale = 1.0;
        for (auto x : oix) {
          v += scale * static_cast<double>(x + 1);
          scale *= 0.1;
        }
        cell.data()[e] = v;
      }
    }
    (void)lo;
    return t;
  });

  // right: plain sparse (zero one m-panel tile)
  TA::Tensor<float> norms_b(tr_b.tiles_range(), 1.0f);
  norms_b[std::array<std::size_t, 3>{1, 1, 0}] = 0.0f;
  PlainSpArr b(world, tr_b, TA::SparseShape<float>(norms_b, tr_b));
  for (auto it = b.begin(); it != b.end(); ++it) {
    auto tile = PlainSpArr::value_type(b.trange().make_tile_range(it.index()));
    for (auto&& ix : tile.range()) {
      double v = 2.0;
      double scale = 1.0;
      for (auto x : ix) {
        v += scale * static_cast<double>(x + 1);
        scale *= 0.1;
      }
      tile[ix] = v;
    }
    *it = tile;
  }
  world.gop.fence();

  // expression route twice (determinism) + legacy einsum oracle
  // mirror the CSV path exactly: through einsum (which canonicalizes the
  // target and permutes at the end), new route vs the legacy oracle
  auto c_ref = TA::einsum(a("i2,i1,m;a"), b("m,i2,K"), "i1,i2,K;a");
  ArenaSpArr c1, c2;
  {
    ScopedEinsumRoute expression_route(false);
    try {
      c1 = TA::einsum(a("i2,i1,m;a"), b("m,i2,K"), "i1,i2,K;a");
      c2 = TA::einsum(a("i2,i1,m;a"), b("m,i2,K"), "i1,i2,K;a");
    } catch (std::exception& ex) {
      std::cerr << "EXPRESSION ROUTE THREW: " << ex.what() << std::endl;
      throw;
    }
  }

  // BISECT: same shape, CANONICAL target, direct expression (no einsum
  // wrapper, no final permutation) vs the legacy oracle
  {
    ArenaSpArr c_canon_expr;
    c_canon_expr("i2,i1,K;a") = a("i2,i1,m;a") * b("m,i2,K");
    auto c_canon_ref = TA::einsum(a("i2,i1,m;a"), b("m,i2,K"), "i2,i1,K;a");
    double md = 0.0;
    const auto n = c_canon_ref.trange().tiles_range().volume();
    for (std::size_t ord = 0; ord < n; ++ord) {
      if (c_canon_expr.is_zero(ord) && c_canon_ref.is_zero(ord)) continue;
      if (c_canon_expr.is_zero(ord) != c_canon_ref.is_zero(ord)) {
        md = std::max(md, 1e10);  // zero-pattern mismatch marker
        continue;
      }
      auto lt = c_canon_expr.find(ord).get();
      auto rt = c_canon_ref.find(ord).get();
      for (std::size_t cc = 0; cc < lt.range().volume(); ++cc) {
        const auto& lc = lt.data()[cc];
        const auto& rc = rt.data()[cc];
        if (!lc && !rc) continue;
        const std::size_t ne = lc ? lc.range().volume() : rc.range().volume();
        for (std::size_t e = 0; e < ne; ++e) {
          const double lv = lc ? lc.data()[e] : 0.0;
          const double rv = rc ? rc.data()[e] : 0.0;
          md = std::max(md, std::abs(lv - rv));
        }
      }
    }
    std::cerr << "CANONICAL-TARGET DIRECT-EXPR max_diff = " << md << std::endl;
    BOOST_CHECK_SMALL(md, 1e-10);
  }

  auto max_diff = [](const ArenaSpArr& lhs, const ArenaSpArr& rhs) {
    double md = 0.0;
    const auto n = lhs.trange().tiles_range().volume();
    for (std::size_t ord = 0; ord < n; ++ord) {
      const bool lz = lhs.is_zero(ord);
      const bool rz = rhs.is_zero(ord);
      if (lz && rz) continue;
      if (lz != rz) {
        auto t = (lz ? rhs : lhs).find(ord).get();
        for (std::size_t c = 0; c < t.range().volume(); ++c) {
          const auto& cell = t.data()[c];
          if (!cell) continue;
          for (std::size_t e = 0; e < cell.range().volume(); ++e)
            md = std::max(md, std::abs(cell.data()[e]));
        }
        continue;
      }
      auto lt = lhs.find(ord).get();
      auto rt = rhs.find(ord).get();
      for (std::size_t c = 0; c < lt.range().volume(); ++c) {
        const auto& lc = lt.data()[c];
        const auto& rc = rt.data()[c];
        if (!lc && !rc) continue;
        if (bool(lc) != bool(rc)) {
          const auto& nz = lc ? lc : rc;
          for (std::size_t e = 0; e < nz.range().volume(); ++e)
            md = std::max(md, std::abs(nz.data()[e]));
          continue;
        }
        for (std::size_t e = 0; e < lc.range().volume(); ++e)
          md = std::max(md, std::abs(lc.data()[e] - rc.data()[e]));
      }
    }
    return md;
  };

  BOOST_CHECK_SMALL(max_diff(c1, c2), 1e-12);
  BOOST_CHECK_SMALL(max_diff(c1, c_ref), 1e-10);
}

BOOST_AUTO_TEST_CASE(einsum_expression_route_matches_legacy) {
  // einsum routes general products through the expression layer by default;
  // differential-check against the retained legacy sub-World path, with an
  // interleaved (non-canonical) target reached by the final permutation
  // assignment
  auto& world = TA::get_default_world();
  ForceLegacyEinsum legacy_oracle;  // keep the einsum reference independent
  TA::TiledRange tr_a{{0, 2, 5}, {0, 3, 4}, {0, 2, 6, 7}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 5}, {0, 2, 6, 7}, {0, 4, 5}};  // b, j, k
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  auto c_legacy = TA::einsum(a("b,i,j"), b("b,j,k"), "i,b,k");
  decltype(c_legacy) c_new;
  {
    ScopedEinsumRoute expression_route(false);
    c_new = TA::einsum(a("b,i,j"), b("b,j,k"), "i,b,k");
  }
  BOOST_CHECK_SMALL(diff_norm(c_new, c_legacy, "i,b,k"), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
