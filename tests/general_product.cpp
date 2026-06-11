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

  TA::TiledRange tr_a{{0, 2, 5}, {0, 3, 4}};  // b, i
  TA::TiledRange tr_b{{0, 2, 5}, {0, 4, 5}};  // b, k
  auto a = make_patterned_array(world, tr_a, 1.0);
  auto b = make_patterned_array(world, tr_b, 2.0);

  TA::TArrayD c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k") = a("b,i") * b("b,k"));
  auto c_ref = TA::einsum(a("b,i"), b("b,k"), "b,i,k");
  BOOST_CHECK_SMALL(diff_norm(c, c_ref, "b,i,k"), 1e-10);
}

BOOST_AUTO_TEST_CASE(expression_general_product_inner_node_gated) {
  // THC-style reconstruction:
  //   g("p,q,r,s") = X("p,r1") * X("q,r1") * Z("r1,r2") * X("r,r2") * X("s,r2")
  // r1 is fused in X("p,r1") * X("q,r1") but contracted downstream. The
  // first product is an INNER node of the expression tree, where the role of
  // r1 cannot be deduced bottom-up (the target reaches only the root);
  // resolving this requires top-down index-set deduction (deferred). Until
  // then: an informative error, not garbage (bottom-up, X*X would contract
  // r1, orphaning the r1 of Z).
  auto& world = TA::get_default_world();
  TA::TiledRange tr_x{{0, 2, 4}, {0, 3, 5}};  // orbital x auxiliary
  TA::TiledRange tr_z{{0, 3, 5}, {0, 3, 5}};  // auxiliary x auxiliary
  TA::TArrayD x(world, tr_x);
  TA::TArrayD z(world, tr_z);
  x.fill(1.0);
  z.fill(1.0);
  TA::TArrayD g;
  BOOST_CHECK_THROW(
      g("p,q,r,s") = x("p,r1") * x("q,r1") * z("r1,r2") * x("r,r2") * x("s,r2"),
      TiledArray::Exception);
}

BOOST_AUTO_TEST_CASE(expression_general_product_thc_intermediates) {
  // the supported way to evaluate the THC factorization today: materialize
  // each general product as the root of its own assignment (fused indices
  // leading in the result annotation), differential-tested against einsum
  auto& world = TA::get_default_world();
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
  TA::TiledRange tr_a{{0, 2, 4}, {0, 2, 3}, {0, 2, 5}};  // b, i, j
  TA::TiledRange tr_b{{0, 2, 4}, {0, 2, 5}, {0, 3, 4}};  // b, j, k
  auto a = make_patterned_tot_array(world, tr_a, {3, 2}, 1.0);
  auto b = make_patterned_tot_array(world, tr_b, {2, 4}, 2.0);

  TArrayToT c;
  BOOST_REQUIRE_NO_THROW(c("b,i,k;m,n") = a("b,i,j;m,c") * b("b,j,k;c,n"));
  auto c_ref = TA::einsum(a("b,i,j;m,c"), b("b,j,k;c,n"), "b,i,k;m,n");
  BOOST_CHECK_SMALL(tot_max_abs_diff(c, c_ref), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
