/// Unit tests for general-product (fused + contracted + free indices)
/// classification and layout in the expression layer.

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

BOOST_AUTO_TEST_CASE(expression_general_product_gated) {
  // the expression layer now *classifies* general products correctly and
  // reports that evaluation is not yet implemented (instead of misrouting
  // to the contraction or Hadamard machinery)
  auto& world = TA::get_default_world();
  TA::TiledRange tr{{0, 2, 4}, {0, 2, 4}, {0, 2, 4}};
  TA::TArrayD a(world, tr);
  TA::TArrayD b(world, tr);
  a.fill(1.0);
  b.fill(1.0);
  TA::TArrayD c;
  BOOST_CHECK_THROW(c("b,i,k") = a("b,i,j") * b("b,j,k"),
                    TiledArray::Exception);
  // pure contraction and pure Hadamard still work
  BOOST_CHECK_NO_THROW(c("i,k") = a("b,i,j") * b("b,j,k"));
  BOOST_CHECK_NO_THROW(c("b,i,j") = a("b,i,j") * b("b,i,j"));
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

BOOST_AUTO_TEST_SUITE_END()
