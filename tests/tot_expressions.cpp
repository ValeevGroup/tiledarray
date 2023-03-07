#include "tot_array_fixture.h"

template <typename Tensor, typename ElementGenerator>
std::enable_if_t<TiledArray::detail::is_btas_tensor_v<Tensor>, Tensor>
make_tensor(typename Tensor::range_type&& range, ElementGenerator&& op) {
  Tensor result(std::move(range));
  result.generate(std::forward<ElementGenerator>(op));
  return result;
}

template <typename Tensor, typename ElementGenerator>
std::enable_if_t<TiledArray::detail::is_ta_tensor_v<Tensor>, Tensor>
make_tensor(typename Tensor::range_type&& range, ElementGenerator&& op) {
  Tensor result(std::move(range));
  std::generate(result.begin(), result.end(),
                std::forward<ElementGenerator>(op));
  return result;
}

//------------------------------------------------------------------------------
//                            Permutations
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_SUITE(tot_permutations, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(no_perm, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& in_rank = std::get<1>(tr_t);
    auto& t = std::get<2>(tr_t);

    std::string out_idx = t.tiles_range().rank() == 1 ? "i" : "i, j";
    std::string in_idx = in_rank == 1 ? "k" : "k, l";
    std::string idx = out_idx + ";" + in_idx;

    tensor_type<TestParam> result;
    result(idx) = t(idx);
    BOOST_TEST(are_equal(result, t));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(permute_outer, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& in_rank = std::get<1>(tr_t);
    auto& t = std::get<2>(tr_t);

    if (t.tiles_range().rank() == 1) continue;

    std::string rhs_out_idx = "i, j";
    std::string lhs_out_idx = "j, i";
    std::string in_idx = in_rank == 1 ? "k" : "k, l";
    std::string rhs_idx = rhs_out_idx + ";" + in_idx;
    std::string lhs_idx = lhs_out_idx + ";" + in_idx;
    tensor_type<TestParam> result;
    result(lhs_idx) = t(rhs_idx);

    for (auto tile_idx : t.tiles_range()) {
      auto rtile = t.find(tile_idx).get();
      auto ltile = result.find({tile_idx[1], tile_idx[0]}).get();
      for (auto outer_idx : ltile.range()) {
        auto inner_range = ltile(outer_idx).range();
        auto outer_idx_t = {outer_idx[1], outer_idx[0]};
        bool same_inner_range = inner_range == rtile(outer_idx_t).range();
        BOOST_CHECK(same_inner_range);
        for (auto inner_idx : inner_range) {
          const auto lelem = ltile(outer_idx)(inner_idx);
          const auto relem = rtile(outer_idx_t)(inner_idx);
          BOOST_CHECK_EQUAL(lelem, relem);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(permute_inner, TestParam, test_params) {
  for (auto tr_t : run_all<TestParam>()) {
    auto& in_rank = std::get<1>(tr_t);
    auto& t = std::get<2>(tr_t);

    if (in_rank == 1) continue;

    std::string rhs_in_idx = "i, j";
    std::string lhs_in_idx = "j, i";
    std::string out_idx = t.tiles_range().rank() == 1 ? "k" : "k, l";
    std::string rhs_idx = out_idx + ";" + rhs_in_idx;
    std::string lhs_idx = out_idx + ";" + lhs_in_idx;
    tensor_type<TestParam> result;
    result(lhs_idx) = t(rhs_idx);

    for (auto tile_idx : t.tiles_range()) {
      auto rtile = t.find(tile_idx).get();
      auto ltile = result.find(tile_idx).get();
      bool same_outer_range = ltile.range() == rtile.range();
      BOOST_CHECK(same_outer_range);
      for (auto outer_idx : ltile.range()) {
        auto inner_range = ltile(outer_idx).range();
        for (auto inner_idx : inner_range) {
          const auto lelem = ltile(outer_idx)(inner_idx);
          const auto inner_idx_t = {inner_idx[1], inner_idx[0]};
          const auto relem = rtile(outer_idx)(inner_idx_t);
          BOOST_CHECK_EQUAL(lelem, relem);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
//                           Addition
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_SUITE(tot_addition, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(vov, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2};
  lhs_0({0}) = 0;
  lhs_0({1}) = 1;
  inner_type lhs_1{r3};
  lhs_1({0}) = 1;
  lhs_1({1}) = 2;
  lhs_1({2}) = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2};
  rhs_0({0}) = 1;
  rhs_0({1}) = 2;
  inner_type rhs_1{r3};
  rhs_1({0}) = 2;
  rhs_1({1}) = 3;
  rhs_1({2}) = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2};
  c_0({0}) = 1;
  c_0({1}) = 3;
  inner_type c_1{r3};
  c_1({0}) = 3;
  c_1({1}) = 5;
  c_1({2}) = 7;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j") = lhs("i;j") + rhs("i;j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 1;
  c_0(0, 1) = 3;
  c_0(0, 2) = 5;
  c_0(1, 0) = 3;
  c_0(1, 1) = 5;
  c_0(1, 2) = 7;
  inner_type c_1{r33};
  c_1(0, 0) = 3;
  c_1(0, 1) = 5;
  c_1(0, 2) = 7;
  c_1(1, 0) = 5;
  c_1(1, 1) = 7;
  c_1(1, 2) = 9;
  c_1(2, 0) = 7;
  c_1(2, 1) = 9;
  c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j,k") = lhs("i;j,k") + rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{range_type{3, 2}};
  c_0(0, 0) = 1;
  c_0(0, 1) = 3;
  c_0(1, 0) = 3;
  c_0(1, 1) = 5;
  c_0(2, 0) = 5;
  c_0(2, 1) = 7;
  inner_type c_1{r33};
  c_1(0, 0) = 3;
  c_1(0, 1) = 5;
  c_1(0, 2) = 7;
  c_1(1, 0) = 5;
  c_1(1, 1) = 7;
  c_1(1, 2) = 9;
  c_1(2, 0) = 7;
  c_1(2, 1) = 9;
  c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;k,j") = lhs("i;j,k") + rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(2, 0) = 2;
  lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 1;
  c_0(0, 1) = 3;
  c_0(0, 2) = 5;
  c_0(1, 0) = 3;
  c_0(1, 1) = 5;
  c_0(1, 2) = 7;
  inner_type c_1{r33};
  c_1(0, 0) = 3;
  c_1(0, 1) = 5;
  c_1(0, 2) = 7;
  c_1(1, 0) = 5;
  c_1(1, 1) = 7;
  c_1(1, 2) = 9;
  c_1(2, 0) = 7;
  c_1(2, 1) = 9;
  c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;k,j") + rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_rhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{range_type{3, 2}};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(2, 0) = 3;
  rhs_0(2, 1) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 1;
  c_0(0, 1) = 3;
  c_0(0, 2) = 5;
  c_0(1, 0) = 3;
  c_0(1, 1) = 5;
  c_0(1, 2) = 7;
  inner_type c_1{r33};
  c_1(0, 0) = 3;
  c_1(0, 1) = 5;
  c_1(0, 2) = 7;
  c_1(1, 0) = 5;
  c_1(1, 1) = 7;
  c_1(1, 2) = 9;
  c_1(2, 0) = 7;
  c_1(2, 1) = 9;
  c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;j,k") + rhs("i;k,j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_and_rhs_transpose, TestParam,
                              test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(2, 0) = 2;
  lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{range_type{3, 2}};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(2, 0) = 3;
  rhs_0(2, 1) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 1;
  c_0(0, 1) = 3;
  c_0(0, 2) = 5;
  c_0(1, 0) = 3;
  c_0(1, 1) = 5;
  c_0(1, 2) = 7;
  inner_type c_1{r33};
  c_1(0, 0) = 3;
  c_1(0, 1) = 5;
  c_1(0, 2) = 7;
  c_1(1, 0) = 5;
  c_1(1, 1) = 7;
  c_1(1, 2) = 9;
  c_1(2, 0) = 7;
  c_1(2, 1) = 9;
  c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;k,j") + rhs("i;k,j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = 1;
  c_00({1}) = 3;
  inner_type c_01(range_type{2});
  c_01({0}) = 3;
  c_01({1}) = 5;
  inner_type c_02(range_type{2});
  c_02({0}) = 5;
  c_02({1}) = 7;
  inner_type c_10(range_type{3});
  c_10({0}) = 3;
  c_10({1}) = 5;
  c_10({2}) = 7;
  inner_type c_11(range_type{3});
  c_11({0}) = 5;
  c_11({1}) = 7;
  c_11({2}) = 9;
  inner_type c_12(range_type{3});
  c_12({0}) = 7;
  c_12({1}) = 9;
  c_12({2}) = 11;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k") = lhs("i,j;k") + rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = 1;
  c_00({1}) = 3;
  inner_type c_01(range_type{2});
  c_01({0}) = 3;
  c_01({1}) = 5;
  inner_type c_02(range_type{2});
  c_02({0}) = 5;
  c_02({1}) = 7;
  inner_type c_10(range_type{3});
  c_10({0}) = 3;
  c_10({1}) = 5;
  c_10({2}) = 7;
  inner_type c_11(range_type{3});
  c_11({0}) = 5;
  c_11({1}) = 7;
  c_11({2}) = 9;
  inner_type c_12(range_type{3});
  c_12({0}) = 7;
  c_12({1}) = 9;
  c_12({2}) = 11;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("j,i;k") = lhs("i,j;k") + rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 1;
  c_00(0, 1) = 3;
  c_00(1, 0) = 3;
  c_00(1, 1) = 5;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 3;
  c_01(0, 1) = 5;
  c_01(0, 2) = 7;
  c_01(1, 0) = 5;
  c_01(1, 1) = 7;
  c_01(1, 2) = 9;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 5;
  c_02(0, 1) = 7;
  c_02(0, 2) = 9;
  c_02(0, 3) = 11;
  c_02(1, 0) = 7;
  c_02(1, 1) = 9;
  c_02(1, 2) = 11;
  c_02(1, 3) = 13;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 3;
  c_10(0, 1) = 5;
  c_10(1, 0) = 5;
  c_10(1, 1) = 7;
  c_10(2, 0) = 7;
  c_10(2, 1) = 9;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 5;
  c_11(0, 1) = 7;
  c_11(0, 2) = 9;
  c_11(1, 0) = 7;
  c_11(1, 1) = 9;
  c_11(1, 2) = 11;
  c_11(2, 0) = 9;
  c_11(2, 1) = 11;
  c_11(2, 2) = 13;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 7;
  c_12(0, 1) = 9;
  c_12(0, 2) = 11;
  c_12(0, 3) = 13;
  c_12(1, 0) = 9;
  c_12(1, 1) = 11;
  c_12(1, 2) = 13;
  c_12(1, 3) = 15;
  c_12(2, 0) = 11;
  c_12(2, 1) = 13;
  c_12(2, 2) = 15;
  c_12(2, 3) = 17;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;k,l") + rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 1;
  c_00(0, 1) = 3;
  c_00(1, 0) = 3;
  c_00(1, 1) = 5;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 3;
  c_01(0, 1) = 5;
  c_01(0, 2) = 7;
  c_01(1, 0) = 5;
  c_01(1, 1) = 7;
  c_01(1, 2) = 9;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 5;
  c_02(0, 1) = 7;
  c_02(0, 2) = 9;
  c_02(0, 3) = 11;
  c_02(1, 0) = 7;
  c_02(1, 1) = 9;
  c_02(1, 2) = 11;
  c_02(1, 3) = 13;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 3;
  c_10(0, 1) = 5;
  c_10(1, 0) = 5;
  c_10(1, 1) = 7;
  c_10(2, 0) = 7;
  c_10(2, 1) = 9;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 5;
  c_11(0, 1) = 7;
  c_11(0, 2) = 9;
  c_11(1, 0) = 7;
  c_11(1, 1) = 9;
  c_11(1, 2) = 11;
  c_11(2, 0) = 9;
  c_11(2, 1) = 11;
  c_11(2, 2) = 13;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 7;
  c_12(0, 1) = 9;
  c_12(0, 2) = 11;
  c_12(0, 3) = 13;
  c_12(1, 0) = 9;
  c_12(1, 1) = 11;
  c_12(1, 2) = 13;
  c_12(1, 3) = 15;
  c_12(2, 0) = 11;
  c_12(2, 1) = 13;
  c_12(2, 2) = 15;
  c_12(2, 3) = 17;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("j,i;k,l") = lhs("i,j;k,l") + rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_lhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{3, 2});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(2, 0) = 3;
  lhs_01(2, 1) = 4;
  inner_type lhs_02(range_type{4, 2});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(2, 0) = 4;
  lhs_02(2, 1) = 5;
  lhs_02(3, 0) = 5;
  lhs_02(3, 1) = 6;
  inner_type lhs_10(range_type{2, 3});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(0, 2) = 3;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(1, 2) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{4, 3});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(3, 0) = 6;
  lhs_12(3, 1) = 7;
  lhs_12(3, 2) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 1;
  c_00(0, 1) = 3;
  c_00(1, 0) = 3;
  c_00(1, 1) = 5;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 3;
  c_01(0, 1) = 5;
  c_01(0, 2) = 7;
  c_01(1, 0) = 5;
  c_01(1, 1) = 7;
  c_01(1, 2) = 9;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 5;
  c_02(0, 1) = 7;
  c_02(0, 2) = 9;
  c_02(0, 3) = 11;
  c_02(1, 0) = 7;
  c_02(1, 1) = 9;
  c_02(1, 2) = 11;
  c_02(1, 3) = 13;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 3;
  c_10(0, 1) = 5;
  c_10(1, 0) = 5;
  c_10(1, 1) = 7;
  c_10(2, 0) = 7;
  c_10(2, 1) = 9;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 5;
  c_11(0, 1) = 7;
  c_11(0, 2) = 9;
  c_11(1, 0) = 7;
  c_11(1, 1) = 9;
  c_11(1, 2) = 11;
  c_11(2, 0) = 9;
  c_11(2, 1) = 11;
  c_11(2, 2) = 13;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 7;
  c_12(0, 1) = 9;
  c_12(0, 2) = 11;
  c_12(0, 3) = 13;
  c_12(1, 0) = 9;
  c_12(1, 1) = 11;
  c_12(1, 2) = 13;
  c_12(1, 3) = 15;
  c_12(2, 0) = 11;
  c_12(2, 1) = 13;
  c_12(2, 2) = 15;
  c_12(2, 3) = 17;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;l,k") + rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
//                            Subtraction
//------------------------------------------------------------------------------

BOOST_FIXTURE_TEST_SUITE(tot_subtaction, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(vov, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2};
  lhs_0({0}) = 0;
  lhs_0({1}) = 1;
  inner_type lhs_1{r3};
  lhs_1({0}) = 1;
  lhs_1({1}) = 2;
  lhs_1({2}) = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2};
  rhs_0({0}) = 1;
  rhs_0({1}) = 2;
  inner_type rhs_1{r3};
  rhs_1({0}) = 2;
  rhs_1({1}) = 3;
  rhs_1({2}) = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2};
  c_0({0}) = -1;
  c_0({1}) = -1;
  inner_type c_1{r3};
  c_1({0}) = -1;
  c_1({1}) = -1;
  c_1({2}) = -1;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j") = lhs("i;j") - rhs("i;j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = -1;
  c_0(0, 1) = -1;
  c_0(0, 2) = -1;
  c_0(1, 0) = -1;
  c_0(1, 1) = -1;
  c_0(1, 2) = -1;
  inner_type c_1{r33};
  c_1(0, 0) = -1;
  c_1(0, 1) = -1;
  c_1(0, 2) = -1;
  c_1(1, 0) = -1;
  c_1(1, 1) = -1;
  c_1(1, 2) = -1;
  c_1(2, 0) = -1;
  c_1(2, 1) = -1;
  c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j,k") = lhs("i;j,k") - rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{range_type{3, 2}};
  c_0(0, 0) = -1;
  c_0(0, 1) = -1;
  c_0(1, 0) = -1;
  c_0(1, 1) = -1;
  c_0(2, 0) = -1;
  c_0(2, 1) = -1;
  inner_type c_1{r33};
  c_1(0, 0) = -1;
  c_1(0, 1) = -1;
  c_1(0, 2) = -1;
  c_1(1, 0) = -1;
  c_1(1, 1) = -1;
  c_1(1, 2) = -1;
  c_1(2, 0) = -1;
  c_1(2, 1) = -1;
  c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;k,j") = lhs("i;j,k") - rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(2, 0) = 2;
  lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = -1;
  c_0(0, 1) = -1;
  c_0(0, 2) = -1;
  c_0(1, 0) = -1;
  c_0(1, 1) = -1;
  c_0(1, 2) = -1;
  inner_type c_1{r33};
  c_1(0, 0) = -1;
  c_1(0, 1) = -1;
  c_1(0, 2) = -1;
  c_1(1, 0) = -1;
  c_1(1, 1) = -1;
  c_1(1, 2) = -1;
  c_1(2, 0) = -1;
  c_1(2, 1) = -1;
  c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;k,j") - rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_rhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{range_type{3, 2}};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(2, 0) = 3;
  rhs_0(2, 1) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = -1;
  c_0(0, 1) = -1;
  c_0(0, 2) = -1;
  c_0(1, 0) = -1;
  c_0(1, 1) = -1;
  c_0(1, 2) = -1;
  inner_type c_1{r33};
  c_1(0, 0) = -1;
  c_1(0, 1) = -1;
  c_1(0, 2) = -1;
  c_1(1, 0) = -1;
  c_1(1, 1) = -1;
  c_1(1, 2) = -1;
  c_1(2, 0) = -1;
  c_1(2, 1) = -1;
  c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;j,k") - rhs("i;k,j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_and_rhs_transpose, TestParam,
                              test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(2, 0) = 2;
  lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{range_type{3, 2}};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(2, 0) = 3;
  rhs_0(2, 1) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = -1;
  c_0(0, 1) = -1;
  c_0(0, 2) = -1;
  c_0(1, 0) = -1;
  c_0(1, 1) = -1;
  c_0(1, 2) = -1;
  inner_type c_1{r33};
  c_1(0, 0) = -1;
  c_1(0, 1) = -1;
  c_1(0, 2) = -1;
  c_1(1, 0) = -1;
  c_1(1, 1) = -1;
  c_1(1, 2) = -1;
  c_1(2, 0) = -1;
  c_1(2, 1) = -1;
  c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;k,j") - rhs("i;k,j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;

  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = -1;
  c_00({1}) = -1;
  inner_type c_01(range_type{2});
  c_01({0}) = -1;
  c_01({1}) = -1;
  inner_type c_02(range_type{2});
  c_02({0}) = -1;
  c_02({1}) = -1;
  inner_type c_10(range_type{3});
  c_10({0}) = -1;
  c_10({1}) = -1;
  c_10({2}) = -1;
  inner_type c_11(range_type{3});
  c_11({0}) = -1;
  c_11({1}) = -1;
  c_11({2}) = -1;
  inner_type c_12(range_type{3});
  c_12({0}) = -1;
  c_12({1}) = -1;
  c_12({2}) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k") = lhs("i,j;k") - rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = -1;
  c_00({1}) = -1;
  inner_type c_01(range_type{2});
  c_01({0}) = -1;
  c_01({1}) = -1;
  inner_type c_02(range_type{2});
  c_02({0}) = -1;
  c_02({1}) = -1;
  inner_type c_10(range_type{3});
  c_10({0}) = -1;
  c_10({1}) = -1;
  c_10({2}) = -1;
  inner_type c_11(range_type{3});
  c_11({0}) = -1;
  c_11({1}) = -1;
  c_11({2}) = -1;
  inner_type c_12(range_type{3});
  c_12({0}) = -1;
  c_12({1}) = -1;
  c_12({2}) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("j,i;k") = lhs("i,j;k") - rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = -1;
  c_00(0, 1) = -1;
  c_00(1, 0) = -1;
  c_00(1, 1) = -1;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = -1;
  c_01(0, 1) = -1;
  c_01(0, 2) = -1;
  c_01(1, 0) = -1;
  c_01(1, 1) = -1;
  c_01(1, 2) = -1;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = -1;
  c_02(0, 1) = -1;
  c_02(0, 2) = -1;
  c_02(0, 3) = -1;
  c_02(1, 0) = -1;
  c_02(1, 1) = -1;
  c_02(1, 2) = -1;
  c_02(1, 3) = -1;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = -1;
  c_10(0, 1) = -1;
  c_10(1, 0) = -1;
  c_10(1, 1) = -1;
  c_10(2, 0) = -1;
  c_10(2, 1) = -1;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = -1;
  c_11(0, 1) = -1;
  c_11(0, 2) = -1;
  c_11(1, 0) = -1;
  c_11(1, 1) = -1;
  c_11(1, 2) = -1;
  c_11(2, 0) = -1;
  c_11(2, 1) = -1;
  c_11(2, 2) = -1;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = -1;
  c_12(0, 1) = -1;
  c_12(0, 2) = -1;
  c_12(0, 3) = -1;
  c_12(1, 0) = -1;
  c_12(1, 1) = -1;
  c_12(1, 2) = -1;
  c_12(1, 3) = -1;
  c_12(2, 0) = -1;
  c_12(2, 1) = -1;
  c_12(2, 2) = -1;
  c_12(2, 3) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;k,l") - rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = -1;
  c_00(0, 1) = -1;
  c_00(1, 0) = -1;
  c_00(1, 1) = -1;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = -1;
  c_01(0, 1) = -1;
  c_01(0, 2) = -1;
  c_01(1, 0) = -1;
  c_01(1, 1) = -1;
  c_01(1, 2) = -1;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = -1;
  c_02(0, 1) = -1;
  c_02(0, 2) = -1;
  c_02(0, 3) = -1;
  c_02(1, 0) = -1;
  c_02(1, 1) = -1;
  c_02(1, 2) = -1;
  c_02(1, 3) = -1;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = -1;
  c_10(0, 1) = -1;
  c_10(1, 0) = -1;
  c_10(1, 1) = -1;
  c_10(2, 0) = -1;
  c_10(2, 1) = -1;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = -1;
  c_11(0, 1) = -1;
  c_11(0, 2) = -1;
  c_11(1, 0) = -1;
  c_11(1, 1) = -1;
  c_11(1, 2) = -1;
  c_11(2, 0) = -1;
  c_11(2, 1) = -1;
  c_11(2, 2) = -1;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = -1;
  c_12(0, 1) = -1;
  c_12(0, 2) = -1;
  c_12(0, 3) = -1;
  c_12(1, 0) = -1;
  c_12(1, 1) = -1;
  c_12(1, 2) = -1;
  c_12(1, 3) = -1;
  c_12(2, 0) = -1;
  c_12(2, 1) = -1;
  c_12(2, 2) = -1;
  c_12(2, 3) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("j,i;k,l") = lhs("i,j;k,l") - rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_lhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{3, 2});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(2, 0) = 3;
  lhs_01(2, 1) = 4;
  inner_type lhs_02(range_type{4, 2});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(2, 0) = 4;
  lhs_02(2, 1) = 5;
  lhs_02(3, 0) = 5;
  lhs_02(3, 1) = 6;
  inner_type lhs_10(range_type{2, 3});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(0, 2) = 3;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(1, 2) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{4, 3});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(3, 0) = 6;
  lhs_12(3, 1) = 7;
  lhs_12(3, 2) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = -1;
  c_00(0, 1) = -1;
  c_00(1, 0) = -1;
  c_00(1, 1) = -1;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = -1;
  c_01(0, 1) = -1;
  c_01(0, 2) = -1;
  c_01(1, 0) = -1;
  c_01(1, 1) = -1;
  c_01(1, 2) = -1;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = -1;
  c_02(0, 1) = -1;
  c_02(0, 2) = -1;
  c_02(0, 3) = -1;
  c_02(1, 0) = -1;
  c_02(1, 1) = -1;
  c_02(1, 2) = -1;
  c_02(1, 3) = -1;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = -1;
  c_10(0, 1) = -1;
  c_10(1, 0) = -1;
  c_10(1, 1) = -1;
  c_10(2, 0) = -1;
  c_10(2, 1) = -1;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = -1;
  c_11(0, 1) = -1;
  c_11(0, 2) = -1;
  c_11(1, 0) = -1;
  c_11(1, 1) = -1;
  c_11(1, 2) = -1;
  c_11(2, 0) = -1;
  c_11(2, 1) = -1;
  c_11(2, 2) = -1;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = -1;
  c_12(0, 1) = -1;
  c_12(0, 2) = -1;
  c_12(0, 3) = -1;
  c_12(1, 0) = -1;
  c_12(1, 1) = -1;
  c_12(1, 2) = -1;
  c_12(1, 3) = -1;
  c_12(2, 0) = -1;
  c_12(2, 1) = -1;
  c_12(2, 2) = -1;
  c_12(2, 3) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;l,k") - rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
//                      Binary Op with Scaling
//------------------------------------------------------------------------------

BOOST_FIXTURE_TEST_SUITE(tot_scale, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(vov_lhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2};
  lhs_0({0}) = 0;
  lhs_0({1}) = 1;
  inner_type lhs_1{r3};
  lhs_1({0}) = 1;
  lhs_1({1}) = 2;
  lhs_1({2}) = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2};
  rhs_0({0}) = 1;
  rhs_0({1}) = 2;
  inner_type rhs_1{r3};
  rhs_1({0}) = 2;
  rhs_1({1}) = 3;
  rhs_1({2}) = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2};
  c_0({0}) = 1;
  c_0({1}) = 4;
  inner_type c_1{r3};
  c_1({0}) = 4;
  c_1({1}) = 7;
  c_1({2}) = 10;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j") = 2 * lhs("i;j") + rhs("i;j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vov_rhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2};
  lhs_0({0}) = 0;
  lhs_0({1}) = 1;
  inner_type lhs_1{r3};
  lhs_1({0}) = 1;
  lhs_1({1}) = 2;
  lhs_1({2}) = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2};
  rhs_0({0}) = 1;
  rhs_0({1}) = 2;
  inner_type rhs_1{r3};
  rhs_1({0}) = 2;
  rhs_1({1}) = 3;
  rhs_1({2}) = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2};
  c_0({0}) = 2;
  c_0({1}) = 5;
  inner_type c_1{r3};
  c_1({0}) = 5;
  c_1({1}) = 8;
  c_1({2}) = 11;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j") = lhs("i;j") + 2 * rhs("i;j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 1;
  c_0(0, 1) = 4;
  c_0(0, 2) = 7;
  c_0(1, 0) = 4;
  c_0(1, 1) = 7;
  c_0(1, 2) = 10;
  inner_type c_1{r33};
  c_1(0, 0) = 4;
  c_1(0, 1) = 7;
  c_1(0, 2) = 10;
  c_1(1, 0) = 7;
  c_1(1, 1) = 10;
  c_1(1, 2) = 13;
  c_1(2, 0) = 10;
  c_1(2, 1) = 13;
  c_1(2, 2) = 16;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j,k") = 2 * lhs("i;j,k") + rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_rhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 2;
  c_0(0, 1) = 5;
  c_0(0, 2) = 8;
  c_0(1, 0) = 5;
  c_0(1, 1) = 8;
  c_0(1, 2) = 11;
  inner_type c_1{r33};
  c_1(0, 0) = 5;
  c_1(0, 1) = 8;
  c_1(0, 2) = 11;
  c_1(1, 0) = 8;
  c_1(1, 1) = 11;
  c_1(1, 2) = 14;
  c_1(2, 0) = 11;
  c_1(2, 1) = 14;
  c_1(2, 2) = 17;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j,k") = lhs("i;j,k") + 2 * rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_lhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;

  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = 1;
  c_00({1}) = 4;
  inner_type c_01(range_type{2});
  c_01({0}) = 4;
  c_01({1}) = 7;
  inner_type c_02(range_type{2});
  c_02({0}) = 7;
  c_02({1}) = 10;
  inner_type c_10(range_type{3});
  c_10({0}) = 4;
  c_10({1}) = 7;
  c_10({2}) = 10;
  inner_type c_11(range_type{3});
  c_11({0}) = 7;
  c_11({1}) = 10;
  c_11({2}) = 13;
  inner_type c_12(range_type{3});
  c_12({0}) = 10;
  c_12({1}) = 13;
  c_12({2}) = 16;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k") = 2 * lhs("i,j;k") + rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_rhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;

  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = 2;
  c_00({1}) = 5;
  inner_type c_01(range_type{2});
  c_01({0}) = 5;
  c_01({1}) = 8;
  inner_type c_02(range_type{2});
  c_02({0}) = 8;
  c_02({1}) = 11;
  inner_type c_10(range_type{3});
  c_10({0}) = 5;
  c_10({1}) = 8;
  c_10({2}) = 11;
  inner_type c_11(range_type{3});
  c_11({0}) = 8;
  c_11({1}) = 11;
  c_11({2}) = 14;
  inner_type c_12(range_type{3});
  c_12({0}) = 11;
  c_12({1}) = 14;
  c_12({2}) = 17;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k") = lhs("i,j;k") + 2 * rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_lhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 1;
  c_00(0, 1) = 4;
  c_00(1, 0) = 4;
  c_00(1, 1) = 7;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 4;
  c_01(0, 1) = 7;
  c_01(0, 2) = 10;
  c_01(1, 0) = 7;
  c_01(1, 1) = 10;
  c_01(1, 2) = 13;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 7;
  c_02(0, 1) = 10;
  c_02(0, 2) = 13;
  c_02(0, 3) = 16;
  c_02(1, 0) = 10;
  c_02(1, 1) = 13;
  c_02(1, 2) = 16;
  c_02(1, 3) = 19;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 4;
  c_10(0, 1) = 7;
  c_10(1, 0) = 7;
  c_10(1, 1) = 10;
  c_10(2, 0) = 10;
  c_10(2, 1) = 13;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 7;
  c_11(0, 1) = 10;
  c_11(0, 2) = 13;
  c_11(1, 0) = 10;
  c_11(1, 1) = 13;
  c_11(1, 2) = 16;
  c_11(2, 0) = 13;
  c_11(2, 1) = 16;
  c_11(2, 2) = 19;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 10;
  c_12(0, 1) = 13;
  c_12(0, 2) = 16;
  c_12(0, 3) = 19;
  c_12(1, 0) = 13;
  c_12(1, 1) = 16;
  c_12(1, 2) = 19;
  c_12(1, 3) = 22;
  c_12(2, 0) = 16;
  c_12(2, 1) = 19;
  c_12(2, 2) = 22;
  c_12(2, 3) = 25;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = 2 * lhs("i,j;k,l") + rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_rhs, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 2;
  c_00(0, 1) = 5;
  c_00(1, 0) = 5;
  c_00(1, 1) = 8;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 5;
  c_01(0, 1) = 8;
  c_01(0, 2) = 11;
  c_01(1, 0) = 8;
  c_01(1, 1) = 11;
  c_01(1, 2) = 14;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 8;
  c_02(0, 1) = 11;
  c_02(0, 2) = 14;
  c_02(0, 3) = 17;
  c_02(1, 0) = 11;
  c_02(1, 1) = 14;
  c_02(1, 2) = 17;
  c_02(1, 3) = 20;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 5;
  c_10(0, 1) = 8;
  c_10(1, 0) = 8;
  c_10(1, 1) = 11;
  c_10(2, 0) = 11;
  c_10(2, 1) = 14;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 8;
  c_11(0, 1) = 11;
  c_11(0, 2) = 14;
  c_11(1, 0) = 11;
  c_11(1, 1) = 14;
  c_11(1, 2) = 17;
  c_11(2, 0) = 14;
  c_11(2, 1) = 17;
  c_11(2, 2) = 20;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 11;
  c_12(0, 1) = 14;
  c_12(0, 2) = 17;
  c_12(0, 3) = 20;
  c_12(1, 0) = 14;
  c_12(1, 1) = 17;
  c_12(1, 2) = 20;
  c_12(1, 3) = 23;
  c_12(2, 0) = 17;
  c_12(2, 1) = 20;
  c_12(2, 2) = 23;
  c_12(2, 3) = 26;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;k,l") + 2 * rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
//                       Element-wise Product
//------------------------------------------------------------------------------

BOOST_FIXTURE_TEST_SUITE(tot_hadamard, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(vov, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2};
  lhs_0({0}) = 0;
  lhs_0({1}) = 1;
  inner_type lhs_1{r3};
  lhs_1({0}) = 1;
  lhs_1({1}) = 2;
  lhs_1({2}) = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2};
  rhs_0({0}) = 1;
  rhs_0({1}) = 2;
  inner_type rhs_1{r3};
  rhs_1({0}) = 2;
  rhs_1({1}) = 3;
  rhs_1({2}) = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2};
  c_0({0}) = 0;
  c_0({1}) = 2;
  inner_type c_1{r3};
  c_1({0}) = 2;
  c_1({1}) = 6;
  c_1({2}) = 12;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j") = lhs("i;j") * rhs("i;j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 0;
  c_0(0, 1) = 2;
  c_0(0, 2) = 6;
  c_0(1, 0) = 2;
  c_0(1, 1) = 6;
  c_0(1, 2) = 12;
  inner_type c_1{r33};
  c_1(0, 0) = 2;
  c_1(0, 1) = 6;
  c_1(0, 2) = 12;
  c_1(1, 0) = 6;
  c_1(1, 1) = 12;
  c_1(1, 2) = 20;
  c_1(2, 0) = 12;
  c_1(2, 1) = 20;
  c_1(2, 2) = 30;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;j,k") = lhs("i;j,k") * rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{range_type{3, 2}};
  c_0(0, 0) = 0;
  c_0(0, 1) = 2;
  c_0(1, 0) = 2;
  c_0(1, 1) = 6;
  c_0(2, 0) = 6;
  c_0(2, 1) = 12;
  inner_type c_1{r33};
  c_1(0, 0) = 2;
  c_1(0, 1) = 6;
  c_1(0, 2) = 12;
  c_1(1, 0) = 6;
  c_1(1, 1) = 12;
  c_1(1, 2) = 20;
  c_1(2, 0) = 12;
  c_1(2, 1) = 20;
  c_1(2, 2) = 30;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i;k,j") = lhs("i;j,k") * rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(2, 0) = 2;
  lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(0, 2) = 3;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 0;
  c_0(0, 1) = 2;
  c_0(0, 2) = 6;
  c_0(1, 0) = 2;
  c_0(1, 1) = 6;
  c_0(1, 2) = 12;
  inner_type c_1{r33};
  c_1(0, 0) = 2;
  c_1(0, 1) = 6;
  c_1(0, 2) = 12;
  c_1(1, 0) = 6;
  c_1(1, 1) = 12;
  c_1(1, 2) = 20;
  c_1(2, 0) = 12;
  c_1(2, 1) = 20;
  c_1(2, 2) = 30;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;k,j") * rhs("i;j,k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_rhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(0, 2) = 2;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{range_type{3, 2}};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(2, 0) = 3;
  rhs_0(2, 1) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 0;
  c_0(0, 1) = 2;
  c_0(0, 2) = 6;
  c_0(1, 0) = 2;
  c_0(1, 1) = 6;
  c_0(1, 2) = 12;
  inner_type c_1{r33};
  c_1(0, 0) = 2;
  c_1(0, 1) = 6;
  c_1(0, 2) = 12;
  c_1(1, 0) = 6;
  c_1(1, 1) = 12;
  c_1(1, 2) = 20;
  c_1(2, 0) = 12;
  c_1(2, 1) = 20;
  c_1(2, 2) = 30;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;j,k") * rhs("i;k,j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_and_rhs_transpose, TestParam,
                              test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
  lhs_0(0, 0) = 0;
  lhs_0(0, 1) = 1;
  lhs_0(1, 0) = 1;
  lhs_0(1, 1) = 2;
  lhs_0(2, 0) = 2;
  lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
  lhs_1(0, 0) = 1;
  lhs_1(0, 1) = 2;
  lhs_1(0, 2) = 3;
  lhs_1(1, 0) = 2;
  lhs_1(1, 1) = 3;
  lhs_1(1, 2) = 4;
  lhs_1(2, 0) = 3;
  lhs_1(2, 1) = 4;
  lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{range_type{3, 2}};
  rhs_0(0, 0) = 1;
  rhs_0(0, 1) = 2;
  rhs_0(1, 0) = 2;
  rhs_0(1, 1) = 3;
  rhs_0(2, 0) = 3;
  rhs_0(2, 1) = 4;
  inner_type rhs_1{r33};
  rhs_1(0, 0) = 2;
  rhs_1(0, 1) = 3;
  rhs_1(0, 2) = 4;
  rhs_1(1, 0) = 3;
  rhs_1(1, 1) = 4;
  rhs_1(1, 2) = 5;
  rhs_1(2, 0) = 4;
  rhs_1(2, 1) = 5;
  rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23};
  c_0(0, 0) = 0;
  c_0(0, 1) = 2;
  c_0(0, 2) = 6;
  c_0(1, 0) = 2;
  c_0(1, 1) = 6;
  c_0(1, 2) = 12;
  inner_type c_1{r33};
  c_1(0, 0) = 2;
  c_1(0, 1) = 6;
  c_1(0, 2) = 12;
  c_1(1, 0) = 6;
  c_1(1, 1) = 12;
  c_1(1, 2) = 20;
  c_1(2, 0) = 12;
  c_1(2, 1) = 20;
  c_1(2, 2) = 30;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result, temp;
  result("i;j,k") = lhs("i;k,j") * rhs("i;k,j");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = 0;
  c_00({1}) = 2;
  inner_type c_01(range_type{2});
  c_01({0}) = 2;
  c_01({1}) = 6;
  inner_type c_02(range_type{2});
  c_02({0}) = 6;
  c_02({1}) = 12;
  inner_type c_10(range_type{3});
  c_10({0}) = 2;
  c_10({1}) = 6;
  c_10({2}) = 12;
  inner_type c_11(range_type{3});
  c_11({0}) = 6;
  c_11({1}) = 12;
  c_11({2}) = 20;
  inner_type c_12(range_type{3});
  c_12({0}) = 12;
  c_12({1}) = 20;
  c_12({2}) = 30;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k") = lhs("i,j;k") * rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2});
  lhs_00({0}) = 0;
  lhs_00({1}) = 1;
  inner_type lhs_01(range_type{2});
  lhs_01({0}) = 1;
  lhs_01({1}) = 2;
  inner_type lhs_02(range_type{2});
  lhs_02({0}) = 2;
  lhs_02({1}) = 3;
  inner_type lhs_10(range_type{3});
  lhs_10({0}) = 1;
  lhs_10({1}) = 2;
  lhs_10({2}) = 3;
  inner_type lhs_11(range_type{3});
  lhs_11({0}) = 2;
  lhs_11({1}) = 3;
  lhs_11({2}) = 4;
  inner_type lhs_12(range_type{3});
  lhs_12({0}) = 3;
  lhs_12({1}) = 4;
  lhs_12({2}) = 5;

  inner_type rhs_02(range_type{2});
  rhs_02({0}) = 3;
  rhs_02({1}) = 4;
  inner_type rhs_12(range_type{3});
  rhs_12({0}) = 4;
  rhs_12({1}) = 5;
  rhs_12({2}) = 6;

  inner_type c_00(range_type{2});
  c_00({0}) = 0;
  c_00({1}) = 2;
  inner_type c_01(range_type{2});
  c_01({0}) = 2;
  c_01({1}) = 6;
  inner_type c_02(range_type{2});
  c_02({0}) = 6;
  c_02({1}) = 12;
  inner_type c_10(range_type{3});
  c_10({0}) = 2;
  c_10({1}) = 6;
  c_10({2}) = 12;
  inner_type c_11(range_type{3});
  c_11({0}) = 6;
  c_11({1}) = 12;
  c_11({2}) = 20;
  inner_type c_12(range_type{3});
  c_12({0}) = 12;
  c_12({1}) = 20;
  c_12({2}) = 30;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("j,i;k") = lhs("i,j;k") * rhs("i,j;k");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 0;
  c_00(0, 1) = 2;
  c_00(1, 0) = 2;
  c_00(1, 1) = 6;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 2;
  c_01(0, 1) = 6;
  c_01(0, 2) = 12;
  c_01(1, 0) = 6;
  c_01(1, 1) = 12;
  c_01(1, 2) = 20;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 6;
  c_02(0, 1) = 12;
  c_02(0, 2) = 20;
  c_02(0, 3) = 30;
  c_02(1, 0) = 12;
  c_02(1, 1) = 20;
  c_02(1, 2) = 30;
  c_02(1, 3) = 42;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 2;
  c_10(0, 1) = 6;
  c_10(1, 0) = 6;
  c_10(1, 1) = 12;
  c_10(2, 0) = 12;
  c_10(2, 1) = 20;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 6;
  c_11(0, 1) = 12;
  c_11(0, 2) = 20;
  c_11(1, 0) = 12;
  c_11(1, 1) = 20;
  c_11(1, 2) = 30;
  c_11(2, 0) = 20;
  c_11(2, 1) = 30;
  c_11(2, 2) = 42;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 12;
  c_12(0, 1) = 20;
  c_12(0, 2) = 30;
  c_12(0, 3) = 42;
  c_12(1, 0) = 20;
  c_12(1, 1) = 30;
  c_12(1, 2) = 42;
  c_12(1, 3) = 56;
  c_12(2, 0) = 30;
  c_12(2, 1) = 42;
  c_12(2, 2) = 56;
  c_12(2, 3) = 72;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;k,l") * rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_result_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{2, 3});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(0, 2) = 3;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{2, 4});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(0, 2) = 4;
  lhs_02(0, 3) = 5;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(1, 2) = 5;
  lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{3, 2});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(2, 0) = 3;
  lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{3, 4});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(0, 3) = 6;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(1, 3) = 7;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 0;
  c_00(0, 1) = 2;
  c_00(1, 0) = 2;
  c_00(1, 1) = 6;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 2;
  c_01(0, 1) = 6;
  c_01(0, 2) = 12;
  c_01(1, 0) = 6;
  c_01(1, 1) = 12;
  c_01(1, 2) = 20;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 6;
  c_02(0, 1) = 12;
  c_02(0, 2) = 20;
  c_02(0, 3) = 30;
  c_02(1, 0) = 12;
  c_02(1, 1) = 20;
  c_02(1, 2) = 30;
  c_02(1, 3) = 42;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 2;
  c_10(0, 1) = 6;
  c_10(1, 0) = 6;
  c_10(1, 1) = 12;
  c_10(2, 0) = 12;
  c_10(2, 1) = 20;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 6;
  c_11(0, 1) = 12;
  c_11(0, 2) = 20;
  c_11(1, 0) = 12;
  c_11(1, 1) = 20;
  c_11(1, 2) = 30;
  c_11(2, 0) = 20;
  c_11(2, 1) = 30;
  c_11(2, 2) = 42;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 12;
  c_12(0, 1) = 20;
  c_12(0, 2) = 30;
  c_12(0, 3) = 42;
  c_12(1, 0) = 20;
  c_12(1, 1) = 30;
  c_12(1, 2) = 42;
  c_12(1, 3) = 56;
  c_12(2, 0) = 30;
  c_12(2, 1) = 42;
  c_12(2, 2) = 56;
  c_12(2, 3) = 72;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("j,i;k,l") = lhs("i,j;k,l") * rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_lhs_transpose, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2, 2});
  lhs_00(0, 0) = 0;
  lhs_00(0, 1) = 1;
  lhs_00(1, 0) = 1;
  lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{3, 2});
  lhs_01(0, 0) = 1;
  lhs_01(0, 1) = 2;
  lhs_01(1, 0) = 2;
  lhs_01(1, 1) = 3;
  lhs_01(2, 0) = 3;
  lhs_01(2, 1) = 4;
  inner_type lhs_02(range_type{4, 2});
  lhs_02(0, 0) = 2;
  lhs_02(0, 1) = 3;
  lhs_02(1, 0) = 3;
  lhs_02(1, 1) = 4;
  lhs_02(2, 0) = 4;
  lhs_02(2, 1) = 5;
  lhs_02(3, 0) = 5;
  lhs_02(3, 1) = 6;
  inner_type lhs_10(range_type{2, 3});
  lhs_10(0, 0) = 1;
  lhs_10(0, 1) = 2;
  lhs_10(0, 2) = 3;
  lhs_10(1, 0) = 2;
  lhs_10(1, 1) = 3;
  lhs_10(1, 2) = 4;
  inner_type lhs_11(range_type{3, 3});
  lhs_11(0, 0) = 2;
  lhs_11(0, 1) = 3;
  lhs_11(0, 2) = 4;
  lhs_11(1, 0) = 3;
  lhs_11(1, 1) = 4;
  lhs_11(1, 2) = 5;
  lhs_11(2, 0) = 4;
  lhs_11(2, 1) = 5;
  lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{4, 3});
  lhs_12(0, 0) = 3;
  lhs_12(0, 1) = 4;
  lhs_12(0, 2) = 5;
  lhs_12(1, 0) = 4;
  lhs_12(1, 1) = 5;
  lhs_12(1, 2) = 6;
  lhs_12(2, 0) = 5;
  lhs_12(2, 1) = 6;
  lhs_12(2, 2) = 7;
  lhs_12(3, 0) = 6;
  lhs_12(3, 1) = 7;
  lhs_12(3, 2) = 8;

  inner_type rhs_00(range_type{2, 2});
  rhs_00(0, 0) = 1;
  rhs_00(0, 1) = 2;
  rhs_00(1, 0) = 2;
  rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{2, 3});
  rhs_01(0, 0) = 2;
  rhs_01(0, 1) = 3;
  rhs_01(0, 2) = 4;
  rhs_01(1, 0) = 3;
  rhs_01(1, 1) = 4;
  rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{2, 4});
  rhs_02(0, 0) = 3;
  rhs_02(0, 1) = 4;
  rhs_02(0, 2) = 5;
  rhs_02(0, 3) = 6;
  rhs_02(1, 0) = 4;
  rhs_02(1, 1) = 5;
  rhs_02(1, 2) = 6;
  rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{3, 2});
  rhs_10(0, 0) = 2;
  rhs_10(0, 1) = 3;
  rhs_10(1, 0) = 3;
  rhs_10(1, 1) = 4;
  rhs_10(2, 0) = 4;
  rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{3, 3});
  rhs_11(0, 0) = 3;
  rhs_11(0, 1) = 4;
  rhs_11(0, 2) = 5;
  rhs_11(1, 0) = 4;
  rhs_11(1, 1) = 5;
  rhs_11(1, 2) = 6;
  rhs_11(2, 0) = 5;
  rhs_11(2, 1) = 6;
  rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{3, 4});
  rhs_12(0, 0) = 4;
  rhs_12(0, 1) = 5;
  rhs_12(0, 2) = 6;
  rhs_12(0, 3) = 7;
  rhs_12(1, 0) = 5;
  rhs_12(1, 1) = 6;
  rhs_12(1, 2) = 7;
  rhs_12(1, 3) = 8;
  rhs_12(2, 0) = 6;
  rhs_12(2, 1) = 7;
  rhs_12(2, 2) = 8;
  rhs_12(2, 3) = 9;

  inner_type c_00(range_type{2, 2});
  c_00(0, 0) = 0;
  c_00(0, 1) = 2;
  c_00(1, 0) = 2;
  c_00(1, 1) = 6;
  inner_type c_01(range_type{2, 3});
  c_01(0, 0) = 2;
  c_01(0, 1) = 6;
  c_01(0, 2) = 12;
  c_01(1, 0) = 6;
  c_01(1, 1) = 12;
  c_01(1, 2) = 20;
  inner_type c_02(range_type{2, 4});
  c_02(0, 0) = 6;
  c_02(0, 1) = 12;
  c_02(0, 2) = 20;
  c_02(0, 3) = 30;
  c_02(1, 0) = 12;
  c_02(1, 1) = 20;
  c_02(1, 2) = 30;
  c_02(1, 3) = 42;
  inner_type c_10(range_type{3, 2});
  c_10(0, 0) = 2;
  c_10(0, 1) = 6;
  c_10(1, 0) = 6;
  c_10(1, 1) = 12;
  c_10(2, 0) = 12;
  c_10(2, 1) = 20;
  inner_type c_11(range_type{3, 3});
  c_11(0, 0) = 6;
  c_11(0, 1) = 12;
  c_11(0, 2) = 20;
  c_11(1, 0) = 12;
  c_11(1, 1) = 20;
  c_11(1, 2) = 30;
  c_11(2, 0) = 20;
  c_11(2, 1) = 30;
  c_11(2, 2) = 42;
  inner_type c_12(range_type{3, 4});
  c_12(0, 0) = 12;
  c_12(0, 1) = 20;
  c_12(0, 2) = 30;
  c_12(0, 3) = 42;
  c_12(1, 0) = 20;
  c_12(1, 1) = 30;
  c_12(1, 2) = 42;
  c_12(1, 3) = 56;
  c_12(2, 0) = 30;
  c_12(2, 1) = 42;
  c_12(2, 2) = 56;
  c_12(2, 3) = 72;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result("i,j;k,l") = lhs("i,j;l,k") * rhs("i,j;k,l");
  BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ij_mn_times_ij_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {86, 79, 35, 52, 36, 68, 39, 28, 83, 16, 18, 74, 83, 62});
  Tensor<double> lhs_elem_0_1(Range{8, 3},
                              {99, 66, 95, 76, 38, 24, 6,  79, 85, 52, 93, 90,
                               5,  4,  41, 42, 49, 4,  42, 94, 60, 99, 69, 8});
  Tensor<double> lhs_elem_1_0(Range{8, 3},
                              {5,  40, 94, 20, 25, 34, 25, 68, 94, 80, 61, 11,
                               94, 62, 25, 2,  64, 63, 28, 24, 7,  98, 59, 30});
  Tensor<double> lhs_elem_1_1(
      Range{9, 4},
      {37, 39, 26, 63, 10, 60, 5,  84, 72, 84, 10, 85, 33, 30, 12, 54, 63, 28,
       21, 7,  27, 12, 47, 86, 58, 67, 99, 52, 92, 97, 72, 92, 10, 42, 23, 8});
  Tensor<double> lhs_elem_2_0(
      Range{9, 4},
      {81, 43, 70, 13, 91, 91, 2,  43, 85, 72, 69, 28, 20, 61, 96, 9,  80, 42,
       17, 39, 85, 50, 18, 82, 56, 14, 95, 23, 22, 90, 80, 19, 83, 93, 43, 35});
  Tensor<double> lhs_elem_2_1(
      Range{10, 5},
      {20, 80, 94, 53, 72, 67, 77, 47, 81, 30, 65, 54, 61, 55, 48, 47, 91,
       20, 64, 69, 38, 49, 24, 39, 57, 24, 2,  31, 85, 12, 25, 15, 21, 59,
       99, 84, 8,  90, 45, 95, 55, 72, 46, 98, 40, 26, 2,  64, 46, 21});
  Tensor<double> lhs_elem_3_0(
      Range{10, 5},
      {30, 40, 24, 61, 11, 46, 35, 34, 79, 18, 55, 96, 43, 5,  11, 74, 78,
       82, 19, 19, 19, 72, 17, 81, 64, 13, 55, 95, 98, 99, 28, 43, 51, 38,
       33, 26, 6,  4,  97, 3,  91, 62, 89, 42, 71, 28, 81, 98, 72, 41});
  Tensor<double> lhs_elem_3_1(
      Range{11, 6},
      {100, 81, 93, 37, 58, 97, 90, 52, 94, 48, 63, 80, 12, 12, 13, 32, 93,
       42,  6,  3,  13, 82, 46, 57, 2,  28, 95, 90, 14, 87, 29, 40, 39, 36,
       69,  33, 10, 25, 57, 21, 77, 42, 68, 46, 4,  89, 21, 71, 20, 10, 31,
       54,  64, 96, 82, 72, 17, 19, 6,  93, 22, 78, 7,  23, 52, 63});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {51, 46, 7, 36, 56, 97, 4, 53, 67, 6, 35, 13, 62, 30});
  Tensor<double> rhs_elem_0_1(Range{8, 3},
                              {20, 83, 5, 78, 25, 77, 5,  3,  59, 54, 92, 98,
                               32, 28, 5, 94, 81, 35, 71, 87, 60, 30, 57, 26});
  Tensor<double> rhs_elem_1_0(Range{8, 3},
                              {56, 27, 42, 85, 52, 87, 70, 73, 63, 79, 33, 18,
                               8,  17, 18, 35, 97, 18, 6,  92, 14, 14, 13, 8});
  Tensor<double> rhs_elem_1_1(
      Range{9, 4},
      {32, 8,  91, 99, 88, 52, 41, 94, 14, 17, 1,  73, 43, 20, 89, 1,  63, 8,
       54, 20, 20, 37, 37, 9,  85, 38, 76, 6,  67, 52, 7,  87, 82, 33, 93, 2});
  Tensor<double> rhs_elem_2_0(
      Range{9, 4},
      {48, 66, 62, 8,  22, 94, 97, 43, 22, 20, 16, 10, 38, 40, 18, 13, 21, 25,
       73, 46, 91, 69, 13, 23, 94, 66, 66, 24, 51, 68, 89, 15, 69, 46, 9,  6});
  Tensor<double> rhs_elem_2_1(
      Range{10, 5},
      {86, 74, 40, 15, 11, 16, 2,  40, 24, 14, 95, 66, 58, 44, 30, 22, 88,
       89, 45, 89, 51, 39, 70, 19, 39, 73, 6,  30, 72, 62, 34, 44, 79, 60,
       81, 12, 68, 18, 68, 57, 83, 86, 47, 28, 20, 34, 60, 47, 99, 20});
  Tensor<double> rhs_elem_3_0(
      Range{10, 5},
      {84, 95, 77,  56, 80, 26, 91, 80, 74, 92, 76, 25, 43, 58, 28, 65, 67,
       38, 60, 1,   65, 52, 77, 34, 52, 5,  47, 77, 65, 20, 30, 51, 97, 37,
       23, 85, 100, 40, 91, 50, 14, 30, 97, 70, 83, 58, 28, 69, 26, 36});
  Tensor<double> rhs_elem_3_1(
      Range{11, 6},
      {3,  13, 87, 60, 90, 52, 7,  20, 4,  56, 41, 44, 83, 48, 41, 38, 65,
       80, 60, 63, 55, 70, 37, 9,  45, 30, 18, 27, 4,  50, 81, 45, 19, 38,
       54, 15, 71, 66, 42, 29, 42, 36, 20, 34, 79, 10, 81, 52, 85, 50, 18,
       7,  96, 84, 64, 87, 68, 55, 45, 67, 29, 14, 76, 12, 89, 86});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1},
                   {rhs_elem_1_0, rhs_elem_1_1},
                   {rhs_elem_2_0, rhs_elem_2_1},
                   {rhs_elem_3_0, rhs_elem_3_1}};
  TiledRange rhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {4386, 3634, 245, 1872, 2016, 6596, 156, 1484, 5561, 96, 630,
                    962, 5146, 1860});
  Tensor<double> corr_elem_0_1(
      Range{8, 3},
      {1980, 5478, 475, 5928, 950,  1848, 30,   237,  5015, 2808, 8556, 8820,
       160,  112,  205, 3948, 3969, 140,  2982, 8178, 3600, 2970, 3933, 208});
  Tensor<double> corr_elem_1_0(
      Range{8, 3},
      {280, 1080, 3948, 1700, 1300, 2958, 1750, 4964, 5922, 6320, 2013, 198,
       752, 1054, 450,  70,   6208, 1134, 168,  2208, 98,   1372, 767,  240});
  Tensor<double> corr_elem_1_1(
      Range{9, 4},
      {1184, 312,  2366, 6237, 880,  3120, 205,  7896, 1008, 1428, 10,   6205,
       1419, 600,  1068, 54,   3969, 224,  1134, 140,  540,  444,  1739, 774,
       4930, 2546, 7524, 312,  6164, 5044, 504,  8004, 820,  1386, 2139, 16});
  Tensor<double> corr_elem_2_0(
      Range{9, 4},
      {3888, 2838, 4340, 104, 2002, 8554, 194,  1849, 1870, 1440, 1104, 280,
       760,  2440, 1728, 117, 1680, 1050, 1241, 1794, 7735, 3450, 234,  1886,
       5264, 924,  6270, 552, 1122, 6120, 7120, 285,  5727, 4278, 387,  210});
  Tensor<double> corr_elem_2_1(
      Range{10, 5},
      {1720, 5920, 3760, 795,  792,  1072, 154,  1880, 1944, 420,
       6175, 3564, 3538, 2420, 1440, 1034, 8008, 1780, 2880, 6141,
       1938, 1911, 1680, 741,  2223, 1752, 12,   930,  6120, 744,
       850,  660,  1659, 3540, 8019, 1008, 544,  1620, 3060, 5415,
       4565, 6192, 2162, 2744, 800,  884,  120,  3008, 4554, 420});
  Tensor<double> corr_elem_3_0(
      Range{10, 5},
      {2520, 3800, 1848, 3416, 880,  1196, 3185, 2720, 5846, 1656,
       4180, 2400, 1849, 290,  308,  4810, 5226, 3116, 1140, 19,
       1235, 3744, 1309, 2754, 3328, 65,   2585, 7315, 6370, 1980,
       840,  2193, 4947, 1406, 759,  2210, 600,  160,  8827, 150,
       1274, 1860, 8633, 2940, 5893, 1624, 2268, 6762, 1872, 1476});
  Tensor<double> corr_elem_3_1(
      Range{11, 6},
      {300,  1053, 8091, 2220, 5220, 5044, 630,  1040, 376,  2688, 2583,
       3520, 996,  576,  533,  1216, 6045, 3360, 360,  189,  715,  5740,
       1702, 513,  90,   840,  1710, 2430, 56,   4350, 2349, 1800, 741,
       1368, 3726, 495,  710,  1650, 2394, 609,  3234, 1512, 1360, 1564,
       316,  890,  1701, 3692, 1700, 500,  558,  378,  6144, 8064, 5248,
       6264, 1156, 1045, 270,  6231, 638,  1092, 532,  276,  4628, 5418});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1},
                    {corr_elem_1_0, corr_elem_1_1},
                    {corr_elem_2_0, corr_elem_2_1},
                    {corr_elem_3_0, corr_elem_3_1}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out;
  out("i,j;m,n") = lhs("i,j;m,n") * rhs("i,j;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ij_mn_eq_ij_mn_times_ji_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {15, 75, 54, 54, 72, 62, 97, 90, 17, 94, 19, 54, 13, 31});
  Tensor<double> lhs_elem_0_1(Range{8, 3},
                              {82, 91, 60, 11, 47, 38, 87, 13, 72, 39, 59, 90,
                               26, 38, 2,  34, 30, 32, 46, 6,  26, 92, 47, 14});
  Tensor<double> lhs_elem_1_0(Range{8, 3},
                              {53, 88, 72, 12, 58, 85, 55, 6,  50, 76, 51, 52,
                               77, 13, 4,  99, 30, 12, 16, 21, 60, 75, 55, 99});
  Tensor<double> lhs_elem_1_1(
      Range{9, 4},
      {16, 65, 6,  84, 85, 30, 97, 79, 2,  13, 4,  90, 32, 98, 88, 40, 25, 27,
       8,  50, 56, 5,  42, 11, 20, 3,  51, 55, 32, 75, 8,  25, 4,  99, 75, 50});
  Tensor<double> lhs_elem_2_0(
      Range{9, 4}, {39, 24, 23, 32, 10, 22,  94, 47, 85, 22, 77, 22,
                    92, 28, 61, 53, 21, 81,  57, 63, 37, 75, 93, 91,
                    24, 14, 56, 69, 42, 100, 17, 44, 78, 47, 33, 67});
  Tensor<double> lhs_elem_2_1(
      Range{10, 5},
      {93, 27, 38, 15, 87, 88, 48, 19, 54, 81, 6,  60, 70, 75, 1,  21, 34,
       6,  74, 26, 5,  5,  75, 21, 31, 62, 53, 18, 17, 14, 19, 33, 96, 56,
       94, 12, 30, 14, 94, 31, 25, 59, 72, 88, 66, 98, 56, 79, 11, 50});
  Tensor<double> lhs_elem_3_0(
      Range{10, 5},
      {49, 46, 13, 98, 77, 100, 23, 99, 77, 64, 10, 31, 10, 70, 30, 18, 89,
       45, 81, 24, 45, 39, 83,  31, 3,  89, 35, 93, 70, 84, 43, 26, 96, 59,
       57, 1,  3,  33, 27, 53,  33, 3,  53, 7,  80, 54, 47, 77, 62, 23});
  Tensor<double> lhs_elem_3_1(
      Range{11, 6},
      {27, 61, 27, 63, 45, 14, 80, 20, 73, 74, 74, 9,  59, 92, 5,  4,  78,
       27, 53, 94, 70, 74, 1,  48, 30, 97, 51, 42, 93, 93, 81, 94, 73, 67,
       23, 98, 58, 17, 75, 73, 92, 16, 59, 5,  82, 22, 43, 58, 68, 44, 27,
       69, 79, 42, 99, 48, 78, 18, 9,  63, 1,  50, 9,  10, 82, 39});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {55, 2, 99, 28, 98, 27, 80, 69, 1, 66, 5, 9, 1, 80});
  Tensor<double> rhs_elem_0_1(Range{8, 3},
                              {19, 23, 52, 93, 6,  89, 68, 10, 4,  23, 24, 20,
                               99, 85, 81, 36, 82, 54, 36, 46, 26, 85, 15, 28});
  Tensor<double> rhs_elem_0_2(
      Range{9, 4},
      {57, 32, 86, 49, 55, 32, 100, 46, 2, 82, 84, 69, 63, 69, 12, 62, 21, 87,
       1,  40, 61, 56, 90, 53, 74,  72, 5, 21, 49, 97, 69, 83, 48, 38, 88, 9});
  Tensor<double> rhs_elem_0_3(
      Range{10, 5},
      {28, 7,   4,  92, 30, 7,  3,  70, 16, 51, 71, 14, 37, 33,  92, 90, 75,
       29, 52,  59, 15, 15, 96, 50, 39, 72, 22, 60, 56, 95, 45,  33, 25, 22,
       23, 100, 26, 27, 38, 88, 89, 36, 48, 46, 6,  88, 16, 100, 54, 43});
  Tensor<double> rhs_elem_1_0(
      Range{8, 3}, {55, 21, 79, 3,  77, 82, 65,  83, 66, 12, 100, 9,
                    40, 55, 8,  75, 82, 85, 100, 78, 39, 42, 65,  56});
  Tensor<double> rhs_elem_1_1(
      Range{9, 4},
      {45, 21, 58, 73, 57, 33, 27, 58, 56, 45, 88, 79, 78, 97, 23, 4,  87, 22,
       9,  21, 54, 44, 81, 98, 53, 60, 29, 70, 83, 75, 30, 56, 61, 67, 18, 61});
  Tensor<double> rhs_elem_1_2(
      Range{10, 5},
      {11, 17, 75, 95, 66, 51, 95, 79, 10, 2,  43, 3,  85, 64, 67, 50, 32,
       8,  48, 58, 35, 20, 87, 82, 40, 46, 70, 39, 46, 37, 38, 81, 87, 64,
       31, 32, 7,  14, 94, 21, 33, 75, 67, 5,  80, 80, 36, 53, 99, 93});
  Tensor<double> rhs_elem_1_3(
      Range{11, 6},
      {64, 8,  79, 99, 13,  5,  64, 76, 2,  81, 78, 89, 88, 89, 83, 99, 71,
       50, 18, 59, 91, 100, 91, 99, 20, 54, 72, 9,  43, 21, 61, 57, 18, 80,
       12, 27, 95, 31, 92,  4,  6,  59, 27, 82, 98, 32, 82, 53, 52, 8,  31,
       32, 38, 63, 32, 47,  24, 86, 64, 29, 86, 46, 96, 79, 48, 58});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1, rhs_elem_0_2, rhs_elem_0_3},
                   {rhs_elem_1_0, rhs_elem_1_1, rhs_elem_1_2, rhs_elem_1_3}};
  TiledRange rhs_trange{{0, 2}, {0, 2, 4}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {825, 150, 5346, 1512, 7056, 1674, 7760, 6210, 17, 6204, 95,
                    486, 13, 2480});
  Tensor<double> corr_elem_0_1(
      Range{8, 3},
      {4510, 1911, 4740, 33,   3619, 3116, 5655, 1079, 4752, 468,  5900, 810,
       1040, 2090, 16,   2550, 2460, 2720, 4600, 468,  1014, 3864, 3055, 784});
  Tensor<double> corr_elem_1_0(
      Range{8, 3},
      {1007, 2024, 3744, 1116, 348,  7565, 3740, 60,  200,  1748, 1224, 1040,
       7623, 1105, 324,  3564, 2460, 648,  576,  966, 1560, 6375, 825,  2772});
  Tensor<double> corr_elem_1_1(
      Range{9, 4},
      {720,  1365, 348,  6132, 4845, 990,  2619, 4582, 112,  585,  352,  7110,
       2496, 9506, 2024, 160,  2175, 594,  72,   1050, 3024, 220,  3402, 1078,
       1060, 180,  1479, 3850, 2656, 5625, 240,  1400, 244,  6633, 1350, 3050});
  Tensor<double> corr_elem_2_0(
      Range{9, 4},
      {2223, 768,  1978, 1568, 550,  704,  9400, 2162, 170,  1804, 6468, 1518,
       5796, 1932, 732,  3286, 441,  7047, 57,   2520, 2257, 4200, 8370, 4823,
       1776, 1008, 280,  1449, 2058, 9700, 1173, 3652, 3744, 1786, 2904, 603});
  Tensor<double> corr_elem_2_1(
      Range{10, 5},
      {1023, 459,  2850, 1425, 5742, 4488, 4560, 1501, 540,  162,
       258,  180,  5950, 4800, 67,   1050, 1088, 48,   3552, 1508,
       175,  100,  6525, 1722, 1240, 2852, 3710, 702,  782,  518,
       722,  2673, 8352, 3584, 2914, 384,  210,  196,  8836, 651,
       825,  4425, 4824, 440,  5280, 7840, 2016, 4187, 1089, 4650});
  Tensor<double> corr_elem_3_0(
      Range{10, 5}, {1372, 322, 52,   9016, 2310, 700,  69,   6930, 1232, 3264,
                     710,  434, 370,  2310, 2760, 1620, 6675, 1305, 4212, 1416,
                     675,  585, 7968, 1550, 117,  6408, 770,  5580, 3920, 7980,
                     1935, 858, 2400, 1298, 1311, 100,  78,   891,  1026, 4664,
                     2937, 108, 2544, 322,  480,  4752, 752,  7700, 3348, 989});
  Tensor<double> corr_elem_3_1(
      Range{11, 6},
      {1728, 488,  2133, 6237, 585,  70,   5120, 1520, 146,  5994, 5772,
       801,  5192, 8188, 415,  396,  5538, 1350, 954,  5546, 6370, 7400,
       91,   4752, 600,  5238, 3672, 378,  3999, 1953, 4941, 5358, 1314,
       5360, 276,  2646, 5510, 527,  6900, 292,  552,  944,  1593, 410,
       8036, 704,  3526, 3074, 3536, 352,  837,  2208, 3002, 2646, 3168,
       2256, 1872, 1548, 576,  1827, 86,   2300, 864,  790,  3936, 2262});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1},
                    {corr_elem_1_0, corr_elem_1_1},
                    {corr_elem_2_0, corr_elem_2_1},
                    {corr_elem_3_0, corr_elem_3_1}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out;
  out("i,j;m,n") = lhs("i,j;m,n") * rhs("j,i;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
//                           Contraction
//------------------------------------------------------------------------------

BOOST_FIXTURE_TEST_SUITE(tot_contraction, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(vov_inner_contraction, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type = detail::vector_il<inner_type>;
  using policy_type = policy_type<TestParam>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2};
  lhs_0({0}) = 0;
  lhs_0({1}) = 1;
  inner_type lhs_1{r3};
  lhs_1({0}) = 1;
  lhs_1({1}) = 2;
  lhs_1({2}) = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2};
  rhs_0({0}) = 1;
  rhs_0({1}) = 2;
  inner_type rhs_1{r3};
  rhs_1({0}) = 2;
  rhs_1({1}) = 3;
  rhs_1({2}) = 4;
  il_type rhs_il{rhs_0, rhs_1};

  using result_type = DistArray<Tensor<scalar_type<TestParam>>, policy_type>;
  // N.B. explicitly declare il type here due to
  // https://bugs.llvm.org//show_bug.cgi?id=23689
  result_type corr(m_world, TiledArray::detail::vector_il<double>{2.0, 20.0});
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> result;
  // einsum(result("i"), lhs("i;j"), rhs("i;j"));
  // BOOST_CHECK(are_equal(result, corr));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_outer_inner_contraction, TestParam,
                              test_params) {
  using inner_type = inner_type<TestParam>;
  using element_type = typename inner_type::value_type;
  using trange_type = TiledRange;
  trange_type tr{{0, 2, 3, 5, 7}, {0, 3, 5, 7, 11}};

  auto init_inner_tensor = [](const auto& elem_idx) {
    using range_type = typename inner_type::range_type;
    int x = 0;
    // N.B. to be able to contract for multiple outer index combinations
    // make all inner tensors same size
    return make_tensor<inner_type>(range_type(3, 5), [&x]() { return x++; });
  };

  static_assert(TiledArray::detail::is_tensor_of_tensor_v<tile_type<TestParam>>,
                "non-ToT tile");
  tensor_type<TestParam> lhs(m_world, tr);
  lhs.init_elements(init_inner_tensor);
  tensor_type<TestParam> rhs(m_world, tr);
  rhs.init_elements(init_inner_tensor);
  tensor_type<TestParam> result;
  result("i,k;l,n") = lhs("i,j;l,m") * rhs("k,j;n,m");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_inner_contraction, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using element_type = typename inner_type::value_type;
  //  detail::type_printer<inner_type> x;
  //  detail::type_printer<element_type> y;
  using trange_type = TiledRange;
  trange_type tr{{0, 2, 3, 5, 7}, {0, 3, 5, 7, 11}};

  auto init_inner_tensor = [](const auto& elem_idx) {
    using range_type = typename inner_type::range_type;
    int x = 0;
    return make_tensor<inner_type>(range_type(elem_idx[0] + 1, elem_idx[0] + 2),
                                   [&x]() { return x++; });
  };

  static_assert(TiledArray::detail::is_tensor_of_tensor_v<tile_type<TestParam>>,
                "non-ToT tile");
  tensor_type<TestParam> lhs(m_world, tr);
  lhs.init_elements(init_inner_tensor);
  tensor_type<TestParam> rhs(m_world, tr);
  rhs.init_elements(init_inner_tensor);
  tensor_type<TestParam> result;
  BOOST_CHECK_NO_THROW(result("i,j;n,l") = lhs("i,j;l,m") * rhs("i,j;n,m"));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_outer_contraction, TestParam, test_params) {
  using inner_type = inner_type<TestParam>;
  using element_type = typename inner_type::value_type;
  using trange_type = TiledRange;
  trange_type tr{{0, 2, 3, 5, 7}, {0, 3, 5, 7, 11}};

  auto init_inner_tensor = [](const auto& elem_idx) {
    using range_type = typename inner_type::range_type;
    int x = 0;
    // N.B. to be able to contract for multiple outer index combinations
    // make all inner tensors same size
    return make_tensor<inner_type>(range_type(3, 5), [&x]() { return x++; });
  };

  static_assert(TiledArray::detail::is_tensor_of_tensor_v<tile_type<TestParam>>,
                "non-ToT tile");
  tensor_type<TestParam> lhs(m_world, tr);
  lhs.init_elements(init_inner_tensor);
  tensor_type<TestParam> rhs(m_world, tr);
  rhs.init_elements(init_inner_tensor);
  tensor_type<TestParam> result;
  result("i,k;l,m") = lhs("i,j;l,m") * rhs("k,j;l,m");
}

BOOST_AUTO_TEST_CASE(ik_mn_eq_ij_mn_times_jk_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {49, 73, 28, 46, 12, 83, 29, 61, 61, 98, 57, 28, 96, 57});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {78, 15, 69, 55, 87, 94, 28, 94, 79, 30, 26, 88, 48, 74});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {70, 32, 25, 71, 6, 56, 4, 13, 72, 50, 15, 95, 52, 89});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {12, 29, 17, 68, 37, 79, 5, 52, 13, 35, 53, 54, 78, 71});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {77, 39, 34, 94, 16, 82, 63, 27, 75, 12, 14, 59, 3, 14});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {65, 90, 37, 41, 65, 75, 59, 16, 44, 85, 86, 11, 40, 24});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {77, 53, 11, 6, 99, 63, 46, 68, 83, 56, 76, 86, 91, 79});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {56, 11, 33, 90, 36, 38, 33, 54, 60, 21, 16, 28, 6, 97});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {53, 10, 70, 2, 14, 82, 81, 27, 22, 76, 51, 68, 77, 17});
  Tensor<double> rhs_elem_0_1(
      Range{7, 2}, {5, 99, 15, 7, 98, 85, 33, 92, 32, 91, 21, 19, 40, 54});
  Tensor<double> rhs_elem_0_2(
      Range{7, 2}, {4, 53, 38, 39, 8, 90, 43, 77, 57, 42, 92, 88, 76, 1});
  Tensor<double> rhs_elem_0_3(
      Range{7, 2}, {44, 86, 81, 43, 9, 40, 13, 6, 24, 12, 10, 60, 62, 59});
  Tensor<double> rhs_elem_0_4(
      Range{7, 2}, {35, 87, 52, 20, 55, 84, 66, 36, 41, 44, 3, 93, 91, 12});
  Tensor<double> rhs_elem_0_5(
      Range{7, 2}, {21, 23, 37, 39, 3, 25, 93, 61, 86, 11, 45, 70, 47, 30});
  Tensor<double> rhs_elem_1_0(
      Range{7, 2}, {63, 19, 60, 77, 10, 60, 68, 52, 5, 31, 44, 6, 90, 37});
  Tensor<double> rhs_elem_1_1(
      Range{7, 2}, {79, 73, 42, 83, 71, 83, 73, 32, 66, 87, 14, 76, 14, 72});
  Tensor<double> rhs_elem_1_2(
      Range{7, 2}, {80, 19, 59, 90, 73, 84, 22, 16, 1, 95, 90, 7, 97, 26});
  Tensor<double> rhs_elem_1_3(
      Range{7, 2}, {32, 55, 64, 81, 63, 97, 76, 86, 69, 61, 76, 87, 19, 72});
  Tensor<double> rhs_elem_1_4(
      Range{7, 2}, {63, 84, 5, 64, 43, 30, 78, 8, 65, 25, 99, 93, 23, 22});
  Tensor<double> rhs_elem_1_5(
      Range{7, 2}, {6, 74, 86, 41, 43, 63, 5, 83, 78, 54, 5, 1, 43, 97});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1, rhs_elem_0_2, rhs_elem_0_3,
                    rhs_elem_0_4, rhs_elem_0_5},
                   {rhs_elem_1_0, rhs_elem_1_1, rhs_elem_1_2, rhs_elem_1_3,
                    rhs_elem_1_4, rhs_elem_1_5}};
  TiledRange rhs_trange{{0, 2}, {0, 2, 4, 6}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {7511, 1015, 6100, 4327, 1038, 12446, 4253, 6535, 1737, 8378,
                    4051, 2432, 11712, 3707});
  Tensor<double> corr_elem_0_1(
      Range{7, 2}, {6407, 8322, 3318, 4887, 7353, 14857, 3001, 8620, 7166,
                    11528, 1561, 7220, 4512, 8406});
  Tensor<double> corr_elem_0_2(
      Range{7, 2}, {6436, 4154, 5135, 6744, 6447, 15366, 1863, 6201, 3556, 6966,
                    7584, 3080, 11952, 1981});
  Tensor<double> corr_elem_0_3(
      Range{7, 2}, {4652, 7103, 6684, 6433, 5589, 12438, 2505, 8450, 6915, 3006,
                    2546, 9336, 6864, 8691});
  Tensor<double> corr_elem_0_4(
      Range{7, 2}, {6629, 7611, 1801, 4440, 4401, 9792, 4098, 2948, 7636, 5062,
                    2745, 10788, 9840, 2312});
  Tensor<double> corr_elem_0_5(
      Range{7, 2}, {1497, 2789, 6970, 4049, 3777, 7997, 2837, 11523, 11408,
                    2698, 2695, 2048, 6576, 8888});
  Tensor<double> corr_elem_1_0(
      Range{7, 2}, {4466, 871, 2770, 5378, 454, 9332, 664, 3055, 1649, 4885,
                    3097, 6784, 11024, 4140});
  Tensor<double> corr_elem_1_1(
      Range{7, 2}, {1298, 5285, 1089, 6141, 3215, 11317, 497, 2860, 3162, 7595,
                    1057, 5909, 3172, 9918});
  Tensor<double> corr_elem_1_2(
      Range{7, 2}, {1240, 2247, 1953, 8889, 2749, 11676, 282, 1833, 4117, 5425,
                    6150, 8738, 11518, 1935});
  Tensor<double> corr_elem_1_3(
      Range{7, 2}, {3464, 4347, 3113, 8561, 2385, 9903, 432, 4550, 2625, 2735,
                    4178, 10398, 4706, 10363});
  Tensor<double> corr_elem_1_4(
      Range{7, 2}, {3206, 5220, 1385, 5772, 1921, 7074, 654, 884, 3797, 3075,
                    5292, 13857, 6526, 2630});
  Tensor<double> corr_elem_1_5(
      Range{7, 2}, {1542, 2882, 2387, 5557, 1609, 6377, 397, 5109, 7206, 2440,
                    940, 6704, 5798, 9557});
  Tensor<double> corr_elem_2_0(
      Range{7, 2}, {8176, 2100, 4600, 3345, 874, 11224, 9115, 1561, 1870, 3547,
                    4498, 4078, 3831, 1126});
  Tensor<double> corr_elem_2_1(
      Range{7, 2}, {5520, 10431, 2064, 4061, 6183, 13195, 6386, 2996, 5304,
                    8487, 1498, 1957, 680, 2484});
  Tensor<double> corr_elem_2_2(
      Range{7, 2}, {5508, 3777, 3475, 7356, 4873, 13680, 4007, 2335, 4319, 8579,
                    9028, 5269, 4108, 638});
  Tensor<double> corr_elem_2_3(
      Range{7, 2}, {5468, 8304, 5122, 7363, 4239, 10555, 5303, 1538, 4836, 5329,
                    6676, 4497, 946, 2554});
  Tensor<double> corr_elem_2_4(
      Range{7, 2}, {6790, 10953, 1953, 4504, 3675, 9138, 8760, 1100, 5935, 2653,
                    8556, 6510, 1193, 696});
  Tensor<double> corr_elem_2_5(
      Range{7, 2}, {2007, 7557, 4440, 5347, 2843, 6775, 6154, 2975, 9882, 4722,
                    1060, 4141, 1861, 2748});
  Tensor<double> corr_elem_3_0(
      Range{7, 2}, {7609, 739, 2750, 6942, 1746, 7446, 5970, 4644, 2126, 4907,
                    4580, 6016, 7547, 4932});
  Tensor<double> corr_elem_3_1(
      Range{7, 2}, {4809, 6050, 1551, 7512, 12258, 8509, 3927, 7984, 6616, 6923,
                    1820, 3762, 3724, 11250});
  Tensor<double> corr_elem_3_2(
      Range{7, 2}, {4788, 3018, 2365, 8334, 3420, 8862, 2704, 6100, 4791, 4347,
                    8432, 7764, 7498, 2601});
  Tensor<double> corr_elem_3_3(
      Range{7, 2}, {5180, 5163, 3003, 7548, 3159, 6206, 3106, 5052, 6132, 1953,
                    1976, 7596, 5756, 11645});
  Tensor<double> corr_elem_3_4(
      Range{7, 2}, {6223, 5535, 737, 5880, 6993, 6432, 5610, 2880, 7303, 2989,
                    1812, 10602, 8419, 3082});
  Tensor<double> corr_elem_3_5(
      Range{7, 2}, {1953, 2033, 3245, 3924, 1845, 3969, 4443, 8630, 11818, 1750,
                    3500, 6048, 4535, 11779});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1, corr_elem_0_2, corr_elem_0_3,
                     corr_elem_0_4, corr_elem_0_5},
                    {corr_elem_1_0, corr_elem_1_1, corr_elem_1_2, corr_elem_1_3,
                     corr_elem_1_4, corr_elem_1_5},
                    {corr_elem_2_0, corr_elem_2_1, corr_elem_2_2, corr_elem_2_3,
                     corr_elem_2_4, corr_elem_2_5},
                    {corr_elem_3_0, corr_elem_3_1, corr_elem_3_2, corr_elem_3_3,
                     corr_elem_3_4, corr_elem_3_5}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2, 4, 6}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out;
  out("i,k;m,n") = lhs("i,j;m,n") * rhs("j,k;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_CASE(ik_mn_eq_ij_mn_times_kj_mn) {
  using dist_array_t = DistArray<Tensor<Tensor<double>>, DensePolicy>;
  using matrix_il = TiledArray::detail::matrix_il<Tensor<double>>;
  auto& world = TiledArray::get_default_world();
  Tensor<double> lhs_elem_0_0(
      Range{7, 2}, {25, 18, 67, 84, 22, 81, 6, 31, 50, 17, 61, 2, 63, 58});
  Tensor<double> lhs_elem_0_1(
      Range{7, 2}, {3, 11, 71, 19, 43, 94, 20, 93, 52, 62, 38, 59, 42, 88});
  Tensor<double> lhs_elem_1_0(
      Range{7, 2}, {43, 50, 30, 27, 97, 41, 12, 21, 57, 8, 37, 81, 26, 94});
  Tensor<double> lhs_elem_1_1(
      Range{7, 2}, {1, 84, 87, 69, 68, 26, 86, 99, 16, 88, 38, 55, 91, 56});
  Tensor<double> lhs_elem_2_0(
      Range{7, 2}, {75, 9, 24, 79, 68, 7, 33, 16, 47, 11, 3, 26, 61, 48});
  Tensor<double> lhs_elem_2_1(
      Range{7, 2}, {71, 41, 63, 70, 75, 87, 76, 31, 20, 67, 45, 84, 92, 17});
  Tensor<double> lhs_elem_3_0(
      Range{7, 2}, {7, 69, 99, 84, 66, 28, 32, 28, 30, 74, 35, 78, 89, 73});
  Tensor<double> lhs_elem_3_1(
      Range{7, 2}, {21, 91, 52, 94, 67, 63, 17, 20, 48, 88, 100, 53, 35, 29});
  matrix_il lhs_il{{lhs_elem_0_0, lhs_elem_0_1},
                   {lhs_elem_1_0, lhs_elem_1_1},
                   {lhs_elem_2_0, lhs_elem_2_1},
                   {lhs_elem_3_0, lhs_elem_3_1}};
  TiledRange lhs_trange{{0, 2, 4}, {0, 2}};
  dist_array_t lhs(world, lhs_trange, lhs_il);
  Tensor<double> rhs_elem_0_0(
      Range{7, 2}, {12, 52, 47, 53, 66, 87, 11, 48, 24, 83, 16, 38, 26, 39});
  Tensor<double> rhs_elem_0_1(
      Range{7, 2}, {96, 38, 53, 99, 32, 16, 11, 69, 68, 17, 38, 53, 79, 83});
  Tensor<double> rhs_elem_1_0(
      Range{7, 2}, {26, 27, 3, 57, 26, 24, 35, 11, 96, 26, 83, 94, 19, 90});
  Tensor<double> rhs_elem_1_1(
      Range{7, 2}, {83, 98, 74, 21, 27, 14, 90, 75, 98, 75, 84, 5, 42, 88});
  Tensor<double> rhs_elem_2_0(
      Range{7, 2}, {72, 77, 10, 51, 62, 81, 58, 50, 56, 43, 68, 64, 54, 82});
  Tensor<double> rhs_elem_2_1(
      Range{7, 2}, {81, 20, 46, 1, 88, 39, 45, 79, 85, 48, 72, 1, 79, 89});
  Tensor<double> rhs_elem_3_0(
      Range{7, 2}, {84, 52, 99, 71, 30, 54, 17, 70, 18, 30, 77, 16, 65, 69});
  Tensor<double> rhs_elem_3_1(
      Range{7, 2}, {49, 80, 72, 71, 18, 51, 48, 22, 72, 43, 95, 22, 41, 88});
  Tensor<double> rhs_elem_4_0(
      Range{7, 2}, {70, 14, 59, 72, 53, 91, 31, 90, 56, 58, 28, 7, 87, 82});
  Tensor<double> rhs_elem_4_1(
      Range{7, 2}, {51, 95, 69, 13, 53, 75, 30, 71, 14, 10, 17, 44, 29, 77});
  Tensor<double> rhs_elem_5_0(
      Range{7, 2}, {95, 8, 90, 64, 96, 71, 41, 3, 100, 32, 14, 42, 77, 23});
  Tensor<double> rhs_elem_5_1(
      Range{7, 2}, {75, 96, 3, 89, 8, 11, 83, 18, 19, 24, 8, 69, 72, 38});
  matrix_il rhs_il{{rhs_elem_0_0, rhs_elem_0_1}, {rhs_elem_1_0, rhs_elem_1_1},
                   {rhs_elem_2_0, rhs_elem_2_1}, {rhs_elem_3_0, rhs_elem_3_1},
                   {rhs_elem_4_0, rhs_elem_4_1}, {rhs_elem_5_0, rhs_elem_5_1}};
  TiledRange rhs_trange{{0, 2, 4, 6}, {0, 2}};
  dist_array_t rhs(world, rhs_trange, rhs_il);
  Tensor<double> corr_elem_0_0(
      Range{7, 2}, {588, 1354, 6912, 6333, 2828, 8551, 286, 7905, 4736, 2465,
                    2420, 3203, 4956, 9566});
  Tensor<double> corr_elem_0_1(
      Range{7, 2}, {899, 1564, 5455, 5187, 1733, 3260, 2010, 7316, 9896, 5092,
                    8255, 483, 2961, 12964});
  Tensor<double> corr_elem_0_2(
      Range{7, 2}, {2043, 1606, 3936, 4303, 5148, 10227, 1248, 8897, 7220, 3707,
                    6884, 187, 6720, 12588});
  Tensor<double> corr_elem_0_3(
      Range{7, 2}, {2247, 1816, 11745, 7313, 1434, 9168, 1062, 4216, 4644, 3176,
                    8307, 1330, 5817, 11746});
  Tensor<double> corr_elem_0_4(
      Range{7, 2}, {1903, 1297, 8852, 6295, 3445, 14421, 786, 9393, 3528, 1606,
                    2354, 2610, 6699, 11532});
  Tensor<double> corr_elem_0_5(
      Range{7, 2}, {2600, 1200, 6243, 7067, 2456, 6785, 1906, 1767, 5988, 2032,
                    1158, 4155, 7875, 4678});
  Tensor<double> corr_elem_1_0(
      Range{7, 2}, {612, 5792, 6021, 8262, 8578, 3983, 1078, 7839, 2456, 2160,
                    2036, 5993, 7865, 8314});
  Tensor<double> corr_elem_1_1(
      Range{7, 2}, {1201, 9582, 6528, 2988, 4358, 1348, 8160, 7656, 7040, 6808,
                    6263, 7889, 4316, 13388});
  Tensor<double> corr_elem_1_2(
      Range{7, 2}, {3177, 5530, 4302, 1446, 11998, 4335, 4566, 8871, 4552, 4568,
                    5252, 5239, 8593, 12692});
  Tensor<double> corr_elem_1_3(
      Range{7, 2}, {3661, 9320, 9234, 6816, 4134, 3540, 4332, 3648, 2178, 4024,
                    6459, 2506, 5421, 11414});
  Tensor<double> corr_elem_1_4(
      Range{7, 2}, {3061, 8680, 7773, 2841, 8745, 5681, 2952, 8919, 3416, 1344,
                    1682, 2987, 4901, 12020});
  Tensor<double> corr_elem_1_5(
      Range{7, 2}, {4160, 8464, 2961, 7869, 9856, 3197, 7630, 1845, 6004, 2368,
                    822, 7197, 8554, 4290});
  Tensor<double> corr_elem_2_0(
      Range{7, 2}, {7716, 2026, 4467, 11117, 6888, 2001, 1199, 2907, 2488, 2052,
                    1758, 5440, 8854, 3283});
  Tensor<double> corr_elem_2_1(
      Range{7, 2}, {7843, 4261, 4734, 5973, 3793, 1386, 7995, 2501, 6472, 5311,
                    4029, 2864, 5023, 5816});
  Tensor<double> corr_elem_2_2(
      Range{7, 2}, {11151, 1513, 3138, 4099, 10816, 3960, 5334, 3249, 4332,
                    3689, 3444, 1748, 10562, 5449});
  Tensor<double> corr_elem_2_3(
      Range{7, 2}, {9779, 3748, 6912, 10579, 3390, 4815, 4209, 1802, 2286, 3211,
                    4506, 2264, 7737, 4808});
  Tensor<double> corr_elem_2_4(
      Range{7, 2}, {8871, 4021, 5763, 6598, 7579, 7162, 3303, 3641, 2912, 1308,
                    849, 3878, 7975, 5245});
  Tensor<double> corr_elem_2_5(
      Range{7, 2}, {12450, 4008, 2349, 11286, 7128, 1454, 7661, 606, 5080, 1960,
                    402, 6888, 11321, 1750});
  Tensor<double> corr_elem_3_0(
      Range{7, 2}, {2100, 7046, 7409, 13758, 6500, 3444, 539, 2724, 3984, 7638,
                    4360, 5773, 5079, 5254});
  Tensor<double> corr_elem_3_1(
      Range{7, 2}, {1925, 10781, 4145, 6762, 3525, 1554, 2650, 1808, 7584, 8524,
                    11305, 7597, 3161, 9122});
  Tensor<double> corr_elem_3_2(
      Range{7, 2}, {2205, 7133, 3382, 4378, 9988, 4725, 2621, 2980, 5760, 7406,
                    9580, 5045, 7571, 8567});
  Tensor<double> corr_elem_3_3(
      Range{7, 2}, {1617, 10868, 13545, 12638, 3186, 4725, 1360, 2400, 3996,
                    6004, 12195, 2414, 7220, 7589});
  Tensor<double> corr_elem_3_4(
      Range{7, 2}, {1561, 9611, 9429, 7270, 7049, 7273, 1502, 3940, 2352, 5172,
                    2680, 2878, 8758, 8219});
  Tensor<double> corr_elem_3_5(
      Range{7, 2}, {2240, 9288, 9066, 13742, 6872, 2681, 2723, 444, 3912, 4480,
                    1290, 6933, 9373, 2781});
  matrix_il corr_il{{corr_elem_0_0, corr_elem_0_1, corr_elem_0_2, corr_elem_0_3,
                     corr_elem_0_4, corr_elem_0_5},
                    {corr_elem_1_0, corr_elem_1_1, corr_elem_1_2, corr_elem_1_3,
                     corr_elem_1_4, corr_elem_1_5},
                    {corr_elem_2_0, corr_elem_2_1, corr_elem_2_2, corr_elem_2_3,
                     corr_elem_2_4, corr_elem_2_5},
                    {corr_elem_3_0, corr_elem_3_1, corr_elem_3_2, corr_elem_3_3,
                     corr_elem_3_4, corr_elem_3_5}};
  TiledRange corr_trange{{0, 2, 4}, {0, 2, 4, 6}};
  dist_array_t corr(world, corr_trange, corr_il);
  dist_array_t out;
  out("i,k;m,n") = lhs("i,j;m,n") * rhs("k,j;m,n");
  const bool are_equal = ToTArrayFixture::are_equal(corr, out);
  BOOST_CHECK(are_equal);
}

BOOST_AUTO_TEST_SUITE_END()
