#include "tot_array_fixture.h"

//------------------------------------------------------------------------------
//                            Permutations
//------------------------------------------------------------------------------
BOOST_FIXTURE_TEST_SUITE(tot_permutations, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(no_perm, TestParam, test_params){
  for(auto tr_t : run_all<TestParam>()){
    auto& in_rank = std::get<1>(tr_t);
    auto& t       = std::get<2>(tr_t);

    std::string out_idx = t.range().rank() == 1 ? "i" : "i, j";
    std::string in_idx  = in_rank == 1 ? "k" : "k, l";
    std::string idx     = out_idx + ";" + in_idx;

    tensor_type<TestParam> result;
    result(idx) = t(idx);
    BOOST_TEST(are_equal(result, t));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(permute_outer, TestParam, test_params){
  for(auto tr_t : run_all<TestParam>()){
    auto& in_rank = std::get<1>(tr_t);
    auto& t       = std::get<2>(tr_t);

    if(t.range().rank() == 1) continue;

    std::string rhs_out_idx = "i, j";
    std::string lhs_out_idx = "j, i";
    std::string in_idx      = in_rank == 1 ? "k" : "k, l";
    std::string rhs_idx     = rhs_out_idx + ";" + in_idx;
    std::string lhs_idx     = lhs_out_idx + ";" + in_idx;
    tensor_type<TestParam> result;
    result(lhs_idx) = t(rhs_idx);

    for(auto tile_idx : t.range()){
      auto rtile = t.find(tile_idx).get();
      auto ltile = result.find({tile_idx[1], tile_idx[0]}).get();
      for(auto outer_idx : ltile.range()){
        auto inner_range = ltile(outer_idx).range();
        auto outer_idx_t = {outer_idx[1], outer_idx[0]};
        bool same_inner_range = inner_range == rtile(outer_idx_t).range();
        BOOST_CHECK(same_inner_range);
        for(auto inner_idx : inner_range){
          const auto lelem = ltile(outer_idx)(inner_idx);
          const auto relem = rtile(outer_idx_t)(inner_idx);
          BOOST_CHECK_EQUAL(lelem, relem);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(permute_inner, TestParam, test_params){
  for(auto tr_t : run_all<TestParam>()){
    auto& in_rank = std::get<1>(tr_t);
    auto& t       = std::get<2>(tr_t);

    if(in_rank == 1) continue;

    std::string rhs_in_idx = "i, j";
    std::string lhs_in_idx = "j, i";
    std::string out_idx    = t.range().rank() == 1 ? "k" : "k, l";
    std::string rhs_idx    = out_idx + ";" + rhs_in_idx;
    std::string lhs_idx    = out_idx + ";" + lhs_in_idx;
    tensor_type<TestParam> result;
    result(lhs_idx) = t(rhs_idx);

    for(auto tile_idx : t.range()){
      auto rtile = t.find(tile_idx).get();
      auto ltile = result.find(tile_idx).get();
      bool same_outer_range = ltile.range() == rtile.range();
      BOOST_CHECK(same_outer_range);
      for(auto outer_idx : ltile.range()){
        auto inner_range = ltile(outer_idx).range();
        for(auto inner_idx : inner_range){
          const auto lelem = ltile(outer_idx)(inner_idx);
          const auto inner_idx_t ={inner_idx[1], inner_idx[0]};
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

BOOST_AUTO_TEST_CASE_TEMPLATE(vov, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2}; lhs_0[0] = 0; lhs_0[1] = 1;
  inner_type lhs_1{r3}; lhs_1[0] = 1; lhs_1[1] = 2; lhs_1[2] = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2}; rhs_0[0] = 1; rhs_0[1] = 2;
  inner_type rhs_1{r3}; rhs_1[0] = 2; rhs_1[1] = 3; rhs_1[2] = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2}; c_0[0] = 1; c_0[1] = 3;
  inner_type c_1{r3}; c_1[0] = 3; c_1[1] = 5; c_1[2] = 7;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;j") = lhs("i;j") + rhs("i;j");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23}; lhs_0(0, 0) = 0; lhs_0(0, 1) = 1; lhs_0(0, 2) = 2;
                         lhs_0(1, 0) = 1; lhs_0(1, 1) = 2; lhs_0(1, 2) = 3;
  inner_type lhs_1{r33}; lhs_1(0, 0) = 1; lhs_1(0, 1) = 2; lhs_1(0, 2) = 3;
                         lhs_1(1, 0) = 2; lhs_1(1, 1) = 3; lhs_1(1, 2) = 4;
                         lhs_1(2, 0) = 3; lhs_1(2, 1) = 4; lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23}; rhs_0(0, 0) = 1; rhs_0(0, 1) = 2; rhs_0(0, 2) = 3;
                         rhs_0(1, 0) = 2; rhs_0(1, 1) = 3; rhs_0(1, 2) = 4;
  inner_type rhs_1{r33}; rhs_1(0, 0) = 2; rhs_1(0, 1) = 3; rhs_1(0, 2) = 4;
                         rhs_1(1, 0) = 3; rhs_1(1, 1) = 4; rhs_1(1, 2) = 5;
                         rhs_1(2, 0) = 4; rhs_1(2, 1) = 5; rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23}; c_0(0, 0) = 1; c_0(0, 1) = 3; c_0(0, 2) = 5;
                       c_0(1, 0) = 3; c_0(1, 1) = 5; c_0(1, 2) = 7;
  inner_type c_1{r33}; c_1(0, 0) = 3; c_1(0, 1) = 5; c_1(0, 2) = 7;
                       c_1(1, 0) = 5; c_1(1, 1) = 7; c_1(1, 2) = 9;
                       c_1(2, 0) = 7; c_1(2, 1) = 9; c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;j,k") = lhs("i;j,k") + rhs("i;j,k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_result_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23};
    lhs_0(0, 0) = 0; lhs_0(0, 1) = 1; lhs_0(0, 2) = 2;
    lhs_0(1, 0) = 1; lhs_0(1, 1) = 2; lhs_0(1, 2) = 3;
  inner_type lhs_1{r33};
    lhs_1(0, 0) = 1; lhs_1(0, 1) = 2; lhs_1(0, 2) = 3;
    lhs_1(1, 0) = 2; lhs_1(1, 1) = 3; lhs_1(1, 2) = 4;
    lhs_1(2, 0) = 3; lhs_1(2, 1) = 4; lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
    rhs_0(0, 0) = 1; rhs_0(0, 1) = 2; rhs_0(0, 2) = 3;
    rhs_0(1, 0) = 2; rhs_0(1, 1) = 3; rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
    rhs_1(0, 0) = 2; rhs_1(0, 1) = 3; rhs_1(0, 2) = 4;
    rhs_1(1, 0) = 3; rhs_1(1, 1) = 4; rhs_1(1, 2) = 5;
    rhs_1(2, 0) = 4; rhs_1(2, 1) = 5; rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{range_type{3, 2}}; c_0(0, 0) = 1; c_0(0, 1) = 3;
                                    c_0(1, 0) = 3; c_0(1, 1) = 5;
                                    c_0(2, 0) = 5; c_0(2, 1) = 7;
  inner_type c_1{r33}; c_1(0, 0) = 3; c_1(0, 1) = 5; c_1(0, 2) = 7;
                       c_1(1, 0) = 5; c_1(1, 1) = 7; c_1(1, 2) = 9;
                       c_1(2, 0) = 7; c_1(2, 1) = 9; c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;k,j") = lhs("i;j,k") + rhs("i;j,k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0(range_type{3, 2});
    lhs_0(0, 0) = 0; lhs_0(0, 1) = 1;
    lhs_0(1, 0) = 1; lhs_0(1, 1) = 2;
    lhs_0(2, 0) = 2; lhs_0(2, 1) = 3;
  inner_type lhs_1{r33};
    lhs_1(0, 0) = 1; lhs_1(0, 1) = 2; lhs_1(0, 2) = 3;
    lhs_1(1, 0) = 2; lhs_1(1, 1) = 3; lhs_1(1, 2) = 4;
    lhs_1(2, 0) = 3; lhs_1(2, 1) = 4; lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23};
    rhs_0(0, 0) = 1; rhs_0(0, 1) = 2; rhs_0(0, 2) = 3;
    rhs_0(1, 0) = 2; rhs_0(1, 1) = 3; rhs_0(1, 2) = 4;
  inner_type rhs_1{r33};
    rhs_1(0, 0) = 2; rhs_1(0, 1) = 3; rhs_1(0, 2) = 4;
    rhs_1(1, 0) = 3; rhs_1(1, 1) = 4; rhs_1(1, 2) = 5;
    rhs_1(2, 0) = 4; rhs_1(2, 1) = 5; rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{range_type{3, 2}}; c_0(0, 0) = 1; c_0(0, 1) = 3;
  c_0(1, 0) = 3; c_0(1, 1) = 5;
  c_0(2, 0) = 5; c_0(2, 1) = 7;
  inner_type c_1{r33}; c_1(0, 0) = 3; c_1(0, 1) = 5; c_1(0, 2) = 7;
  c_1(1, 0) = 5; c_1(1, 1) = 7; c_1(1, 2) = 9;
  c_1(2, 0) = 7; c_1(2, 1) = 9; c_1(2, 2) = 11;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;j,k") = lhs("i;k,j") + rhs("i;j,k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2}); lhs_00[0] = 0; lhs_00[1] = 1;
  inner_type lhs_01(range_type{2}); lhs_01[0] = 1; lhs_01[1] = 2;
  inner_type lhs_02(range_type{2}); lhs_02[0] = 2; lhs_02[1] = 3;
  inner_type lhs_10(range_type{3}); lhs_10[0] = 1; lhs_10[1] = 2; lhs_10[2] = 3;
  inner_type lhs_11(range_type{3}); lhs_11[0] = 2; lhs_11[1] = 3; lhs_11[2] = 4;
  inner_type lhs_12(range_type{3}); lhs_12[0] = 3; lhs_12[1] = 4; lhs_12[2] = 5;

  inner_type rhs_02(range_type{2}); rhs_02[0] = 3; rhs_02[1] = 4;
  inner_type rhs_12(range_type{3}); rhs_12[0] = 4; rhs_12[1] = 5; rhs_12[2] = 6;

  inner_type c_00(range_type{2}); c_00[0] = 1; c_00[1] = 3;
  inner_type c_01(range_type{2}); c_01[0] = 3; c_01[1] = 5;
  inner_type c_02(range_type{2}); c_02[0] = 5; c_02[1] = 7;
  inner_type c_10(range_type{3}); c_10[0] = 3; c_10[1] = 5; c_10[2] = 7;
  inner_type c_11(range_type{3}); c_11[0] = 5; c_11[1] = 7; c_11[2] = 9;
  inner_type c_12(range_type{3}); c_12[0] = 7; c_12[1] = 9; c_12[2] = 11;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i,j;k") = lhs("i,j;k") + rhs("i,j;k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_result_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2}); lhs_00[0] = 0; lhs_00[1] = 1;
  inner_type lhs_01(range_type{2}); lhs_01[0] = 1; lhs_01[1] = 2;
  inner_type lhs_02(range_type{2}); lhs_02[0] = 2; lhs_02[1] = 3;
  inner_type lhs_10(range_type{3}); lhs_10[0] = 1; lhs_10[1] = 2; lhs_10[2] = 3;
  inner_type lhs_11(range_type{3}); lhs_11[0] = 2; lhs_11[1] = 3; lhs_11[2] = 4;
  inner_type lhs_12(range_type{3}); lhs_12[0] = 3; lhs_12[1] = 4; lhs_12[2] = 5;

  inner_type rhs_02(range_type{2}); rhs_02[0] = 3; rhs_02[1] = 4;
  inner_type rhs_12(range_type{3}); rhs_12[0] = 4; rhs_12[1] = 5; rhs_12[2] = 6;

  inner_type c_00(range_type{2}); c_00[0] = 1; c_00[1] = 3;
  inner_type c_01(range_type{2}); c_01[0] = 3; c_01[1] = 5;
  inner_type c_02(range_type{2}); c_02[0] = 5; c_02[1] = 7;
  inner_type c_10(range_type{3}); c_10[0] = 3; c_10[1] = 5; c_10[2] = 7;
  inner_type c_11(range_type{3}); c_11[0] = 5; c_11[1] = 7; c_11[2] = 9;
  inner_type c_12(range_type{3}); c_12[0] = 7; c_12[1] = 9; c_12[2] = 11;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("j,i;k") = lhs("i,j;k") + rhs("i,j;k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{{2, 2}});
    lhs_00(0, 0) = 0; lhs_00(0, 1) = 1;
    lhs_00(1, 0) = 1; lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{{2, 3}});
    lhs_01(0, 0) = 1; lhs_01(0, 1) = 2; lhs_01(0, 2) = 3;
    lhs_01(1, 0) = 2; lhs_01(1, 1) = 3; lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{{2, 4}});
    lhs_02(0, 0) = 2; lhs_02(0, 1) = 3; lhs_02(0, 2) = 4; lhs_02(0, 3) = 5;
    lhs_02(1, 0) = 3; lhs_02(1, 1) = 4; lhs_02(1, 2) = 5; lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{{3, 2}});
    lhs_10(0, 0) = 1; lhs_10(0, 1) = 2;
    lhs_10(1, 0) = 2; lhs_10(1, 1) = 3;
    lhs_10(2, 0) = 3; lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{{3, 3}});
    lhs_11(0, 0) = 2; lhs_11(0, 1) = 3; lhs_11(0, 2) = 4;
    lhs_11(1, 0) = 3; lhs_11(1, 1) = 4; lhs_11(1, 2) = 5;
    lhs_11(2, 0) = 4; lhs_11(2, 1) = 5; lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{{3, 4}});
    lhs_12(0, 0) = 3; lhs_12(0, 1) = 4; lhs_12(0, 2) = 5; lhs_12(0, 3) = 6;
    lhs_12(1, 0) = 4; lhs_12(1, 1) = 5; lhs_12(1, 2) = 6; lhs_12(1, 3) = 7;
    lhs_12(2, 0) = 5; lhs_12(2, 1) = 6; lhs_12(2, 2) = 7; lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{{2, 2}});
    rhs_00(0, 0) = 1; rhs_00(0, 1) = 2;
    rhs_00(1, 0) = 2; rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{{2, 3}});
    rhs_01(0, 0) = 2; rhs_01(0, 1) = 3; rhs_01(0, 2) = 4;
    rhs_01(1, 0) = 3; rhs_01(1, 1) = 4; rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{{2, 4}});
    rhs_02(0, 0) = 3; rhs_02(0, 1) = 4; rhs_02(0, 2) = 5; rhs_02(0, 3) = 6;
    rhs_02(1, 0) = 4; rhs_02(1, 1) = 5; rhs_02(1, 2) = 6; rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{{3, 2}});
    rhs_10(0, 0) = 2; rhs_10(0, 1) = 3;
    rhs_10(1, 0) = 3; rhs_10(1, 1) = 4;
    rhs_10(2, 0) = 4; rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{{3, 3}});
    rhs_11(0, 0) = 3; rhs_11(0, 1) = 4; rhs_11(0, 2) = 5;
    rhs_11(1, 0) = 4; rhs_11(1, 1) = 5; rhs_11(1, 2) = 6;
    rhs_11(2, 0) = 5; rhs_11(2, 1) = 6; rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{{3, 4}});
    rhs_12(0, 0) = 4; rhs_12(0, 1) = 5; rhs_12(0, 2) = 6; rhs_12(0, 3) = 7;
    rhs_12(1, 0) = 5; rhs_12(1, 1) = 6; rhs_12(1, 2) = 7; rhs_12(1, 3) = 8;
    rhs_12(2, 0) = 6; rhs_12(2, 1) = 7; rhs_12(2, 2) = 8; rhs_12(2, 3) = 9;

  inner_type c_00(range_type{{2, 2}});
    c_00(0, 0) = 1; c_00(0, 1) = 3;
    c_00(1, 0) = 3; c_00(1, 1) = 5;
  inner_type c_01(range_type{{2, 3}});
    c_01(0, 0) = 3; c_01(0, 1) = 5; c_01(0, 2) = 7;
    c_01(1, 0) = 5; c_01(1, 1) = 7; c_01(1, 2) = 9;
  inner_type c_02(range_type{{2, 4}});
    c_02(0, 0) = 5; c_02(0, 1) = 7; c_02(0, 2) = 9; c_02(0, 3) = 11;
    c_02(1, 0) = 7; c_02(1, 1) = 9; c_02(1, 2) = 11; c_02(1, 3) = 13;
  inner_type c_10(range_type{{3, 2}});
    c_10(0, 0) = 3; c_10(0, 1) = 5;
    c_10(1, 0) = 5; c_10(1, 1) = 7;
    c_10(2, 0) = 7; c_10(2, 1) = 9;
  inner_type c_11(range_type{{3, 3}});
    c_11(0, 0) = 5; c_11(0, 1) = 7; c_11(0, 2) = 9;
    c_11(1, 0) = 7; c_11(1, 1) = 9; c_11(1, 2) = 11;
    c_11(2, 0) = 9; c_11(2, 1) = 11; c_11(2, 2) = 13;
  inner_type c_12(range_type{{3, 4}});
    c_12(0, 0) = 7; c_12(0, 1) = 9; c_12(0, 2) = 11; c_12(0, 3) = 13;
    c_12(1, 0) = 9; c_12(1, 1) = 11; c_12(1, 2) = 13; c_12(1, 3) = 15;
    c_12(2, 0) = 11; c_12(2, 1) = 13; c_12(2, 2) = 15; c_12(2, 3)= 17;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i,j;k,l") = lhs("i,j;k,l") + rhs("i,j;k,l");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_result_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{{2, 2}});
    lhs_00(0, 0) = 0; lhs_00(0, 1) = 1;
    lhs_00(1, 0) = 1; lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{{2, 3}});
    lhs_01(0, 0) = 1; lhs_01(0, 1) = 2; lhs_01(0, 2) = 3;
    lhs_01(1, 0) = 2; lhs_01(1, 1) = 3; lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{{2, 4}});
    lhs_02(0, 0) = 2; lhs_02(0, 1) = 3; lhs_02(0, 2) = 4; lhs_02(0, 3) = 5;
    lhs_02(1, 0) = 3; lhs_02(1, 1) = 4; lhs_02(1, 2) = 5; lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{{3, 2}});
    lhs_10(0, 0) = 1; lhs_10(0, 1) = 2;
    lhs_10(1, 0) = 2; lhs_10(1, 1) = 3;
    lhs_10(2, 0) = 3; lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{{3, 3}});
    lhs_11(0, 0) = 2; lhs_11(0, 1) = 3; lhs_11(0, 2) = 4;
    lhs_11(1, 0) = 3; lhs_11(1, 1) = 4; lhs_11(1, 2) = 5;
    lhs_11(2, 0) = 4; lhs_11(2, 1) = 5; lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{{3, 4}});
    lhs_12(0, 0) = 3; lhs_12(0, 1) = 4; lhs_12(0, 2) = 5; lhs_12(0, 3) = 6;
    lhs_12(1, 0) = 4; lhs_12(1, 1) = 5; lhs_12(1, 2) = 6; lhs_12(1, 3) = 7;
    lhs_12(2, 0) = 5; lhs_12(2, 1) = 6; lhs_12(2, 2) = 7; lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{{2, 2}});
    rhs_00(0, 0) = 1; rhs_00(0, 1) = 2;
    rhs_00(1, 0) = 2; rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{{2, 3}});
    rhs_01(0, 0) = 2; rhs_01(0, 1) = 3; rhs_01(0, 2) = 4;
    rhs_01(1, 0) = 3; rhs_01(1, 1) = 4; rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{{2, 4}});
    rhs_02(0, 0) = 3; rhs_02(0, 1) = 4; rhs_02(0, 2) = 5; rhs_02(0, 3) = 6;
    rhs_02(1, 0) = 4; rhs_02(1, 1) = 5; rhs_02(1, 2) = 6; rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{{3, 2}});
    rhs_10(0, 0) = 2; rhs_10(0, 1) = 3;
    rhs_10(1, 0) = 3; rhs_10(1, 1) = 4;
    rhs_10(2, 0) = 4; rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{{3, 3}});
    rhs_11(0, 0) = 3; rhs_11(0, 1) = 4; rhs_11(0, 2) = 5;
    rhs_11(1, 0) = 4; rhs_11(1, 1) = 5; rhs_11(1, 2) = 6;
    rhs_11(2, 0) = 5; rhs_11(2, 1) = 6; rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{{3, 4}});
    rhs_12(0, 0) = 4; rhs_12(0, 1) = 5; rhs_12(0, 2) = 6; rhs_12(0, 3) = 7;
    rhs_12(1, 0) = 5; rhs_12(1, 1) = 6; rhs_12(1, 2) = 7; rhs_12(1, 3) = 8;
    rhs_12(2, 0) = 6; rhs_12(2, 1) = 7; rhs_12(2, 2) = 8; rhs_12(2, 3) = 9;

  inner_type c_00(range_type{{2, 2}});
    c_00(0, 0) = 1; c_00(0, 1) = 3;
    c_00(1, 0) = 3; c_00(1, 1) = 5;
  inner_type c_01(range_type{{2, 3}});
    c_01(0, 0) = 3; c_01(0, 1) = 5; c_01(0, 2) = 7;
    c_01(1, 0) = 5; c_01(1, 1) = 7; c_01(1, 2) = 9;
  inner_type c_02(range_type{{2, 4}});
    c_02(0, 0) = 5; c_02(0, 1) = 7; c_02(0, 2) = 9; c_02(0, 3) = 11;
    c_02(1, 0) = 7; c_02(1, 1) = 9; c_02(1, 2) = 11; c_02(1, 3) = 13;
  inner_type c_10(range_type{{3, 2}});
    c_10(0, 0) = 3; c_10(0, 1) = 5;
    c_10(1, 0) = 5; c_10(1, 1) = 7;
    c_10(2, 0) = 7; c_10(2, 1) = 9;
  inner_type c_11(range_type{{3, 3}});
    c_11(0, 0) = 5; c_11(0, 1) = 7; c_11(0, 2) = 9;
    c_11(1, 0) = 7; c_11(1, 1) = 9; c_11(1, 2) = 11;
    c_11(2, 0) = 9; c_11(2, 1) = 11; c_11(2, 2) = 13;
  inner_type c_12(range_type{{3, 4}});
    c_12(0, 0) = 7; c_12(0, 1) = 9; c_12(0, 2) = 11; c_12(0, 3) = 13;
    c_12(1, 0) = 9; c_12(1, 1) = 11; c_12(1, 2) = 13; c_12(1, 3) = 15;
    c_12(2, 0) = 11; c_12(2, 1) = 13; c_12(2, 2) = 15; c_12(2, 3)= 17;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("j,i;k,l") = lhs("i,j;k,l") + rhs("i,j;k,l");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_SUITE_END()

//------------------------------------------------------------------------------
//                            Subtraction
//------------------------------------------------------------------------------

BOOST_FIXTURE_TEST_SUITE(tot_subtaction, ToTArrayFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(vov, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r2{2}, r3{3};
  inner_type lhs_0{r2}; lhs_0[0] = 0; lhs_0[1] = 1;
  inner_type lhs_1{r3}; lhs_1[0] = 1; lhs_1[1] = 2; lhs_1[2] = 3;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r2}; rhs_0[0] = 1; rhs_0[1] = 2;
  inner_type rhs_1{r3}; rhs_1[0] = 2; rhs_1[1] = 3; rhs_1[2] = 4;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r2}; c_0[0] = -1; c_0[1] = -1;
  inner_type c_1{r3}; c_1[0] = -1; c_1[1] = -1; c_1[2] = -1;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;j") = lhs("i;j") - rhs("i;j");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23}; lhs_0(0, 0) = 0; lhs_0(0, 1) = 1; lhs_0(0, 2) = 2;
                         lhs_0(1, 0) = 1; lhs_0(1, 1) = 2; lhs_0(1, 2) = 3;
  inner_type lhs_1{r33}; lhs_1(0, 0) = 1; lhs_1(0, 1) = 2; lhs_1(0, 2) = 3;
                         lhs_1(1, 0) = 2; lhs_1(1, 1) = 3; lhs_1(1, 2) = 4;
                         lhs_1(2, 0) = 3; lhs_1(2, 1) = 4; lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23}; rhs_0(0, 0) = 1; rhs_0(0, 1) = 2; rhs_0(0, 2) = 3;
                         rhs_0(1, 0) = 2; rhs_0(1, 1) = 3; rhs_0(1, 2) = 4;
  inner_type rhs_1{r33}; rhs_1(0, 0) = 2; rhs_1(0, 1) = 3; rhs_1(0, 2) = 4;
                         rhs_1(1, 0) = 3; rhs_1(1, 1) = 4; rhs_1(1, 2) = 5;
                         rhs_1(2, 0) = 4; rhs_1(2, 1) = 5; rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{r23}; c_0(0, 0) = -1; c_0(0, 1) = -1; c_0(0, 2) = -1;
                       c_0(1, 0) = -1; c_0(1, 1) = -1; c_0(1, 2) = -1;
  inner_type c_1{r33}; c_1(0, 0) = -1; c_1(0, 1) = -1; c_1(0, 2) = -1;
                       c_1(1, 0) = -1; c_1(1, 1) = -1; c_1(1, 2) = -1;
                       c_1(2, 0) = -1; c_1(2, 1) = -1; c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};

  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;j,k") = lhs("i;j,k") - rhs("i;j,k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(vom_result_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::vector_il<inner_type>;
  range_type r23{2, 3}, r33{3, 3};
  inner_type lhs_0{r23}; lhs_0(0, 0) = 0; lhs_0(0, 1) = 1; lhs_0(0, 2) = 2;
                         lhs_0(1, 0) = 1; lhs_0(1, 1) = 2; lhs_0(1, 2) = 3;
  inner_type lhs_1{r33}; lhs_1(0, 0) = 1; lhs_1(0, 1) = 2; lhs_1(0, 2) = 3;
                         lhs_1(1, 0) = 2; lhs_1(1, 1) = 3; lhs_1(1, 2) = 4;
                         lhs_1(2, 0) = 3; lhs_1(2, 1) = 4; lhs_1(2, 2) = 5;
  il_type lhs_il{lhs_0, lhs_1};

  inner_type rhs_0{r23}; rhs_0(0, 0) = 1; rhs_0(0, 1) = 2; rhs_0(0, 2) = 3;
                         rhs_0(1, 0) = 2; rhs_0(1, 1) = 3; rhs_0(1, 2) = 4;
  inner_type rhs_1{r33}; rhs_1(0, 0) = 2; rhs_1(0, 1) = 3; rhs_1(0, 2) = 4;
                         rhs_1(1, 0) = 3; rhs_1(1, 1) = 4; rhs_1(1, 2) = 5;
                         rhs_1(2, 0) = 4; rhs_1(2, 1) = 5; rhs_1(2, 2) = 6;
  il_type rhs_il{rhs_0, rhs_1};

  inner_type c_0{range_type{3, 2}}; c_0(0, 0) = -1; c_0(0, 1) = -1;
                                    c_0(1, 0) = -1; c_0(1, 1) = -1;
                                    c_0(2, 0) = -1; c_0(2, 1) = -1;
  inner_type c_1{r33}; c_1(0, 0) = -1; c_1(0, 1) = -1; c_1(0, 2) = -1;
                       c_1(1, 0) = -1; c_1(1, 1) = -1; c_1(1, 2) = -1;
                       c_1(2, 0) = -1; c_1(2, 1) = -1; c_1(2, 2) = -1;
  il_type corr_il{c_0, c_1};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;k,j") = lhs("i;j,k") - rhs("i;j,k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;

  inner_type lhs_00(range_type{2}); lhs_00[0] = 0; lhs_00[1] = 1;
  inner_type lhs_01(range_type{2}); lhs_01[0] = 1; lhs_01[1] = 2;
  inner_type lhs_02(range_type{2}); lhs_02[0] = 2; lhs_02[1] = 3;
  inner_type lhs_10(range_type{3}); lhs_10[0] = 1; lhs_10[1] = 2; lhs_10[2] = 3;
  inner_type lhs_11(range_type{3}); lhs_11[0] = 2; lhs_11[1] = 3; lhs_11[2] = 4;
  inner_type lhs_12(range_type{3}); lhs_12[0] = 3; lhs_12[1] = 4; lhs_12[2] = 5;

  inner_type rhs_02(range_type{2}); rhs_02[0] = 3; rhs_02[1] = 4;
  inner_type rhs_12(range_type{3}); rhs_12[0] = 4; rhs_12[1] = 5; rhs_12[2] = 6;

  inner_type c_00(range_type{2}); c_00[0] = -1; c_00[1] = -1;
  inner_type c_01(range_type{2}); c_01[0] = -1; c_01[1] = -1;
  inner_type c_02(range_type{2}); c_02[0] = -1; c_02[1] = -1;
  inner_type c_10(range_type{3}); c_10[0] = -1; c_10[1] = -1; c_10[2] = -1;
  inner_type c_11(range_type{3}); c_11[0] = -1; c_11[1] = -1; c_11[2] = -1;
  inner_type c_12(range_type{3}); c_12[0] = -1; c_12[1] = -1; c_12[2] = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i,j;k") = lhs("i,j;k") - rhs("i,j;k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov_result_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2}); lhs_00[0] = 0; lhs_00[1] = 1;
  inner_type lhs_01(range_type{2}); lhs_01[0] = 1; lhs_01[1] = 2;
  inner_type lhs_02(range_type{2}); lhs_02[0] = 2; lhs_02[1] = 3;
  inner_type lhs_10(range_type{3}); lhs_10[0] = 1; lhs_10[1] = 2; lhs_10[2] = 3;
  inner_type lhs_11(range_type{3}); lhs_11[0] = 2; lhs_11[1] = 3; lhs_11[2] = 4;
  inner_type lhs_12(range_type{3}); lhs_12[0] = 3; lhs_12[1] = 4; lhs_12[2] = 5;

  inner_type rhs_02(range_type{2}); rhs_02[0] = 3; rhs_02[1] = 4;
  inner_type rhs_12(range_type{3}); rhs_12[0] = 4; rhs_12[1] = 5; rhs_12[2] = 6;

  inner_type c_00(range_type{2}); c_00[0] = -1; c_00[1] = -1;
  inner_type c_01(range_type{2}); c_01[0] = -1; c_01[1] = -1;
  inner_type c_02(range_type{2}); c_02[0] = -1; c_02[1] = -1;
  inner_type c_10(range_type{3}); c_10[0] = -1; c_10[1] = -1; c_10[2] = -1;
  inner_type c_11(range_type{3}); c_11[0] = -1; c_11[1] = -1; c_11[2] = -1;
  inner_type c_12(range_type{3}); c_12[0] = -1; c_12[1] = -1; c_12[2] = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{lhs_01, lhs_02, rhs_02}, {lhs_11, lhs_12, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("j,i;k") = lhs("i,j;k") - rhs("i,j;k");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{{2, 2}});
    lhs_00(0, 0) = 0; lhs_00(0, 1) = 1;
    lhs_00(1, 0) = 1; lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{{2, 3}});
    lhs_01(0, 0) = 1; lhs_01(0, 1) = 2; lhs_01(0, 2) = 3;
    lhs_01(1, 0) = 2; lhs_01(1, 1) = 3; lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{{2, 4}});
    lhs_02(0, 0) = 2; lhs_02(0, 1) = 3; lhs_02(0, 2) = 4; lhs_02(0, 3) = 5;
    lhs_02(1, 0) = 3; lhs_02(1, 1) = 4; lhs_02(1, 2) = 5; lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{{3, 2}});
    lhs_10(0, 0) = 1; lhs_10(0, 1) = 2;
    lhs_10(1, 0) = 2; lhs_10(1, 1) = 3;
    lhs_10(2, 0) = 3; lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{{3, 3}});
    lhs_11(0, 0) = 2; lhs_11(0, 1) = 3; lhs_11(0, 2) = 4;
    lhs_11(1, 0) = 3; lhs_11(1, 1) = 4; lhs_11(1, 2) = 5;
    lhs_11(2, 0) = 4; lhs_11(2, 1) = 5; lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{{3, 4}});
    lhs_12(0, 0) = 3; lhs_12(0, 1) = 4; lhs_12(0, 2) = 5; lhs_12(0, 3) = 6;
    lhs_12(1, 0) = 4; lhs_12(1, 1) = 5; lhs_12(1, 2) = 6; lhs_12(1, 3) = 7;
    lhs_12(2, 0) = 5; lhs_12(2, 1) = 6; lhs_12(2, 2) = 7; lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{{2, 2}});
    rhs_00(0, 0) = 1; rhs_00(0, 1) = 2;
    rhs_00(1, 0) = 2; rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{{2, 3}});
    rhs_01(0, 0) = 2; rhs_01(0, 1) = 3; rhs_01(0, 2) = 4;
    rhs_01(1, 0) = 3; rhs_01(1, 1) = 4; rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{{2, 4}});
    rhs_02(0, 0) = 3; rhs_02(0, 1) = 4; rhs_02(0, 2) = 5; rhs_02(0, 3) = 6;
    rhs_02(1, 0) = 4; rhs_02(1, 1) = 5; rhs_02(1, 2) = 6; rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{{3, 2}});
    rhs_10(0, 0) = 2; rhs_10(0, 1) = 3;
    rhs_10(1, 0) = 3; rhs_10(1, 1) = 4;
    rhs_10(2, 0) = 4; rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{{3, 3}});
    rhs_11(0, 0) = 3; rhs_11(0, 1) = 4; rhs_11(0, 2) = 5;
    rhs_11(1, 0) = 4; rhs_11(1, 1) = 5; rhs_11(1, 2) = 6;
    rhs_11(2, 0) = 5; rhs_11(2, 1) = 6; rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{{3, 4}});
    rhs_12(0, 0) = 4; rhs_12(0, 1) = 5; rhs_12(0, 2) = 6; rhs_12(0, 3) = 7;
    rhs_12(1, 0) = 5; rhs_12(1, 1) = 6; rhs_12(1, 2) = 7; rhs_12(1, 3) = 8;
    rhs_12(2, 0) = 6; rhs_12(2, 1) = 7; rhs_12(2, 2) = 8; rhs_12(2, 3) = 9;

  inner_type c_00(range_type{{2, 2}});
    c_00(0, 0) = -1; c_00(0, 1) = -1;
    c_00(1, 0) = -1; c_00(1, 1) = -1;
  inner_type c_01(range_type{{2, 3}});
    c_01(0, 0) = -1; c_01(0, 1) = -1; c_01(0, 2) = -1;
    c_01(1, 0) = -1; c_01(1, 1) = -1; c_01(1, 2) = -1;
  inner_type c_02(range_type{{2, 4}});
    c_02(0, 0) = -1; c_02(0, 1) = -1; c_02(0, 2) = -1; c_02(0, 3) = -1;
    c_02(1, 0) = -1; c_02(1, 1) = -1; c_02(1, 2) = -1; c_02(1, 3) = -1;
  inner_type c_10(range_type{{3, 2}});
    c_10(0, 0) = -1; c_10(0, 1) = -1;
    c_10(1, 0) = -1; c_10(1, 1) = -1;
    c_10(2, 0) = -1; c_10(2, 1) = -1;
  inner_type c_11(range_type{{3, 3}});
    c_11(0, 0) = -1; c_11(0, 1) = -1; c_11(0, 2) = -1;
    c_11(1, 0) = -1; c_11(1, 1) = -1; c_11(1, 2) = -1;
    c_11(2, 0) = -1; c_11(2, 1) = -1; c_11(2, 2) = -1;
  inner_type c_12(range_type{{3, 4}});
    c_12(0, 0) = -1; c_12(0, 1) = -1; c_12(0, 2) = -1; c_12(0, 3) = -1;
    c_12(1, 0) = -1; c_12(1, 1) = -1; c_12(1, 2) = -1; c_12(1, 3) = -1;
    c_12(2, 0) = -1; c_12(2, 1) = -1; c_12(2, 2) = -1; c_12(2, 3) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_01, c_02}, {c_10, c_11, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i,j;k,l") = lhs("i,j;k,l") - rhs("i,j;k,l");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mom_result_transpose, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{{2, 2}});
    lhs_00(0, 0) = 0; lhs_00(0, 1) = 1;
    lhs_00(1, 0) = 1; lhs_00(1, 1) = 2;
  inner_type lhs_01(range_type{{2, 3}});
    lhs_01(0, 0) = 1; lhs_01(0, 1) = 2; lhs_01(0, 2) = 3;
    lhs_01(1, 0) = 2; lhs_01(1, 1) = 3; lhs_01(1, 2) = 4;
  inner_type lhs_02(range_type{{2, 4}});
    lhs_02(0, 0) = 2; lhs_02(0, 1) = 3; lhs_02(0, 2) = 4; lhs_02(0, 3) = 5;
    lhs_02(1, 0) = 3; lhs_02(1, 1) = 4; lhs_02(1, 2) = 5; lhs_02(1, 3) = 6;
  inner_type lhs_10(range_type{{3, 2}});
    lhs_10(0, 0) = 1; lhs_10(0, 1) = 2;
    lhs_10(1, 0) = 2; lhs_10(1, 1) = 3;
    lhs_10(2, 0) = 3; lhs_10(2, 1) = 4;
  inner_type lhs_11(range_type{{3, 3}});
    lhs_11(0, 0) = 2; lhs_11(0, 1) = 3; lhs_11(0, 2) = 4;
    lhs_11(1, 0) = 3; lhs_11(1, 1) = 4; lhs_11(1, 2) = 5;
    lhs_11(2, 0) = 4; lhs_11(2, 1) = 5; lhs_11(2, 2) = 6;
  inner_type lhs_12(range_type{{3, 4}});
    lhs_12(0, 0) = 3; lhs_12(0, 1) = 4; lhs_12(0, 2) = 5; lhs_12(0, 3) = 6;
    lhs_12(1, 0) = 4; lhs_12(1, 1) = 5; lhs_12(1, 2) = 6; lhs_12(1, 3) = 7;
    lhs_12(2, 0) = 5; lhs_12(2, 1) = 6; lhs_12(2, 2) = 7; lhs_12(2, 3) = 8;

  inner_type rhs_00(range_type{{2, 2}});
    rhs_00(0, 0) = 1; rhs_00(0, 1) = 2;
    rhs_00(1, 0) = 2; rhs_00(1, 1) = 3;
  inner_type rhs_01(range_type{{2, 3}});
    rhs_01(0, 0) = 2; rhs_01(0, 1) = 3; rhs_01(0, 2) = 4;
    rhs_01(1, 0) = 3; rhs_01(1, 1) = 4; rhs_01(1, 2) = 5;
  inner_type rhs_02(range_type{{2, 4}});
    rhs_02(0, 0) = 3; rhs_02(0, 1) = 4; rhs_02(0, 2) = 5; rhs_02(0, 3) = 6;
    rhs_02(1, 0) = 4; rhs_02(1, 1) = 5; rhs_02(1, 2) = 6; rhs_02(1, 3) = 7;
  inner_type rhs_10(range_type{{3, 2}});
    rhs_10(0, 0) = 2; rhs_10(0, 1) = 3;
    rhs_10(1, 0) = 3; rhs_10(1, 1) = 4;
    rhs_10(2, 0) = 4; rhs_10(2, 1) = 5;
  inner_type rhs_11(range_type{{3, 3}});
    rhs_11(0, 0) = 3; rhs_11(0, 1) = 4; rhs_11(0, 2) = 5;
    rhs_11(1, 0) = 4; rhs_11(1, 1) = 5; rhs_11(1, 2) = 6;
    rhs_11(2, 0) = 5; rhs_11(2, 1) = 6; rhs_11(2, 2) = 7;
  inner_type rhs_12(range_type{{3, 4}});
    rhs_12(0, 0) = 4; rhs_12(0, 1) = 5; rhs_12(0, 2) = 6; rhs_12(0, 3) = 7;
    rhs_12(1, 0) = 5; rhs_12(1, 1) = 6; rhs_12(1, 2) = 7; rhs_12(1, 3) = 8;
    rhs_12(2, 0) = 6; rhs_12(2, 1) = 7; rhs_12(2, 2) = 8; rhs_12(2, 3) = 9;

  inner_type c_00(range_type{{2, 2}});
    c_00(0, 0) = -1; c_00(0, 1) = -1;
    c_00(1, 0) = -1; c_00(1, 1) = -1;
  inner_type c_01(range_type{{2, 3}});
    c_01(0, 0) = -1; c_01(0, 1) = -1; c_01(0, 2) = -1;
    c_01(1, 0) = -1; c_01(1, 1) = -1; c_01(1, 2) = -1;
  inner_type c_02(range_type{{2, 4}});
    c_02(0, 0) = -1; c_02(0, 1) = -1; c_02(0, 2) = -1; c_02(0, 3) = -1;
    c_02(1, 0) = -1; c_02(1, 1) = -1; c_02(1, 2) = -1; c_02(1, 3) = -1;
  inner_type c_10(range_type{{3, 2}});
    c_10(0, 0) = -1; c_10(0, 1) = -1;
    c_10(1, 0) = -1; c_10(1, 1) = -1;
    c_10(2, 0) = -1; c_10(2, 1) = -1;
  inner_type c_11(range_type{{3, 3}});
    c_11(0, 0) = -1; c_11(0, 1) = -1; c_11(0, 2) = -1;
    c_11(1, 0) = -1; c_11(1, 1) = -1; c_11(1, 2) = -1;
    c_11(2, 0) = -1; c_11(2, 1) = -1; c_11(2, 2) = -1;
  inner_type c_12(range_type{{3, 4}});
    c_12(0, 0) = -1; c_12(0, 1) = -1; c_12(0, 2) = -1; c_12(0, 3) = -1;
    c_12(1, 0) = -1; c_12(1, 1) = -1; c_12(1, 2) = -1; c_12(1, 3) = -1;
    c_12(2, 0) = -1; c_12(2, 1) = -1; c_12(2, 2) = -1; c_12(2, 3) = -1;

  il_type lhs_il{{lhs_00, lhs_01, lhs_02}, {lhs_10, lhs_11, lhs_12}};
  il_type rhs_il{{rhs_00, rhs_01, rhs_02}, {rhs_10, rhs_11, rhs_12}};
  il_type corr_il{{c_00, c_10}, {c_01, c_11}, {c_02, c_12}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("j,i;k,l") = lhs("i,j;k,l") - rhs("i,j;k,l");
  are_equal(result, corr);
}

BOOST_AUTO_TEST_SUITE_END()
