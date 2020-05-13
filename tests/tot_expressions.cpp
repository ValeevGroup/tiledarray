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
  il_type lhs_il{inner_type{range_type{2}, {0, 1}},
                 inner_type{range_type{3}, {1, 2, 3}}};
  il_type rhs_il{inner_type{range_type{2}, {1, 2}},
                 inner_type{range_type{3}, {2, 3, 4}}};
  il_type corr_il{inner_type{range_type{2}, {1,3}},
                  inner_type{range_type{3}, {3, 5, 7}}};
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
  il_type lhs_il{inner_type{range_type{{2, 3}},
                            {0, 1, 2, 1, 2, 3}},
                 inner_type{range_type{{3, 3}}, {1, 2, 3, 2, 3, 4, 3, 4, 5}}};
  il_type rhs_il{inner_type{range_type{{2, 3}}, {1, 2, 3, 2, 3, 4}},
                 inner_type{range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6}}};
  il_type corr_il{inner_type{range_type{{2, 3}}, {1, 3, 5, 3, 5, 7}},
                  inner_type{range_type{{3, 3}}, {3, 5, 7, 5, 7, 9, 7, 9, 11}}};
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
  il_type lhs_il{inner_type{range_type{{2, 3}},
                            {0, 1, 2, 1, 2, 3}},
                 inner_type{range_type{{3, 3}}, {1, 2, 3, 2, 3, 4, 3, 4, 5}}};
  il_type rhs_il{inner_type{range_type{{2, 3}}, {1, 2, 3, 2, 3, 4}},
                 inner_type{range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6}}};
  il_type corr_il{inner_type{range_type{{3, 2}}, {1, 3, 3, 5, 5, 7}},
                  inner_type{range_type{{3, 3}}, {3, 5, 7, 5, 7, 9, 7, 9, 11}}};
  tensor_type<TestParam> lhs(m_world, lhs_il);
  tensor_type<TestParam> rhs(m_world, rhs_il);
  tensor_type<TestParam> corr(m_world, corr_il);
  tensor_type<TestParam> result;
  result ("i;k,j") = lhs("i;j,k") + rhs("i;j,k");
  are_equal(result, corr);
}

//BOOST_AUTO_TEST_CASE_TEMPLATE(vom_lhs_transpose, TestParam, test_params){
//  using inner_type = inner_type<TestParam>;
//  using range_type = Range;
//  using il_type    = detail::vector_il<inner_type>;
//  il_type lhs_il{inner_type{range_type{{3, 2}},
//                            {0, 1, 1, 2, 2, 3}},
//                 inner_type{range_type{{3, 3}}, {1, 2, 3, 2, 3, 4, 3, 4, 5}}};
//  il_type rhs_il{inner_type{range_type{{2, 3}}, {1, 2, 3, 2, 3, 4}},
//                 inner_type{range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6}}};
//  il_type corr_il{inner_type{range_type{{2, 3}}, {1, 3, 5, 3, 5, 7}},
//                  inner_type{range_type{{3, 3}}, {3, 5, 7, 5, 7, 9, 7, 9, 11}}};
//  tensor_type<TestParam> lhs(m_world, lhs_il);
//  tensor_type<TestParam> rhs(m_world, rhs_il);
//  tensor_type<TestParam> corr(m_world, corr_il);
//  tensor_type<TestParam> result;
//  result ("i;j,k") = lhs("i;k,j") + rhs("i;j,k");
//  are_equal(result, corr);
//}

BOOST_AUTO_TEST_CASE_TEMPLATE(mov, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  using range_type = Range;
  using il_type    = detail::matrix_il<inner_type>;
  inner_type lhs_00(range_type{2}, {0, 1});
  inner_type lhs_01(range_type{2}, {1, 2});
  inner_type lhs_02(range_type{2}, {2, 3});
  inner_type lhs_10(range_type{3}, {1, 2, 3});
  inner_type lhs_11(range_type{3}, {2, 3, 4});
  inner_type lhs_12(range_type{3}, {3, 4, 5});

  inner_type rhs_02(range_type{2}, {3, 4});
  inner_type rhs_12(range_type{3}, {4, 5, 6});

  inner_type c_00(range_type{2}, {1, 3});
  inner_type c_01(range_type{2}, {3, 5});
  inner_type c_02(range_type{2}, {5, 7});
  inner_type c_10(range_type{3}, {3, 5, 7});
  inner_type c_11(range_type{3}, {5, 7, 9});
  inner_type c_12(range_type{3}, {7, 9, 11});

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
  inner_type lhs_00(range_type{2}, {0, 1});
  inner_type lhs_01(range_type{2}, {1, 2});
  inner_type lhs_02(range_type{2}, {2, 3});
  inner_type lhs_10(range_type{3}, {1, 2, 3});
  inner_type lhs_11(range_type{3}, {2, 3, 4});
  inner_type lhs_12(range_type{3}, {3, 4, 5});

  inner_type rhs_02(range_type{2}, {3, 4});
  inner_type rhs_12(range_type{3}, {4, 5, 6});

  inner_type c_00(range_type{2}, {1, 3});
  inner_type c_01(range_type{2}, {3, 5});
  inner_type c_02(range_type{2}, {5, 7});
  inner_type c_10(range_type{3}, {3, 5, 7});
  inner_type c_11(range_type{3}, {5, 7, 9});
  inner_type c_12(range_type{3}, {7, 9, 11});

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
  inner_type lhs_00(range_type{{2, 2}}, {0, 1, 1, 2});
  inner_type lhs_01(range_type{{2, 3}}, {1, 2, 3, 2, 3, 4});
  inner_type lhs_02(range_type{{2, 4}}, {2, 3, 4, 5, 3, 4, 5, 6});
  inner_type lhs_10(range_type{{3, 2}}, {1, 2, 2, 3, 3, 4});
  inner_type lhs_11(range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6});
  inner_type lhs_12(range_type{{3, 4}}, {3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8});

  inner_type rhs_00(range_type{{2, 2}}, {1, 2, 2, 3});
  inner_type rhs_01(range_type{{2, 3}}, {2, 3, 4, 3, 4, 5});
  inner_type rhs_02(range_type{{2, 4}}, {3, 4, 5, 6, 4, 5, 6, 7});
  inner_type rhs_10(range_type{{3, 2}}, {2, 3, 3, 4, 4, 5});
  inner_type rhs_11(range_type{{3, 3}}, {3, 4, 5, 4, 5, 6, 5, 6, 7});
  inner_type rhs_12(range_type{{3, 4}}, {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9});

  inner_type c_00(range_type{{2, 2}}, {1, 3, 3, 5});
  inner_type c_01(range_type{{2, 3}}, {3, 5, 7, 5, 7, 9});
  inner_type c_02(range_type{{2, 4}}, {5, 7, 9, 11, 7, 9, 11, 13});
  inner_type c_10(range_type{{3, 2}}, {3, 5, 5, 7, 7, 9});
  inner_type c_11(range_type{{3, 3}}, {5, 7, 9, 7, 9, 11, 9, 11, 13});
  inner_type c_12(range_type{{3, 4}}, {7, 9, 11, 13, 9, 11, 13, 15, 11, 13, 15, 17});

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
  inner_type lhs_00(range_type{{2, 2}}, {0, 1, 1, 2});
  inner_type lhs_01(range_type{{2, 3}}, {1, 2, 3, 2, 3, 4});
  inner_type lhs_02(range_type{{2, 4}}, {2, 3, 4, 5, 3, 4, 5, 6});
  inner_type lhs_10(range_type{{3, 2}}, {1, 2, 2, 3, 3, 4});
  inner_type lhs_11(range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6});
  inner_type lhs_12(range_type{{3, 4}}, {3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8});

  inner_type rhs_00(range_type{{2, 2}}, {1, 2, 2, 3});
  inner_type rhs_01(range_type{{2, 3}}, {2, 3, 4, 3, 4, 5});
  inner_type rhs_02(range_type{{2, 4}}, {3, 4, 5, 6, 4, 5, 6, 7});
  inner_type rhs_10(range_type{{3, 2}}, {2, 3, 3, 4, 4, 5});
  inner_type rhs_11(range_type{{3, 3}}, {3, 4, 5, 4, 5, 6, 5, 6, 7});
  inner_type rhs_12(range_type{{3, 4}}, {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9});

  inner_type c_00(range_type{{2, 2}}, {1, 3, 3, 5});
  inner_type c_01(range_type{{2, 3}}, {3, 5, 7, 5, 7, 9});
  inner_type c_02(range_type{{2, 4}}, {5, 7, 9, 11, 7, 9, 11, 13});
  inner_type c_10(range_type{{3, 2}}, {3, 5, 5, 7, 7, 9});
  inner_type c_11(range_type{{3, 3}}, {5, 7, 9, 7, 9, 11, 9, 11, 13});
  inner_type c_12(range_type{{3, 4}}, {7, 9, 11, 13, 9, 11, 13, 15, 11, 13, 15, 17});

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
  il_type lhs_il{inner_type{range_type{2}, {0, 1}},
                 inner_type{range_type{3}, {1, 2, 3}}};
  il_type rhs_il{inner_type{range_type{2}, {1, 2}},
                 inner_type{range_type{3}, {2, 3, 4}}};
  il_type corr_il{inner_type{range_type{2}, {-1, -1}},
                  inner_type{range_type{3}, {-1, -1, -1}}};
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
  il_type lhs_il{inner_type{range_type{{2, 3}},
                            {0, 1, 2, 1, 2, 3}},
                 inner_type{range_type{{3, 3}}, {1, 2, 3, 2, 3, 4, 3, 4, 5}}};
  il_type rhs_il{inner_type{range_type{{2, 3}}, {1, 2, 3, 2, 3, 4}},
                 inner_type{range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6}}};
  il_type corr_il{inner_type{range_type{{2, 3}}, {-1, -1, -1, -1, -1, -1}},
                  inner_type{range_type{{3, 3}}, {-1, -1, -1, -1, -1, -1, -1, -1, -1}}};
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
  il_type lhs_il{inner_type{range_type{{2, 3}},
                            {0, 1, 2, 1, 2, 3}},
                 inner_type{range_type{{3, 3}}, {1, 2, 3, 2, 3, 4, 3, 4, 5}}};
  il_type rhs_il{inner_type{range_type{{2, 3}}, {1, 2, 3, 2, 3, 4}},
                 inner_type{range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6}}};
  il_type corr_il{inner_type{range_type{{3, 2}}, {-1, -1, -1, -1, -1, -1}},
                  inner_type{range_type{{3, 3}}, {-1, -1, -1, -1, -1, -1, -1, -1, -1}}};
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
  inner_type lhs_00(range_type{2}, {0, 1});
  inner_type lhs_01(range_type{2}, {1, 2});
  inner_type lhs_02(range_type{2}, {2, 3});
  inner_type lhs_10(range_type{3}, {1, 2, 3});
  inner_type lhs_11(range_type{3}, {2, 3, 4});
  inner_type lhs_12(range_type{3}, {3, 4, 5});

  inner_type rhs_02(range_type{2}, {3, 4});
  inner_type rhs_12(range_type{3}, {4, 5, 6});

  inner_type c_00(range_type{2}, {-1, -1});
  inner_type c_01(range_type{2}, {-1, -1});
  inner_type c_02(range_type{2}, {-1, -1});
  inner_type c_10(range_type{3}, {-1, -1, -1});
  inner_type c_11(range_type{3}, {-1, -1, -1});
  inner_type c_12(range_type{3}, {-1, -1, -1});

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
  inner_type lhs_00(range_type{2}, {0, 1});
  inner_type lhs_01(range_type{2}, {1, 2});
  inner_type lhs_02(range_type{2}, {2, 3});
  inner_type lhs_10(range_type{3}, {1, 2, 3});
  inner_type lhs_11(range_type{3}, {2, 3, 4});
  inner_type lhs_12(range_type{3}, {3, 4, 5});

  inner_type rhs_02(range_type{2}, {3, 4});
  inner_type rhs_12(range_type{3}, {4, 5, 6});

  inner_type c_00(range_type{2}, {-1, -1});
  inner_type c_01(range_type{2}, {-1, -1});
  inner_type c_02(range_type{2}, {-1, -1});
  inner_type c_10(range_type{3}, {-1, -1, -1});
  inner_type c_11(range_type{3}, {-1, -1, -1});
  inner_type c_12(range_type{3}, {-1, -1, -1});

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
  inner_type lhs_00(range_type{{2, 2}}, {0, 1, 1, 2});
  inner_type lhs_01(range_type{{2, 3}}, {1, 2, 3, 2, 3, 4});
  inner_type lhs_02(range_type{{2, 4}}, {2, 3, 4, 5, 3, 4, 5, 6});
  inner_type lhs_10(range_type{{3, 2}}, {1, 2, 2, 3, 3, 4});
  inner_type lhs_11(range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6});
  inner_type lhs_12(range_type{{3, 4}}, {3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8});

  inner_type rhs_00(range_type{{2, 2}}, {1, 2, 2, 3});
  inner_type rhs_01(range_type{{2, 3}}, {2, 3, 4, 3, 4, 5});
  inner_type rhs_02(range_type{{2, 4}}, {3, 4, 5, 6, 4, 5, 6, 7});
  inner_type rhs_10(range_type{{3, 2}}, {2, 3, 3, 4, 4, 5});
  inner_type rhs_11(range_type{{3, 3}}, {3, 4, 5, 4, 5, 6, 5, 6, 7});
  inner_type rhs_12(range_type{{3, 4}}, {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9});

  inner_type c_00(range_type{{2, 2}}, {-1, -1, -1, -1});
  inner_type c_01(range_type{{2, 3}}, {-1, -1, -1, -1, -1, -1});
  inner_type c_02(range_type{{2, 4}}, {-1, -1, -1, -1, -1, -1, -1, -1});
  inner_type c_10(range_type{{3, 2}}, {-1, -1, -1, -1, -1, -1});
  inner_type c_11(range_type{{3, 3}}, {-1, -1, -1, -1, -1, -1, -1, -1, -1});
  inner_type c_12(range_type{{3, 4}}, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});

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
  inner_type lhs_00(range_type{{2, 2}}, {0, 1, 1, 2});
  inner_type lhs_01(range_type{{2, 3}}, {1, 2, 3, 2, 3, 4});
  inner_type lhs_02(range_type{{2, 4}}, {2, 3, 4, 5, 3, 4, 5, 6});
  inner_type lhs_10(range_type{{3, 2}}, {1, 2, 2, 3, 3, 4});
  inner_type lhs_11(range_type{{3, 3}}, {2, 3, 4, 3, 4, 5, 4, 5, 6});
  inner_type lhs_12(range_type{{3, 4}}, {3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8});

  inner_type rhs_00(range_type{{2, 2}}, {1, 2, 2, 3});
  inner_type rhs_01(range_type{{2, 3}}, {2, 3, 4, 3, 4, 5});
  inner_type rhs_02(range_type{{2, 4}}, {3, 4, 5, 6, 4, 5, 6, 7});
  inner_type rhs_10(range_type{{3, 2}}, {2, 3, 3, 4, 4, 5});
  inner_type rhs_11(range_type{{3, 3}}, {3, 4, 5, 4, 5, 6, 5, 6, 7});
  inner_type rhs_12(range_type{{3, 4}}, {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9});

  inner_type c_00(range_type{{2, 2}}, {-1, -1, -1, -1});
  inner_type c_01(range_type{{2, 3}}, {-1, -1, -1, -1, -1, -1});
  inner_type c_02(range_type{{2, 4}}, {-1, -1, -1, -1, -1, -1, -1, -1});
  inner_type c_10(range_type{{3, 2}}, {-1, -1, -1, -1, -1, -1});
  inner_type c_11(range_type{{3, 3}}, {-1, -1, -1, -1, -1, -1, -1, -1, -1});
  inner_type c_12(range_type{{3, 4}}, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});

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
