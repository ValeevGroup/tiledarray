#include "tot_array_fixture.h"
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

    std::cout << t << std::endl;
    std::cout << result << std::endl;

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


BOOST_AUTO_TEST_SUITE_END()
