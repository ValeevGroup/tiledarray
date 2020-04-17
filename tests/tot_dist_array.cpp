/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "tot_array_fixture.h"

BOOST_FIXTURE_TEST_SUITE(tot_array_suite2, ToTArrayFixture)
//------------------------------------------------------------------------------
//                       Fill and Initialize
//------------------------------------------------------------------------------

/* fill_local is a thin wrapper around init_tiles, which initializes each
 * element of a tile to single value.
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(fill_local, TestParam, test_params){
  using tensor_type = tensor_type<TestParam>;
  using inner_type = inner_type<TestParam>;
  using except_t = TiledArray::Exception;
  // Throws if PIMPL is empty
  {
    tensor_type t;
    if(m_world.nproc() == 1) {
      BOOST_CHECK_THROW(t.fill_local(inner_type{}), except_t);
    }
  }

  for(auto tr_t : run_all<TestParam>()){
    auto& tr = std::get<0>(tr_t);
    auto inner_rank = std::get<1>(tr_t);
    auto& already_set = std::get<2>(tr_t);

    // Test that it skips filled tiles
    {
      auto corr = already_set.clone();
      already_set.fill_local(inner_type{}, true);
      BOOST_CHECK(are_equal(corr, already_set));
    }

    // Test that it throws if a tile is already set
    {
        if(m_world.nproc() == 1)
          BOOST_CHECK_THROW(already_set.fill_local(inner_type{}), except_t);
    }

    // Test we can actually fill tiles
    {
      tensor_type t(m_world, tr);
      tensor_type corr(m_world, tr);
      auto start_idx = tr.tiles_range().lobound();
      auto tile = inner_rank == 1 ? inner_vector_tile<TestParam>(start_idx)
                                  : inner_matrix_tile<TestParam>(start_idx);
      for (auto idx : tr.tiles_range()) {
        if (!t.is_local(idx)) continue;
        tile_type<TestParam> outer_tile(tr.tile(idx), tile);
        corr.set(idx, outer_tile);
      }
      t.fill_local(tile);
      BOOST_CHECK(are_equal(t, corr));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fill, TestParam, test_params) {

}

BOOST_AUTO_TEST_CASE_TEMPLATE(fill_random, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  //t1.fill_random();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(init_tiles, TestParam, test_params) {

}

BOOST_AUTO_TEST_CASE_TEMPLATE(init_elements, TestParam, test_params) {

}

//------------------------------------------------------------------------------
//                     Accessors
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(trange, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()) {
    tensor_type<TestParam> t1(m_world, tr);
    BOOST_TEST(t1.trange() == tr);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(range, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()) {
    tensor_type<TestParam> t1(m_world, tr);
    BOOST_TEST(t1.range() == tr.tiles_range());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(elements_range, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()) {
    tensor_type<TestParam> t1(m_world, tr);
    BOOST_TEST(t1.elements_range() == tr.elements_range());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(size, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()) {
   tensor_type<TestParam> t1(m_world, tr);
   BOOST_TEST(t1.size() == tr.elements_range().volume());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(world, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    BOOST_TEST(&t1.world() == &m_world);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pmap, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.pmap();
  }
}

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE_TEMPLATE(call_operator, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    //auto expr = t1("i;j");
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(const_call_operator, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    //auto expr = std::as_const(t1)("i;j");
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE(is_dense, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    BOOST_TEST(t1.is_dense() == false);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(shape, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.shape();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owner, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.owner(0);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(owner_init_list, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.owner({0});
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_local, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.is_local(0);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_local_init_list, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.owner({0});
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_zero, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.is_zero(0);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_zero_init_list, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.owner({0});
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(swap, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    tensor_type<TestParam> t2;
    t1.swap(t2);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(make_replicated, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.make_replicated();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(truncate, TestParam, test_params){
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    t1.truncate();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(is_initialized, TestParam, test_params){
  // Not initialized
  {
    tensor_type<TestParam> t1;
    BOOST_TEST(t1.is_initialized() == false);
  }

  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    BOOST_TEST(t1.is_initialized());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(serialization, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    auto file_name = "a_file.temp";
    {
      output_archive_type ar_out(file_name);
      t1.serialize(ar_out);
    }

    tensor_type<TestParam> t2;
    {
      input_archive_type ar_in(file_name);
      t2.serialize(ar_in);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(load_store, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()){
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    auto file_name = "a_file.temp";
    {
      madness::archive::ParallelOutputArchive ar_out(m_world, file_name, 1);
      t1.store(ar_out);
    }

    tensor_type<TestParam> t2;
    {
      madness::archive::ParallelInputArchive ar_in(m_world, file_name, 1);
      t2.load(m_world, ar_in);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(printing, TestParam, test_params) {
  for(auto tr : vector_tiled_ranges<TestParam>()) {
    tensor_type<TestParam> t1 = tensor_of_vector<TestParam>(tr);
    std::stringstream ss;
    //ss << t1;
  }
}

BOOST_AUTO_TEST_SUITE_END()
