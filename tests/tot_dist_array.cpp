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
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

// These are all of the template parameters we are going to test over
using test_params = boost::mpl::list<
    std::tuple<int, Tensor<Tensor<int>>>,
    std::tuple<float, Tensor<Tensor<float>>>,
    std::tuple<double, Tensor<Tensor<double>>>
>;

// These typedefs unpack the unit test template parameter
//{
template<typename TupleElementType>
using scalar_type = std::tuple_element_t<0, TupleElementType>;

template<typename TupleElementType>
using tile_type = std::tuple_element_t<1, TupleElementType>;

template<typename>
using policy_type = DensePolicy;
//}

// The type of a DistArray consistent with the unit test template parameter
template<typename TupleElementType>
using tensor_type =
    DistArray<tile_type<TupleElementType>, policy_type<TupleElementType>>;

// Type of the object storing the tiling
template<typename TupleElementType>
using trange_type = typename tensor_type<TupleElementType>::trange_type;

// Type of the inner tile
template<typename TupleElementType>
using inner_type = typename tile_type<TupleElementType>::value_type;

namespace {

/*
 *
 * When generating arrays containing tensors of tensors (ToT) we adopt simple
 * algorithms for initialing the values. These algorithms map the outer indices
 * to the values of the inner tensor in such a way that the inner tensors will
 * have differing extents (they must all have the same rank).
 */
struct ToTArrayFixture {

  ToTArrayFixture() : m_world(*GlobalFixture::world) {}
  ~ToTArrayFixture() { GlobalFixture::world->gop.fence(); }

  template<typename TupleElementType, typename Index>
  static auto inner_vector_tile(Index&& idx) {
    auto sum = std::accumulate(idx.begin(), idx.end(), 0);
    inner_type<TupleElementType> elem(Range(sum + 1));
    std::iota(elem.begin(), elem.end(), 1);
    return elem;
  }

  /* This function creates a tensor of vectors given a tiled range. We do this
   * by mapping the outer indices to the value of the inner vector. More
   * specifically outer index i maps to a vector with i+1 elements in it (the
   * values are 1 to i inclusive).
   */
  template<typename TupleElementType>
  auto tensor_of_vector(const TiledRange& tr) {
    return make_array<tensor_type<TupleElementType>>(m_world, tr,
        [](tile_type<TupleElementType>& tile, const Range& r){
          tile_type<TupleElementType> new_tile(r);
          for(auto idx : r){
            new_tile(idx) = inner_vector_tile<TupleElementType>(idx);
          }
          tile = new_tile;
    });
  }

  // The world to use for the test suite
  madness::World& m_world;

}; // TotArrayFixture

} // namespace

BOOST_FIXTURE_TEST_SUITE(tot_array_suite, ToTArrayFixture)

/*
 * This test case ensures the typedefs are what we think they are. We only test
 * the types which are directly affected by the tile type. The remaining types
 * are indirectly affected by the tile type (through the PIMPL type) and it is
 * assumed that the unit tests of the PIMPL's types ensure the correctness of
 * those. The directly affected types are:
 *
 * - DistArray_
 * - impl_type
 * - element_type
 * - scalar_type
 */
BOOST_AUTO_TEST_CASE_TEMPLATE(typedefs, TestParam, test_params){

  //Unpack the types for the test
  using scalar_type = scalar_type<TestParam>;
  using tile_type   = tile_type<TestParam>;
  using policy_type = policy_type<TestParam>;

  //The type of the DistArray whose types are being tested
  using tensor_t = tensor_type<TestParam>;

  //------------ Actual type checks start here -------------------------
  static_assert(std::is_same_v<typename tensor_t::DistArray_, tensor_t>);

  using corr_impl_type = detail::ArrayImpl<tile_type, policy_type>;
  static_assert(std::is_same_v<typename tensor_t::impl_type, corr_impl_type>);

  static_assert(std::is_same_v<typename tensor_t::element_type, scalar_type>);

  static_assert(std::is_same_v<typename tensor_t::scalar_type, scalar_type>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(default_ctor, TestParam, test_params){
  tensor_type<TestParam> t;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(copy_ctor, TestParam, test_params){
  tensor_type<TestParam> t;
  tensor_type<TestParam> copy_of_t(t);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dense_array_ctor, TestParam, test_params){
  trange_type<TestParam> tr{{0, 1}};
  tensor_type<TestParam> t(m_world, tr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sparse_array_ctor, TestParam, test_params){
  trange_type<TestParam> tr{{0, 1}};
  typename tensor_type<TestParam>::shape_type shape(1,tr);
  tensor_type<TestParam> t(m_world, tr, shape);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(initializer_list_ctor, TestParam, test_params){
  using inner_type = inner_type<TestParam>;
  inner_type e0(Range({1}), {1});
  inner_type e1(Range({2}), {1, 2});
  detail::vector_il<inner_type> il{e0, e1};
  tensor_type<TestParam> t(m_world, il);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(tiled_initializer_list_ctor, TestParam,
    test_params){
  using inner_type = inner_type<TestParam>;
  inner_type e0(Range({1}), {1});
  inner_type e1(Range({2}), {1, 2});
  detail::vector_il<inner_type> il{e0, e1};
  trange_type<TestParam> tr{{0, 1, 2}};
  tensor_type<TestParam> t(m_world, tr, il);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unary_transform_ctor, TestParam,
    test_params){

  using tile_type = tile_type<TestParam>;
  using scalar_type = scalar_type<TestParam>;

  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);

  tensor_type<TestParam> t2(t1, [](tile_type& out, const tile_type& in){
    tile_type buffer(in.range());
    for(auto idx : in.range()) {
      buffer(idx) = in(idx);
    }
  });
  
}

BOOST_AUTO_TEST_CASE_TEMPLATE(clone, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  auto t2 = t1.clone();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(copy_assignment, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  tensor_type<TestParam> t2;
  auto pt2 = &(t2 = t1);

  // Make sure it returns *this
  BOOST_CHECK_EQUAL(pt2, &t2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(begin, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  auto itr = t1.begin();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(const_begin, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  auto itr = std::as_const(t1).begin();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(end, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  auto itr = t1.end();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(const_end, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  auto itr = std::as_const(t1).end();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(find, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);

  { auto tile = t1.find(0); }
  { auto tile = t1.find(std::vector{0ul}); }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(find_init_list, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  auto tile = t1.find({0});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_sequence, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  std::vector<inner_type<TestParam>> v{
    inner_vector_tile<TestParam>(std::vector{0}),
    inner_vector_tile<TestParam>(std::vector{1})
  };
  { t1.set(0, v.begin()); }
  { t1.set(std::vector{0ul}, v.begin()); }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_sequence_init_list, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  std::vector<inner_type<TestParam>> v{
      inner_vector_tile<TestParam>(std::vector{0}),
      inner_vector_tile<TestParam>(std::vector{1})
  };
  t1.set({0}, v.begin());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_value, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
//  { t1.set(0, 42); }
//  { t1.set(std::vector{0ul}, 42); }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_value_init_list, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  auto elem = inner_vector_tile<TestParam>(std::vector{0});
  //t1.set({0}, 42);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_future, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  tensor_type<TestParam> t2(m_world, tr);
  auto elem = inner_vector_tile<TestParam>(std::vector{0});
  { t2.set(0, t1.find(0)); }
  { t2.set(std::vector{0ul}, t1.find(0)); }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_future_init_list, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  tensor_type<TestParam> t2(m_world, tr);
  t2.set({0}, t1.find(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_tile, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  tensor_type<TestParam> t2(m_world, tr);
  auto elem = inner_vector_tile<TestParam>(std::vector{0});
  { t2.set(0, t1.find(0).get()); }
  { t2.set(std::vector{0ul}, t1.find(0).get()); }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(set_tile_init_list, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  auto t1 = tensor_of_vector<TestParam>(tr);
  tensor_type<TestParam> t2(m_world, tr);
  t2.set({0}, t1.find(0).get());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fill_local, TestParam, test_params){
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  // Test w/o skipping filled
  {t1.fill_local(inner_vector_tile<TestParam>(0));}
  // Test skipping filled
  {
    t1.set(0, inner_vector_tile<TestParam>(1));
    t1.fill_local(inner_vector_tile<TestParam>(0), true);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fill, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  // Test w/o skipping filled
  {t1.fill(inner_vector_tile<TestParam>(0));}
  // Test skipping filled
  {
    t1.set(0, inner_vector_tile<TestParam>(1));
    t1.fill(inner_vector_tile<TestParam>(0), true);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fill_random, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  t1.fill_random();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(init_tiles, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  // Test w/o skipping filled
  { t1.init_tiles([](const Range& r){
      tile_type<TestParam> t(r);
      for(auto idx : r) t(idx) = inner_vector_tile<TestParam>(idx);
      return t;
    });
  }
  // Test skipping filled
  {
    t1.set(0, inner_vector_tile<TestParam>(1));
    t1.init_tiles([](const Range& r){
      tile_type<TestParam> t(r);
      for(auto idx : r) t(idx) = inner_vector_tile<TestParam>(idx);
      return t;
    }, true);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(init_elements, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  using index_type = typename tensor_type<TestParam>::index_type;
  //t1.init_elements([](const index_type&){ return })
}

BOOST_AUTO_TEST_CASE_TEMPLATE(trange, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};
  tensor_type<TestParam> t1(m_world, tr);
  BOOST_TEST(t1.trange() == tr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(range, TestParam, test_params) {
  trange_type<TestParam> tr{{0, 2}};;
  tensor_type<TestParam> t1(m_world, tr);
  Range corr(1);
  BOOST_TEST(t1.range() == corr);

}

//BOOST_AUTO_TEST_CASE_TEMPLATE(call_operator, TileType, tile_types){
//  TArray<TileType> t;
//}

BOOST_AUTO_TEST_SUITE_END()
