#include "TiledArray/tensor_impl_base.h"
#include "TiledArray/tensor.h"
#include "range_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct StaticTensorImplBaseFixture : public TiledRangeFixture {
  typedef StaticTiledRange<GlobalFixture::coordinate_system> trange_type;
  typedef expressions::Tensor<int, trange_type::tile_range_type> value_type;
  typedef DynamicTiledRange dynamic_trange_type;
  typedef detail::TensorImplBase<trange_type, value_type> tensor_impl_base;

  StaticTensorImplBaseFixture() : impl(* GlobalFixture::world, tr) { }

  tensor_impl_base impl;
}; // struct StaticTensorImplBaseFixture

BOOST_FIXTURE_TEST_SUITE( static_tensor_impl_base_suite , StaticTensorImplBaseFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
}

BOOST_AUTO_TEST_SUITE_END()



struct DynamicTensorImplBaseFixture : public TiledRangeFixture {
  typedef DynamicTiledRange trange_type;
  typedef expressions::Tensor<int, trange_type::tile_range_type> value_type;
  typedef detail::TensorImplBase<trange_type, value_type> tensor_impl_base;

  DynamicTensorImplBaseFixture() : impl(* GlobalFixture::world, tr) { }

  tensor_impl_base impl;
}; // struct DynamicTensorImplBaseFixture

BOOST_FIXTURE_TEST_SUITE( dynamic_tensor_impl_base_suite , DynamicTensorImplBaseFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr)));
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr));

}

BOOST_AUTO_TEST_SUITE_END()
