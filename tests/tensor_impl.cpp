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

#include "TiledArray/tensor_impl.h"
#include "tiledarray.h"
#include "sparse_shape_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct TensorImplBaseFixture : public SparseShapeFixture {
  typedef detail::TensorImpl<DensePolicy> tensor_impl_base;
  TensorImplBaseFixture() :
    pmap(new detail::HashPmap(* GlobalFixture::world, tr.tiles_range().volume()))
  { }

  std::shared_ptr<tensor_impl_base::pmap_interface> pmap;
}; // struct TensorImplBaseFixture

struct TensorImplFixture : public TensorImplBaseFixture {
  typedef TiledRange trange_type;
  typedef Tensor<int> value_type;
  typedef detail::TensorImpl<DensePolicy> tensor_impl_base;
  typedef detail::TensorImpl<SparsePolicy> sp_tensor_impl_base;
  typedef tensor_impl_base::shape_type dense_shape_type;
  typedef sp_tensor_impl_base::shape_type sparse_shape_type;

  TensorImplFixture() :
    impl(* GlobalFixture::world, tr, dense_shape_type(), pmap),
    sp_impl(* GlobalFixture::world, tr, make_shape(tr, 0.5, 42), pmap)
  { }

  ~TensorImplFixture() {
    GlobalFixture::world->gop.fence();
  }

  tensor_impl_base impl;
  sp_tensor_impl_base sp_impl;
}; // struct TensorImplFixture

BOOST_FIXTURE_TEST_SUITE( tensor_impl_suite , TensorImplFixture )

BOOST_AUTO_TEST_CASE( constructor_dense_policy )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base(* GlobalFixture::world, tr, dense_shape_type(), pmap));
  tensor_impl_base x(* GlobalFixture::world, tr, dense_shape_type(), pmap);

  // Check that the initial conditions are correct after construction.
  BOOST_CHECK_EQUAL(& x.world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap() == pmap);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles_range());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles_range().volume());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));
}

BOOST_AUTO_TEST_CASE( constructor_shape_policy )
{
  BOOST_REQUIRE_NO_THROW(sp_tensor_impl_base(* GlobalFixture::world, tr, make_shape(tr, 0.5, 23), pmap));
  sp_tensor_impl_base x(* GlobalFixture::world, tr, make_shape(tr, 0.5, 23), pmap);

  // Check that the initial conditions are correct after construction.
  BOOST_CHECK_EQUAL(& x.world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap() == pmap);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles_range());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles_range().volume());
  BOOST_CHECK(! x.is_dense());
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if(x.shape()[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(x.is_zero(i));
    } else {
      BOOST_CHECK(! x.is_zero(i));
    }
  }
}

BOOST_AUTO_TEST_CASE( process_map )
{
  BOOST_CHECK(impl.pmap() == pmap);

  // Check that the impl ownership and locality are correct
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    BOOST_CHECK_EQUAL(impl.owner(i), pmap->owner(i));
    if(impl.owner(i) == GlobalFixture::world->rank())
      BOOST_CHECK(impl.is_local(i));
    else
      BOOST_CHECK(! impl.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( shape_access )
{
  BOOST_CHECK(impl.shape().is_dense());

  // Check that the tensor shape and s are the same
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    BOOST_CHECK(! impl.shape().is_zero(i));
  }
}

BOOST_AUTO_TEST_CASE( zero )
{
  BOOST_CHECK(impl.is_dense());

  // Check that all tiles are non-zero when shape is dense
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    BOOST_CHECK(! impl.is_zero(i));
  }

  // Check that every third tile is non-zero
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if(sp_impl.shape()[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(sp_impl.is_zero(i));
    } else {
      BOOST_CHECK(! sp_impl.is_zero(i));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
