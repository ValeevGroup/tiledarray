/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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

 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  array_impl.cpp
 *  Oct 24, 2014
 *
 */

#include "TiledArray/array_impl.h"
#include "tiledarray.h"
#include "sparse_shape_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct ArrayImplBaseFixture : public SparseShapeFixture {
  typedef Tensor<int> value_type;
  typedef detail::ArrayImpl<value_type, DensePolicy> array_impl_base;
  ArrayImplBaseFixture() :
    pmap(new detail::HashPmap(* GlobalFixture::world, tr.tiles_range().volume()))
  { }

  std::shared_ptr<array_impl_base::pmap_interface> pmap;
}; // struct TensorImplBaseFixture

struct ArrayImplFixture : public ArrayImplBaseFixture {
  typedef TiledRange trange_type;
  typedef Tensor<int> value_type;
  typedef detail::ArrayImpl<value_type, DensePolicy> array_impl_base;
  typedef detail::ArrayImpl<value_type, SparsePolicy> sp_array_impl_base;
  typedef array_impl_base::shape_type dense_shape_type;
  typedef sp_array_impl_base::shape_type sparse_shape_type;

  ArrayImplFixture() :
    impl(* GlobalFixture::world, tr, dense_shape_type(), pmap),
    sp_impl(* GlobalFixture::world, tr, make_shape(tr, 0.5, 42), pmap)
  { }

  ~ArrayImplFixture() {
    GlobalFixture::world->gop.fence();
  }

  array_impl_base impl;
  sp_array_impl_base sp_impl;
}; // struct TensorImplFixture

BOOST_FIXTURE_TEST_SUITE( array_impl_suite , ArrayImplFixture )

BOOST_AUTO_TEST_CASE( constructor_dense_policy )
{
  BOOST_REQUIRE_NO_THROW(array_impl_base(* GlobalFixture::world, tr, dense_shape_type(), pmap));
  array_impl_base x(* GlobalFixture::world, tr, dense_shape_type(), pmap);

  // Check that the initial conditions are correct after constructution.
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
  BOOST_REQUIRE_NO_THROW(sp_array_impl_base(* GlobalFixture::world, tr, make_shape(tr, 0.5, 23), pmap));
  sp_array_impl_base x(* GlobalFixture::world, tr, make_shape(tr, 0.5, 23), pmap);

  // Check that the initial conditions are correct after constructution.
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

BOOST_AUTO_TEST_CASE( tile_get_and_set_w_value )
{
  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      const ProcessID owner = impl.owner(i);

      // Set each tile on node 0
      BOOST_CHECK_NO_THROW(impl.set(i, value_type(impl.trange().make_tile_range(i), owner)));

      // Get the tile future (may or may not be remote) and wait for the data to arrive.
      Future<value_type> tile = impl.get(i);
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      BOOST_CHECK(tile.probe());
      for(std::size_t j = 0ul; j < tile.get().size(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], owner);
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        Future<value_type> tile = impl.get(i);

        // Check that the future has been set and the data is what we expect.
        BOOST_CHECK(tile.probe());
        for(std::size_t j = 0ul; j < tile.get().size(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_get_and_set_w_future )
{
  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      const ProcessID owner = impl.owner(i);
      Future<value_type> tile;

      // Set each tile on node 0
      BOOST_CHECK_NO_THROW(impl.set(i, tile));

      // Get the tile future (may or may not be remote) and wait for the data to arrive.
      tile.set(value_type(impl.trange().make_tile_range(i), owner));
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      BOOST_CHECK(tile.probe());
      for(std::size_t j = 0ul; j < tile.get().size(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], owner);
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        Future<value_type> tile = impl.get(i);

        // Check that the future has been set and the data is what we expect.
        BOOST_CHECK(tile.probe());
        for(std::size_t j = 0ul; j < tile.get().size(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_iterator_w_value )
{
  // Set local tiles via iterators
  for(array_impl_base::iterator it = impl.begin(); it != impl.end(); ++it) {
    BOOST_CHECK_NO_THROW(*it = value_type(impl.trange().make_tile_range(it.ordinal()), impl.owner(it.ordinal())));
  }

  GlobalFixture::world->gop.fence();

  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      // Get the tile future (may or may not be remote) and wait for the data to arrive.
      value_type tile = impl.get(i);
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      for(std::size_t j = 0ul; j < tile.size(); ++j)
        BOOST_CHECK_EQUAL(tile[j], impl.owner(i));
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        value_type tile = impl.get(i);

        // Check that the future has been set and the data is what we expect.
        for(std::size_t j = 0ul; j < tile.size(); ++j)
          BOOST_CHECK_EQUAL(tile[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_iterator_w_future )
{
  // Set local tiles via iterators
  for(array_impl_base::iterator it = impl.begin(); it != impl.end(); ++it) {
    Future<value_type> tile;
    BOOST_CHECK_NO_THROW(*it = tile);
    tile.set(value_type(impl.trange().make_tile_range(it.ordinal()), GlobalFixture::world->rank()));
  }

  GlobalFixture::world->gop.fence();

  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      // Get the tile, which may be local or remote.
      Future<value_type> tile = impl.get(i);
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      BOOST_CHECK(tile.probe());
      for(std::size_t j = 0ul; j < tile.get().size(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], impl.owner(i));
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        Future<value_type> tile = impl.get(i);

        // Check that the future has been set and the data is what we expect.
        BOOST_CHECK(tile.probe());
        for(std::size_t j = 0ul; j < tile.get().size(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
