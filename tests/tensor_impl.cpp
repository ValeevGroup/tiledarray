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
#include "TiledArray/tensor.h"
#include "sparse_shape_fixture.h"
#include "unit_test_config.h"
#include "TiledArray/pmap/hash_pmap.h"
#include "TiledArray/policies/dense_policy.h"
#include "TiledArray/policies/sparse_policy.h"

using namespace TiledArray;

struct TensorImplBaseFixture : public SparseShapeFixture {
  typedef Tensor<int> value_type;
  typedef detail::TensorImpl<value_type, DensePolicy> tensor_impl_base;
  TensorImplBaseFixture() :
    pmap(new detail::HashPmap(* GlobalFixture::world, tr.tiles().volume()))
  { }

  std::shared_ptr<tensor_impl_base::pmap_interface> pmap;
}; // struct TensorImplBaseFixture

struct TensorImplFixture : public TensorImplBaseFixture {
  typedef TiledRange trange_type;
  typedef Tensor<int> value_type;
  typedef detail::TensorImpl<value_type, DensePolicy> tensor_impl_base;
  typedef detail::TensorImpl<value_type, SparsePolicy> sp_tensor_impl_base;
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

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap() == pmap);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));
}

BOOST_AUTO_TEST_CASE( constructor_shape_policy )
{
  BOOST_REQUIRE_NO_THROW(sp_tensor_impl_base(* GlobalFixture::world, tr, make_shape(tr, 0.5, 23), pmap));
  sp_tensor_impl_base x(* GlobalFixture::world, tr, make_shape(tr, 0.5, 23), pmap);

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap() == pmap);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(! x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
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
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
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
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(! impl.shape().is_zero(i));
  }
}

BOOST_AUTO_TEST_CASE( zero )
{
  BOOST_CHECK(impl.is_dense());

  // Check that all tiles are non-zero when shape is dense
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(! impl.is_zero(i));
  }

  // Check that every third tile is non-zero
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(sp_impl.shape()[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(sp_impl.is_zero(i));
    } else {
      BOOST_CHECK(! sp_impl.is_zero(i));
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_get_and_set_w_value )
{
  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      const ProcessID owner = impl.owner(i);

      // Set each tile on node 0
      BOOST_CHECK_NO_THROW(impl.set(i, value_type(impl.trange().make_tile_range(i), owner)));

      // Get the tile future (may or may not be remote) and wait for the data to arrive.
      madness::Future<value_type> tile = impl.get(i);
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      BOOST_CHECK(tile.probe());
      for(std::size_t j = 0ul; j < tile.get().size(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], owner);
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        madness::Future<value_type> tile = impl.get(i);

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
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      const ProcessID owner = impl.owner(i);
      madness::Future<value_type> tile;

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
        madness::Future<value_type> tile = impl.get(i);

        // Check that the future has been set and the data is what we expect.
        BOOST_CHECK(tile.probe());
        for(std::size_t j = 0ul; j < tile.get().size(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_reference_w_value )
{
  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      const ProcessID owner = impl.owner(i);

      // Set each tile on node 0
      BOOST_CHECK_NO_THROW(impl[i] = value_type(impl.trange().make_tile_range(i), owner));

      // Get the tile future (may or may not be remote) and wait for the data to arrive.
      value_type tile = impl[i];
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      for(std::size_t j = 0ul; j < tile.size(); ++j)
        BOOST_CHECK_EQUAL(tile[j], owner);
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        value_type tile = impl[i];

        // Check that the future has been set and the data is what we expect.
        for(std::size_t j = 0ul; j < tile.size(); ++j)
          BOOST_CHECK_EQUAL(tile[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_reference_w_future )
{
  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      const ProcessID owner = impl.owner(i);
      madness::Future<value_type> tile;

      // Set each tile on node 0
      BOOST_CHECK_NO_THROW(impl[i] = tile);

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
        madness::Future<value_type> tile = impl[i];

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
  for(tensor_impl_base::iterator it = impl.begin(); it != impl.end(); ++it) {
    BOOST_CHECK_NO_THROW(*it = value_type(impl.trange().make_tile_range(it.ordinal()), impl.owner(it.ordinal())));
  }

  GlobalFixture::world->gop.fence();

  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      // Get the tile future (may or may not be remote) and wait for the data to arrive.
      value_type tile = impl[i];
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      for(std::size_t j = 0ul; j < tile.size(); ++j)
        BOOST_CHECK_EQUAL(tile[j], impl.owner(i));
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        value_type tile = impl[i];

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
  for(tensor_impl_base::iterator it = impl.begin(); it != impl.end(); ++it) {
    madness::Future<value_type> tile;
    BOOST_CHECK_NO_THROW(*it = tile);
    tile.set(value_type(impl.trange().make_tile_range(it.ordinal()), GlobalFixture::world->rank()));
  }

  GlobalFixture::world->gop.fence();

  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      // Get the tile, which may be local or remote.
      madness::Future<value_type> tile = impl[i];
      GlobalFixture::world->gop.fence();

      // Check that the future has been set and the data is what we expect.
      BOOST_CHECK(tile.probe());
      for(std::size_t j = 0ul; j < tile.get().size(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], impl.owner(i));
    } else {
      GlobalFixture::world->gop.fence();
      if(impl.is_local(i)) {
        // Get the local tile
        madness::Future<value_type> tile = impl[i];

        // Check that the future has been set and the data is what we expect.
        BOOST_CHECK(tile.probe());
        for(std::size_t j = 0ul; j < tile.get().size(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], GlobalFixture::world->rank());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( set_value )
{
  // Check that we can set all elements
  for(std::size_t i = 0; i < impl.size(); ++i)
    if(impl.is_local(i))
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));

  GlobalFixture::world->gop.fence();
  std::size_t n = impl.local_size();
  GlobalFixture::world->gop.sum(n);

  BOOST_CHECK_EQUAL(n, impl.size());

  // Check throw for an out-of-range set.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.set(impl.size(), value_type()), TiledArray::Exception);
  BOOST_CHECK_THROW(impl.set(impl.size() + 2, value_type()), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( assign_value )
{
  // Check that we can set all elements
  for(std::size_t i = 0; i < impl.size(); ++i)
    if(impl.is_local(i)) {
      BOOST_CHECK_NO_THROW(impl[i] =
          value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
    }

  GlobalFixture::world->gop.fence();
  std::size_t n = impl.local_size();
  GlobalFixture::world->gop.sum(n);

  BOOST_CHECK_EQUAL(n, impl.size());
}

BOOST_AUTO_TEST_CASE( array_operator )
{
  // Check that elements are inserted properly for access requests.
  for(std::size_t i = 0; i < impl.size(); ++i) {
    tensor_impl_base::future f = impl[i];
    f.probe();
    if(impl.is_local(i))
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
  }

  GlobalFixture::world->gop.fence();
  std::size_t n = impl.local_size();
  GlobalFixture::world->gop.sum(n);

  // Check that all tiles are accounted for
  BOOST_CHECK_EQUAL(n, impl.size());

  // Check throw for an out-of-range set.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl[impl.size()], TiledArray::Exception);
  BOOST_CHECK_THROW(impl[impl.size() + 2], TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_SUITE_END()
