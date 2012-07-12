#include "TiledArray/tensor_impl_base.h"
#include "TiledArray/tensor.h"
#include "range_fixture.h"
#include "unit_test_config.h"
#include "TiledArray/hash_pmap.h"

using namespace TiledArray;

struct StaticTensorImplBaseFixture : public TiledRangeFixture {
  typedef StaticTiledRange<GlobalFixture::coordinate_system> trange_type;
  typedef expressions::Tensor<int, trange_type::tile_range_type> value_type;
  typedef DynamicTiledRange dynamic_trange_type;
  typedef detail::TensorImplBase<trange_type, value_type> tensor_impl_base;

  StaticTensorImplBaseFixture() : impl(* GlobalFixture::world, tr),
      pmap(new detail::HashPmap(* GlobalFixture::world, tr.tiles().volume())) {
    impl.pmap(pmap);
  }

  ~StaticTensorImplBaseFixture() {
    GlobalFixture::world->gop.fence();
  }

  tensor_impl_base impl;
  std::shared_ptr<tensor_impl_base::pmap_interface> pmap;
}; // struct StaticTensorImplBaseFixture

BOOST_FIXTURE_TEST_SUITE( static_tensor_impl_base_suite , StaticTensorImplBaseFixture )

BOOST_AUTO_TEST_CASE( constructor_with_static_tiled_range )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr));
  tensor_impl_base x(* GlobalFixture::world, tr);

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_dyanmic_tiled_range )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr)));
  tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_dyanmic_tiled_range_and_shape )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(tr.tiles().volume())));
  tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(tr.tiles().volume()));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK_EQUAL(x.shape().size(), tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_static_tiled_range_and_shape )
{
    BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(tr.tiles().volume())));
    tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(tr.tiles().volume()));

    // Check that the initial conditions are correct after constructution.
    BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
    BOOST_CHECK(x.pmap().get() == NULL);
    BOOST_CHECK_EQUAL(x.range(), tr.tiles());
    BOOST_CHECK_EQUAL(x.trange(), tr);
    BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
    BOOST_CHECK(x.begin() == x.end());
    BOOST_CHECK(! x.is_dense());
    BOOST_CHECK_EQUAL(x.shape().size(), tr.tiles().volume());
    for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
      BOOST_CHECK(x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_dyanmic_tiled_range_and_empty_shape )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(0)));
  tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(0));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_static_tiled_range_and_empty_shape )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(0)));
  tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(0));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}


BOOST_AUTO_TEST_CASE( process_map )
{
  BOOST_CHECK(impl.pmap() == pmap);

  // Check that the process map cannot be set more than once
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.pmap(pmap), Exception);
#endif // TA_EXCEPTION_ERROR

  // Check that the impl ownership and locality are correct
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK_EQUAL(impl.owner(i), pmap->owner(i));
    if(impl.owner(i) == GlobalFixture::world->rank())
      BOOST_CHECK(impl.is_local(i));
    else
      BOOST_CHECK(! impl.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( shape_set_and_get )
{
  // Make sure that shape cannot be accessed when dense
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.shape(), Exception);
#endif // TA_EXCEPTION_ERROR

  // Set an empty shape
  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Check that the tensor shape and s are the same
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(! impl.shape()[i]);
  }

  // Check that all tiles are zero when shape is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      s.set(i);
  }

  impl.shape(s);

  // Check that the tensor shape and s are the same
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(s[i])
      BOOST_CHECK(impl.shape()[i]);
    else
      BOOST_CHECK(! impl.shape()[i]);
  }
}

BOOST_AUTO_TEST_CASE( shape_modify )
{
  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Check that we can modify zero and non-zero tiles
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2) {
      impl.shape(i, false);
      BOOST_CHECK(! impl.shape()[i]);
    } else {
      impl.shape(i);
      BOOST_CHECK(impl.shape()[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE( zero )
{
  // Check that all tiles are non-zero when shape is not set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(! impl.is_zero(i));
  }

  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Check that all tiles are zero when shape is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(impl.is_zero(i));
  }

  // Check that all tiles are zero when shape is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      s.set(i);
  }

  impl.shape(s);

  // Check that even tiles are set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      BOOST_CHECK(! impl.is_zero(i));
    else
      BOOST_CHECK(impl.is_zero(i));
  }

  // Check that we can modify zero and non-zero tiles
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2) {
      impl.shape(i, false);
      BOOST_CHECK(impl.is_zero(i));
    } else {
      impl.shape(i);
      BOOST_CHECK(! impl.is_zero(i));
    }
  }

  // Check that range is checked correctly
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.is_zero(tr.tiles().volume()), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( tile_set_and_get )
{
  std::vector<madness::Future<value_type> > tiles;
  tiles.reserve(tr.tiles().volume());


  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    tiles.push_back(impl[i]);
  }

}

BOOST_AUTO_TEST_CASE( dense )
{
  // Make sure the tensor is dense when shape is not set
  BOOST_CHECK(impl.is_dense());

  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Make sure the tensor is not dense when shape is set
  BOOST_CHECK(! impl.is_dense());
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

BOOST_AUTO_TEST_CASE( array_operator )
{
  // Check that elements are inserted properly for access requests.
  for(std::size_t i = 0; i < impl.size(); ++i) {
    impl[i].probe();
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

BOOST_AUTO_TEST_CASE( move_local )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
  }

  GlobalFixture::world->gop.fence();

  // Set all local tiles and check that the tile is removed from the tensor
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i)) {
      madness::Future<value_type> f = impl.move(i);
      --local_size;

      BOOST_CHECK_EQUAL(f.get()[0], GlobalFixture::world->rank());
      BOOST_CHECK_EQUAL(local_size, impl.local_size());
    }
  }
}

BOOST_AUTO_TEST_CASE( delayed_move_local )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  // Move all local tiles
  std::vector<madness::Future<value_type> > local_data;
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      local_data.push_back(impl.move(i));
  }

  // Ensure that all tiles are still present since none have been set.
  BOOST_CHECK_EQUAL(impl.local_size(), local_size);

  // Set all tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i)) {
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
      --local_size;
    }

    // Ensure that the tile has been removed from the tensor impl
    BOOST_CHECK_EQUAL(impl.local_size(), local_size);
  }

  // Check that the moved tiles have the correct value
  for(std::vector<madness::Future<value_type> >::const_iterator it = local_data.begin(); it != local_data.end(); ++it) {
    BOOST_CHECK_EQUAL(it->get()[0], GlobalFixture::world->rank());
  }

}

BOOST_AUTO_TEST_CASE( move_remote )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  GlobalFixture::world->gop.fence();

  // Move all tiles to node 0 and check that the tile is removed from the local tensor
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      madness::Future<value_type> f = impl.move(i);
      BOOST_CHECK_EQUAL(f.get()[0], impl.owner(i));
    }

    if(impl.is_local(i))
      --local_size;

    GlobalFixture::world->gop.fence();

    BOOST_CHECK_EQUAL(local_size, impl.local_size());
  }
}


BOOST_AUTO_TEST_CASE( delayed_move_remote )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  // Move all the tiles to node 0
  std::vector<madness::Future<value_type> > local_data;
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(GlobalFixture::world->rank() == 0)
      local_data.push_back(impl.move(i));
  }

  // Set each tile and check that it is moved immediately.
  for(std::size_t i = 0; i < impl.size(); ++i) {

    GlobalFixture::world->gop.fence();

    if(impl.is_local(i)) {
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
      --local_size;
    }

    BOOST_CHECK_EQUAL(local_size, impl.local_size());
  }

  // Check that the moved tiles have the correct value
  if(GlobalFixture::world->rank() == 0) {
    for(std::vector<madness::Future<value_type> >::iterator it = local_data.begin(); it != local_data.end(); ++it) {
      BOOST_CHECK_EQUAL(it->get()[0], impl.owner(std::distance(local_data.begin(), it)));
    }
  }

}


BOOST_AUTO_TEST_CASE( access_zero_tile ) {
  impl.shape(detail::Bitset<>(impl.size()));

#ifdef TA_EXCEPTION_ERROR
  // Check that you cannot access a tile that is zero
  BOOST_CHECK_THROW(impl[0], TiledArray::Exception);
  // Check that you cannot move a tile that is zero
  BOOST_CHECK_THROW(impl.move(0), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( clear )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Check that there are tiles inserted locally
  BOOST_CHECK_EQUAL(impl.local_size(), pmap->local_size());
  if(impl.local_size() > 0)
    BOOST_CHECK(impl.begin() != impl.end());
  else
    BOOST_CHECK(impl.begin() == impl.end());

  impl.clear();

  BOOST_CHECK_EQUAL(impl.local_size(), 0ul);
  BOOST_CHECK(impl.begin() == impl.end());

}

BOOST_AUTO_TEST_SUITE_END()



struct DynamicTensorImplBaseFixture : public TiledRangeFixture {
  typedef DynamicTiledRange trange_type;
  typedef expressions::Tensor<int, trange_type::tile_range_type> value_type;
  typedef detail::TensorImplBase<trange_type, value_type> tensor_impl_base;

  DynamicTensorImplBaseFixture() : impl(* GlobalFixture::world, tr),
      pmap(new detail::HashPmap(* GlobalFixture::world, tr.tiles().volume())) {
    impl.pmap(pmap);
  }

  ~DynamicTensorImplBaseFixture() {
    GlobalFixture::world->gop.fence();
  }

  tensor_impl_base impl;
  std::shared_ptr<tensor_impl_base::pmap_interface> pmap;
}; // struct DynamicTensorImplBaseFixture

BOOST_FIXTURE_TEST_SUITE( dynamic_tensor_impl_base_suite , DynamicTensorImplBaseFixture )

BOOST_AUTO_TEST_CASE( constructor_with_static_tiled_range )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr));
  tensor_impl_base x(* GlobalFixture::world, tr);

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_dyanmic_tiled_range )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr)));
  tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_dyanmic_tiled_range_and_shape )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(tr.tiles().volume())));
  tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(tr.tiles().volume()));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK_EQUAL(x.shape().size(), tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_static_tiled_range_and_shape )
{
    BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(tr.tiles().volume())));
    tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(tr.tiles().volume()));

    // Check that the initial conditions are correct after constructution.
    BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
    BOOST_CHECK(x.pmap().get() == NULL);
    BOOST_CHECK_EQUAL(x.range(), tr.tiles());
    BOOST_CHECK_EQUAL(x.trange(), tr);
    BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
    BOOST_CHECK(x.begin() == x.end());
    BOOST_CHECK(! x.is_dense());
    BOOST_CHECK_EQUAL(x.shape().size(), tr.tiles().volume());
    for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
      BOOST_CHECK(x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_dyanmic_tiled_range_and_empty_shape )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(0)));
  tensor_impl_base x(* GlobalFixture::world, DynamicTiledRange(tr), detail::Bitset<>(0));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor_with_static_tiled_range_and_empty_shape )
{
  BOOST_REQUIRE_NO_THROW(tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(0)));
  tensor_impl_base x(* GlobalFixture::world, tr, detail::Bitset<>(0));

  // Check that the initial conditions are correct after constructution.
  BOOST_CHECK_EQUAL(& x.get_world(), GlobalFixture::world);
  BOOST_CHECK(x.pmap().get() == NULL);
  BOOST_CHECK_EQUAL(x.range(), tr.tiles());
  BOOST_CHECK_EQUAL(x.trange(), tr);
  BOOST_CHECK_EQUAL(x.size(), tr.tiles().volume());
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK(x.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! x.is_zero(i));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x.shape(), Exception);
  BOOST_CHECK_THROW(x.shape(0ul,true), Exception);
  BOOST_CHECK_THROW(x.is_local(0ul), Exception);
  BOOST_CHECK_THROW(x.owner(0ul), Exception);
#endif // TA_EXCEPTION_ERROR
}


BOOST_AUTO_TEST_CASE( process_map )
{
  BOOST_CHECK(impl.pmap() == pmap);

  // Check that the process map cannot be set more than once
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.pmap(pmap), Exception);
#endif // TA_EXCEPTION_ERROR

  // Check that the impl ownership and locality are correct
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK_EQUAL(impl.owner(i), pmap->owner(i));
    if(impl.owner(i) == GlobalFixture::world->rank())
      BOOST_CHECK(impl.is_local(i));
    else
      BOOST_CHECK(! impl.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( shape_set_and_get )
{
  // Make sure that shape cannot be accessed when dense
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.shape(), Exception);
#endif // TA_EXCEPTION_ERROR

  // Set an empty shape
  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Check that the tensor shape and s are the same
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(! impl.shape()[i]);
  }

  // Check that all tiles are zero when shape is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      s.set(i);
  }

  impl.shape(s);

  // Check that the tensor shape and s are the same
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(s[i])
      BOOST_CHECK(impl.shape()[i]);
    else
      BOOST_CHECK(! impl.shape()[i]);
  }
}

BOOST_AUTO_TEST_CASE( shape_modify )
{
  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Check that we can modify zero and non-zero tiles
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2) {
      impl.shape(i, false);
      BOOST_CHECK(! impl.shape()[i]);
    } else {
      impl.shape(i);
      BOOST_CHECK(impl.shape()[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE( zero )
{
  // Check that all tiles are non-zero when shape is not set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(! impl.is_zero(i));
  }

  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Check that all tiles are zero when shape is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK(impl.is_zero(i));
  }

  // Check that all tiles are zero when shape is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      s.set(i);
  }

  impl.shape(s);

  // Check that even tiles are set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      BOOST_CHECK(! impl.is_zero(i));
    else
      BOOST_CHECK(impl.is_zero(i));
  }

  // Check that we can modify zero and non-zero tiles
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2) {
      impl.shape(i, false);
      BOOST_CHECK(impl.is_zero(i));
    } else {
      impl.shape(i);
      BOOST_CHECK(! impl.is_zero(i));
    }
  }

  // Check that range is checked correctly
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(impl.is_zero(tr.tiles().volume()), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( tile_set_and_get )
{
  std::vector<madness::Future<value_type> > tiles;
  tiles.reserve(tr.tiles().volume());


  // Get each tile before it is set
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    tiles.push_back(impl[i]);
  }

}

BOOST_AUTO_TEST_CASE( dense )
{
  // Make sure the tensor is dense when shape is not set
  BOOST_CHECK(impl.is_dense());

  detail::Bitset<> s(tr.tiles().volume());
  impl.shape(s);

  // Make sure the tensor is not dense when shape is set
  BOOST_CHECK(! impl.is_dense());
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

BOOST_AUTO_TEST_CASE( array_operator )
{
  // Check that elements are inserted properly for access requests.
  for(std::size_t i = 0; i < impl.size(); ++i) {
    impl[i].probe();
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

BOOST_AUTO_TEST_CASE( move_local )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
  }

  GlobalFixture::world->gop.fence();

  // Set all local tiles and check that the tile is removed from the tensor
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i)) {
      madness::Future<value_type> f = impl.move(i);
      --local_size;

      BOOST_CHECK_EQUAL(f.get()[0], GlobalFixture::world->rank());
      BOOST_CHECK_EQUAL(local_size, impl.local_size());
    }
  }
}

BOOST_AUTO_TEST_CASE( delayed_move_local )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  // Move all local tiles
  std::vector<madness::Future<value_type> > local_data;
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      local_data.push_back(impl.move(i));
  }

  // Ensure that all tiles are still present since none have been set.
  BOOST_CHECK_EQUAL(impl.local_size(), local_size);

  // Set all tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i)) {
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
      --local_size;
    }

    // Ensure that the tile has been removed from the tensor impl
    BOOST_CHECK_EQUAL(impl.local_size(), local_size);
  }

  // Check that the moved tiles have the correct value
  for(std::vector<madness::Future<value_type> >::const_iterator it = local_data.begin(); it != local_data.end(); ++it) {
    BOOST_CHECK_EQUAL(it->get()[0], GlobalFixture::world->rank());
  }

}

BOOST_AUTO_TEST_CASE( move_remote )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  GlobalFixture::world->gop.fence();

  // Move all tiles to node 0 and check that the tile is removed from the local tensor
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(GlobalFixture::world->rank() == 0) {
      madness::Future<value_type> f = impl.move(i);
      BOOST_CHECK_EQUAL(f.get()[0], impl.owner(i));
    }

    if(impl.is_local(i))
      --local_size;

    GlobalFixture::world->gop.fence();

    BOOST_CHECK_EQUAL(local_size, impl.local_size());
  }
}


BOOST_AUTO_TEST_CASE( delayed_move_remote )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Get total local tile count
  std::size_t local_size = impl.local_size();

  // Move all the tiles to node 0
  std::vector<madness::Future<value_type> > local_data;
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(GlobalFixture::world->rank() == 0)
      local_data.push_back(impl.move(i));
  }

  // Set each tile and check that it is moved immediately.
  for(std::size_t i = 0; i < impl.size(); ++i) {

    GlobalFixture::world->gop.fence();

    if(impl.is_local(i)) {
      impl.set(i, value_type(impl.trange().make_tile_range(i), GlobalFixture::world->rank()));
      --local_size;
    }

    BOOST_CHECK_EQUAL(local_size, impl.local_size());
  }

  // Check that the moved tiles have the correct value
  if(GlobalFixture::world->rank() == 0) {
    for(std::vector<madness::Future<value_type> >::iterator it = local_data.begin(); it != local_data.end(); ++it) {
      BOOST_CHECK_EQUAL(it->get()[0], impl.owner(std::distance(local_data.begin(), it)));
    }
  }

}

BOOST_AUTO_TEST_CASE( access_zero_tile ) {
  impl.shape(detail::Bitset<>(impl.size()));

#ifdef TA_EXCEPTION_ERROR
  // Check that you cannot access a tile that is zero
  BOOST_CHECK_THROW(impl[0], TiledArray::Exception);
  // Check that you cannot move a tile that is zero
  BOOST_CHECK_THROW(impl.move(0), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( clear )
{
  // Insert all local tiles
  for(std::size_t i = 0; i < impl.size(); ++i) {
    if(impl.is_local(i))
      impl[i];
  }

  // Check that there are tiles inserted locally
  BOOST_CHECK_EQUAL(impl.local_size(), pmap->local_size());
  if(impl.local_size() > 0)
    BOOST_CHECK(impl.begin() != impl.end());
  else
    BOOST_CHECK(impl.begin() == impl.end());

  impl.clear();

  BOOST_CHECK_EQUAL(impl.local_size(), 0ul);
  BOOST_CHECK(impl.begin() == impl.end());

}

BOOST_AUTO_TEST_SUITE_END()
