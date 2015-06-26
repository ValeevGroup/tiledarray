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

#include "TiledArray/array.h"
#include "tiledarray.h"
#include "array_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

ArrayFixture::ArrayFixture() :
    shape_tensor(tr.tiles(), 0.0),
    world(*GlobalFixture::world),
    a(world, tr)
{
  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it)
    if(a.is_local(*it))
      a.set(*it, world.rank() + 1); // Fill the tile at *it (the index)

  world.gop.fence();

  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(i % 3)
      shape_tensor[i] = 1.0;
}

ArrayFixture::~ArrayFixture() {
  GlobalFixture::world->gop.fence();
}


BOOST_FIXTURE_TEST_SUITE( array_suite , ArrayFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  // Construct a dense array
  BOOST_REQUIRE_NO_THROW(ArrayN ad(world, tr));
  ArrayN ad(world, tr);

  // Check that none of the tiles have been set.
  for(ArrayN::const_iterator it = ad.begin(); it != ad.end(); ++it)
    BOOST_CHECK(! it->probe());

  // Construct a sparse array
  BOOST_REQUIRE_NO_THROW(SpArrayN as(world, tr, TiledArray::SparseShape<float>(shape_tensor, tr)));
  SpArrayN as(world, tr, TiledArray::SparseShape<float>(shape_tensor, tr));

  // Check that none of the tiles have been set.
  for(SpArrayN::const_iterator it = as.begin(); it != as.end(); ++it)
    BOOST_CHECK(! it->probe());

}

BOOST_AUTO_TEST_CASE( all_owned )
{
  std::size_t count = 0ul;
  for(std::size_t i = 0ul; i < tr.tiles().volume(); ++i)
    if(a.owner(i) == GlobalFixture::world->rank())
      ++count;
  world.gop.sum(count);

  // Check that all tiles are in the array
  BOOST_CHECK_EQUAL(tr.tiles().volume(), count);
}

BOOST_AUTO_TEST_CASE( owner )
{
  // Test to make sure everyone agrees who owns which tiles.
  std::shared_ptr<ProcessID> group_owner(new ProcessID[world.size()],
      std::default_delete<ProcessID[]>());

  size_type o = 0;
  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it, ++o) {
    // Check that local ownership agrees
    const int owner = a.owner(*it);
    BOOST_CHECK_EQUAL(a.owner(o), owner);

    // Get the owner from all other processes
    int count = (owner == world.rank() ? 1 : 0);
    world.gop.sum(count);
    // Check that only one node claims ownership
    BOOST_CHECK_EQUAL(count, 1);

    std::fill_n(group_owner.get(), world.size(), 0);
    group_owner.get()[world.rank()] = owner;
    world.gop.reduce(group_owner.get(), world.size(), std::plus<ProcessID>());


    // Check that everyone agrees who the owner of the tile is.
    BOOST_CHECK((std::find_if(group_owner.get(), group_owner.get() + world.size(),
        std::bind1st(std::not_equal_to<ProcessID>(), owner)) == (group_owner.get() + world.size())));

  }
}

BOOST_AUTO_TEST_CASE( is_local )
{
  // Test to make sure everyone agrees who owns which tiles.

  size_type o = 0;
  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it, ++o) {
    // Check that local ownership agrees
    const bool local_tile = a.owner(o) == world.rank();
    BOOST_CHECK_EQUAL(a.is_local(*it), local_tile);
    BOOST_CHECK_EQUAL(a.is_local(o), local_tile);

    // Find out how many claim ownership
    int count = (local_tile ? 1 : 0);
    world.gop.sum(count);

    // Check how many process claim ownership
    // "There can be only one!"
    BOOST_CHECK_EQUAL(count, 1);
  }
}

BOOST_AUTO_TEST_CASE( find_local )
{
  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it) {

    if(a.is_local(*it)) {
      Future<ArrayN::value_type> tile = a.find(*it);

      BOOST_CHECK(tile.probe());

      const int value = world.rank() + 1;
      for(ArrayN::value_type::iterator it = tile.get().begin(); it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, value);
    }
  }
}

BOOST_AUTO_TEST_CASE( find_remote )
{
  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it) {

    if(! a.is_local(*it)) {
      Future<ArrayN::value_type> tile = a.find(*it);

      const int owner = a.owner(*it);
      for(ArrayN::value_type::iterator it = tile.get().begin(); it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, owner + 1);
    }
  }
}

BOOST_AUTO_TEST_CASE( fill_tiles )
{
  ArrayN a(world, tr);

  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it) {
    if(a.is_local(*it)) {
      a.set(*it, 0); // Fill the tile at *it (the index) with 0

      Future<ArrayN::value_type> tile = a.find(*it);

      // Check that the range for the constructed tile is correct.
      BOOST_CHECK_EQUAL(tile.get().range(), tr.make_tile_range(*it));

      for(ArrayN::value_type::iterator it = tile.get().begin(); it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, 0);
    }
  }
}

BOOST_AUTO_TEST_CASE( assign_tiles )
{
  std::vector<int> data;
  ArrayN a(world, tr);

  for(ArrayN::range_type::const_iterator it = a.range().begin(); it != a.range().end(); ++it) {
    ArrayN::trange_type::tile_range_type range = a.trange().make_tile_range(*it);
    if(a.is_local(*it)) {
      if(data.size() < range.volume())
        data.resize(range.volume(), 1);
      a.set(*it, data.begin());

      Future<ArrayN::value_type> tile = a.find(*it);
      BOOST_CHECK(tile.probe());

      // Check that the range for the constructed tile is correct.
      BOOST_CHECK_EQUAL(tile.get().range(), tr.make_tile_range(*it));

      for(ArrayN::value_type::iterator it = tile.get().begin(); it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, 1);
    }
  }
}

BOOST_AUTO_TEST_CASE( make_replicated )
{
  // Get a copy of the original process map
  std::shared_ptr<ArrayN::pmap_interface> distributed_pmap = a.get_pmap();

  // Convert array to a replicated array.
  BOOST_REQUIRE_NO_THROW(a.make_replicated());

  if(GlobalFixture::world->size() == 1)
    BOOST_CHECK(! a.get_pmap()->is_replicated());
  else
    BOOST_CHECK(a.get_pmap()->is_replicated());

  // Check that all the data is local
  for(std::size_t i = 0; i < a.size(); ++i) {
    BOOST_CHECK(a.is_local(i));
    BOOST_CHECK_EQUAL(a.get_pmap()->owner(i), GlobalFixture::world->rank());
    Future<ArrayN::value_type> tile = a.find(i);
    BOOST_CHECK_EQUAL(tile.get().range(), a.trange().make_tile_range(i));
    for(ArrayN::value_type::const_iterator it = tile.get().begin(); it != tile.get().end(); ++it)
      BOOST_CHECK_EQUAL(*it, distributed_pmap->owner(i) + 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()

