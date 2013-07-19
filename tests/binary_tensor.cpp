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

#include "TiledArray/binary_tensor.h"
#include "TiledArray/array.h"
#include "TiledArray/math/functional.h"
#include "config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct BinaryTensorFixture : public AnnotatedTensorFixture {
  typedef TensorExpression<array_annotation::value_type> tensor_expression;

  BinaryTensorFixture() :
      btt(make_binary_tensor(a(vars), a(vars), make_binary_tile_op(std::plus<int>())))
  {
    btt.eval(btt.vars(), std::shared_ptr<tensor_expression::pmap_interface>(
        new TiledArray::detail::BlockedPmap(* GlobalFixture::world, a.size()))).get();
  }

  tensor_expression btt;
}; // struct BinaryTensorFixture



BOOST_FIXTURE_TEST_SUITE( binary_tensor_suite, BinaryTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  BOOST_CHECK_EQUAL(btt.range(), a.range());
  BOOST_CHECK_EQUAL(btt.size(), a.size());
  BOOST_CHECK_EQUAL(btt.trange(), a.trange());
}

BOOST_AUTO_TEST_CASE( vars )
{
  BOOST_CHECK_EQUAL(btt.vars(), aa.vars());
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(btt.is_dense(), a.is_dense());
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(btt.get_shape(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( location )
{
  BOOST_CHECK((& btt.get_world()) == (& a.get_world()));
  for(std::size_t i = 0; i < btt.size(); ++i) {
    BOOST_CHECK(! btt.is_zero(i));
    BOOST_CHECK_EQUAL(btt.owner(i), a.owner(i));
    BOOST_CHECK_EQUAL(btt.is_local(i), a.is_local(i));
  }
}

BOOST_AUTO_TEST_CASE( result )
{
  for(tensor_expression::const_iterator it = btt.begin(); it != btt.end(); ++it) {
    madness::Future<array_annotation::value_type> input = a.find(it.index());

    BOOST_CHECK_EQUAL(it->get().range(), input.get().range());

    array_annotation::value_type::const_iterator input_it = input.get().begin();
    tensor_expression::value_type::const_iterator result_it = it->get().begin();
    for(; result_it != it->get().end(); ++result_it, ++input_it)
      BOOST_CHECK_EQUAL(*result_it, 2 * (*input_it));
  }
}

BOOST_AUTO_TEST_CASE( result_add_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> even(tr.tiles().volume());
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(i % 2)
      odd.set(i);
    else
      even.set(i);

  // Construct argument tensors
  ArrayN aeven(world, tr, even);
  ArrayN aodd(world, tr, odd);
  aeven.set_all_local(3);
  aodd.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aeven(vars), aodd(vars), make_binary_tile_op(std::plus<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
        if(i % 2)
          BOOST_CHECK_EQUAL(tile.get()[j], 2);
        else
          BOOST_CHECK_EQUAL(tile.get()[j], 3);

    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_subtract_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> even(tr.tiles().volume());
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(i % 2)
      odd.set(i);
    else
      even.set(i);

  // Construct argument tensors
  ArrayN aeven(world, tr, even);
  ArrayN aodd(world, tr, odd);
  aeven.set_all_local(3);
  aodd.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aeven(vars), aodd(vars), make_binary_tile_op(std::minus<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], -2);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 3);

    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_multiply_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> even(tr.tiles().volume());
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(i % 2)
      odd.set(i);
    else
      even.set(i);

  // Construct argument tensors
  ArrayN aeven(world, tr, even);
  ArrayN aodd(world, tr, odd);
  aeven.set_all_local(3);
  aodd.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be no tiles.
  ArrayN aresult = make_binary_tensor(aeven(vars), aodd(vars), make_binary_tile_op(std::multiplies<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(aresult.is_zero(i));

}

BOOST_AUTO_TEST_CASE( result_add_sparse_full_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  TiledArray::detail::Bitset<> all(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
    all.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr, all);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aodd(vars), aall(vars), make_binary_tile_op(std::plus<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 5);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 2);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_subtract_sparse_full_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  TiledArray::detail::Bitset<> all(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
    all.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr, all);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aodd(vars), aall(vars), make_binary_tile_op(std::minus<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 1);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], -2);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_multiply_sparse_full_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  TiledArray::detail::Bitset<> all(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
    all.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr, all);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  world.gop.fence();

  // Add the even and odd tiled arrays. The result should be odd tiles are filled.
  ArrayN aresult = make_binary_tensor(aodd(vars), aall(vars), make_binary_tile_op(std::multiplies<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      if(i % 2) {
        BOOST_CHECK(! aresult.is_zero(i));
        madness::Future<ArrayN::value_type> tile = aresult.find(i);
        BOOST_CHECK(tile.probe());
        ++local_count;
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 6);
      } else {
        BOOST_CHECK(aresult.is_zero(i));
      }
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume() / 2ul);
}

BOOST_AUTO_TEST_CASE( result_add_sparse_dense )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aodd(vars), aall(vars), make_binary_tile_op(std::plus<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 5);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 2);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_subtract_sparse_dense )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aodd(vars), aall(vars), make_binary_tile_op(std::minus<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 1);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], -2);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_multiply_sparse_dense )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  world.gop.fence();

  // Add the even and odd tiled arrays. The result should be odd tiles are filled.
  ArrayN aresult = make_binary_tensor(aodd(vars), aall(vars), make_binary_tile_op(std::multiplies<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      if(i % 2) {
        BOOST_CHECK(! aresult.is_zero(i));
        madness::Future<ArrayN::value_type> tile = aresult.find(i);
        BOOST_CHECK(tile.probe());
        ++local_count;
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 6);
      } else {
        BOOST_CHECK(aresult.is_zero(i));
      }
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume() / 2ul);
}

BOOST_AUTO_TEST_CASE( result_add_dense_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
  }

  // Construct tensors with even and odd tiles
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aall(vars), aodd(vars), make_binary_tile_op(std::plus<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 5);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 2);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_subtract_dense_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
  }

  // Construct tensors with even and odd tiles
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aall(vars), aodd(vars), make_binary_tile_op(std::minus<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      if(i % 2)
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], -1);
      else
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 2);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_multiply_dense_sparse )
{
  // Cerate even and odd bitsets
  TiledArray::detail::Bitset<> odd(tr.tiles().volume());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i) {
    if(i % 2)
      odd.set(i);
  }

  // Construct argument tensors
  ArrayN aodd(world, tr, odd);
  ArrayN aall(world, tr);
  aodd.set_all_local(3);
  aall.set_all_local(2);

  world.gop.fence();

  // Add the even and odd tiled arrays. The result should be odd tiles are filled.
  ArrayN aresult = make_binary_tensor(aall(vars), aodd(vars), make_binary_tile_op(std::multiplies<int>()));

  BOOST_CHECK(! aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      if(i % 2) {
        BOOST_CHECK(! aresult.is_zero(i));
        madness::Future<ArrayN::value_type> tile = aresult.find(i);
        BOOST_CHECK(tile.probe());
        ++local_count;
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 6);
      } else {
        BOOST_CHECK(aresult.is_zero(i));
      }
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume() / 2ul);
}

BOOST_AUTO_TEST_CASE( result_add_dense_dense )
{
  // Construct tensors with even and odd tiles
  ArrayN aall1(world, tr);
  ArrayN aall2(world, tr);
  aall1.set_all_local(3);
  aall2.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aall1(vars), aall2(vars), make_binary_tile_op(std::plus<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(!aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], 5);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_subtract_dense_dense )
{
  // Construct tensors with even and odd tiles
  ArrayN aall1(world, tr);
  ArrayN aall2(world, tr);
  aall1.set_all_local(3);
  aall2.set_all_local(2);

  // Add the even and odd tiled arrays. The result should be all tiles are filled.
  ArrayN aresult = make_binary_tensor(aall1(vars), aall2(vars), make_binary_tile_op(std::minus<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
      BOOST_CHECK(! aresult.is_zero(i));
      madness::Future<ArrayN::value_type> tile = aresult.find(i);
      BOOST_REQUIRE(tile.probe());
      ++local_count;
      for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
        BOOST_CHECK_EQUAL(tile.get()[j], 1);
    }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}

BOOST_AUTO_TEST_CASE( result_multiply_dense_dense )
{

  // Construct argument tensors
  ArrayN aall1(world, tr);
  ArrayN aall2(world, tr);
  aall1.set_all_local(3);
  aall2.set_all_local(2);

  world.gop.fence();

  // Add the even and odd tiled arrays. The result should be odd tiles are filled.
  ArrayN aresult = make_binary_tensor(aall1(vars), aall2(vars), make_binary_tile_op(std::multiplies<int>()));

  BOOST_CHECK(aresult.is_dense());

  world.gop.fence();

  std::size_t local_count = 0ul;
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    if(aresult.is_local(i)) {
        BOOST_CHECK(! aresult.is_zero(i));
        madness::Future<ArrayN::value_type> tile = aresult.find(i);
        BOOST_CHECK(tile.probe());
        ++local_count;
        for(std::size_t j = 0; j < tile.get().range().volume(); ++j)
          BOOST_CHECK_EQUAL(tile.get()[j], 6);
      }

  // check that all tiles are present
  world.gop.sum(&local_count, 1);
  BOOST_CHECK_EQUAL(local_count, tr.tiles().volume());
}
BOOST_AUTO_TEST_SUITE_END()
