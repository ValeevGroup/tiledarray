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

#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <random>

#include <madness/world/binary_fstream_archive.h>
#include <madness/world/text_fstream_archive.h>

#include <array_fixture.h>
#include "../src/TiledArray/dist_array.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

ArrayFixture::ArrayFixture()
    : shape_tensor(tr.tiles_range(), 0.0),
      world(*GlobalFixture::world),
      a(world, tr) {
  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it)
    if (a.is_local(*it))
      a.set(*it, world.rank() + 1);  // Fill the tile at *it (the index)

  for (std::size_t i = 0; i < tr.tiles_range().volume(); ++i) {
    if (i % 3) shape_tensor[i] = 1.0;
  }

  b = decltype(b)(world, tr, TiledArray::SparseShape<float>(shape_tensor, tr));
  for (SpArrayN::range_type::const_iterator it = b.range().begin();
       it != b.range().end(); ++it)
    if (!b.is_zero(*it) && b.is_local(*it))
      b.set(*it, world.rank() + 1);  // Fill the tile at *it (the index)

  world.gop.fence();
}

ArrayFixture::~ArrayFixture() { GlobalFixture::world->gop.fence(); }

namespace {
std::string to_parallel_archive_file_name(const char* prefix_name, int rank) {
  char buf[256];
  MADNESS_ASSERT(strlen(prefix_name) + 7 <= sizeof(buf));
  sprintf(buf, "%s.%5.5d", prefix_name, rank);
  return buf;
}
}  // namespace

BOOST_FIXTURE_TEST_SUITE(array_suite, ArrayFixture)

BOOST_AUTO_TEST_CASE(constructors) {
  // Construct a dense array
  BOOST_REQUIRE_NO_THROW(ArrayN ad(world, tr));
  ArrayN ad(world, tr);

  // Check that none of the tiles have been set.
  for (ArrayN::const_iterator it = ad.begin(); it != ad.end(); ++it)
    BOOST_CHECK(!it->probe());

  // Construct a sparse array
  BOOST_REQUIRE_NO_THROW(
      SpArrayN as(world, tr, TiledArray::SparseShape<float>(shape_tensor, tr)));
  SpArrayN as(world, tr, TiledArray::SparseShape<float>(shape_tensor, tr));

  // Check that none of the tiles have been set.
  for (SpArrayN::const_iterator it = as.begin(); it != as.end(); ++it)
    BOOST_CHECK(!it->probe());

  // now fill it
  BOOST_REQUIRE_NO_THROW(as.fill(1));

  // Construct a sparse array from another sparse array
  {
    auto op = [](auto& result, const auto& input) { result = input.clone(); };
    BOOST_REQUIRE_NO_THROW(SpArrayN as1(as, op));
  }
}

BOOST_AUTO_TEST_CASE(single_tile_initializer_list_ctors) {
  // Create a vector with an initializer list
  {
    detail::vector_il<double> il{1, 2, 3};
    TArray<double> a_vector(world, il);
    for (typename TArray<double>::value_type tile : a_vector) {
      auto itr = tile.begin();
      for (auto i : il) {
        BOOST_CHECK_EQUAL(i, *itr);
        ++itr;
      }
    }
  }

  // Create a matrix with an initializer list
  {
    detail::matrix_il<double> il{{1, 2, 3}, {4, 5, 6}};
    TArray<double> a_matrix(world, il);
    for (typename TArray<double>::value_type tile : a_matrix) {
      auto itr = tile.begin();
      for (auto i : il) {
        for (auto j : i) {
          BOOST_CHECK_EQUAL(j, *itr);
          ++itr;
        }
      }
    }
  }

  // Create a rank 3 tensor with an initializer list
  {
    detail::tensor3_il<double> il{{{
                                       1,
                                       2,
                                   },
                                   {3, 4}},
                                  {{5, 6}, {7, 8}}};
    TArray<double> a_tensor3(world, il);
    for (typename TArray<double>::value_type tile : a_tensor3) {
      auto itr = tile.begin();
      for (auto i : il) {
        for (auto j : i) {
          for (auto k : j) {
            BOOST_CHECK_EQUAL(k, *itr);
            ++itr;
          }
        }
      }
    }
  }

  // Create a rank 4 tensor with an initializer list
  {
    detail::tensor4_il<double> il{{{{
                                        1,
                                        2,
                                    },
                                    {3, 4}},
                                   {{5, 6}, {7, 8}}}};
    TArray<double> a_tensor4(world, il);
    for (typename TArray<double>::value_type tile : a_tensor4) {
      auto itr = tile.begin();
      for (auto i : il) {
        for (auto j : i) {
          for (auto k : j) {
            for (auto l : k) {
              BOOST_CHECK_EQUAL(l, *itr);
              ++itr;
            }
          }
        }
      }
    }
  }

  // Create a rank 5 tensor with an initializer list
  {
    detail::tensor5_il<double> il{{{{{
                                         1,
                                         2,
                                     },
                                     {3, 4}},
                                    {{5, 6}, {7, 8}}}}};
    TArray<double> a_tensor5(world, il);
    for (typename TArray<double>::value_type tile : a_tensor5) {
      auto itr = tile.begin();
      for (auto i : il) {
        for (auto j : i) {
          for (auto k : j) {
            for (auto l : k) {
              for (auto m : l) {
                BOOST_CHECK_EQUAL(m, *itr);
                ++itr;
              }
            }
          }
        }
      }
    }
  }

  // Create a rank 6 tensor with an initializer list
  {
    detail::tensor6_il<double> il{{{{{{
                                          1,
                                          2,
                                      },
                                      {3, 4}},
                                     {{5, 6}, {7, 8}}}}}};
    TArray<double> a_tensor6(world, il);
    for (typename TArray<double>::value_type tile : a_tensor6) {
      auto itr = tile.begin();
      for (auto i : il) {
        for (auto j : i) {
          for (auto k : j) {
            for (auto l : k) {
              for (auto m : l) {
                for (auto n : m) {
                  BOOST_CHECK_EQUAL(n, *itr);
                  ++itr;
                }
              }
            }
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(multi_tile_initializer_list_ctors) {
  // Create a vector with an initializer list
  {
    detail::vector_il<double> il{1, 2, 3};
    TiledRange tr{{0, 1, 3}};
    TArray<double> a_vector(world, tr, il);
    BOOST_CHECK_EQUAL(a_vector.size(), 2);
  }

  {
    detail::matrix_il<double> il{{1, 2, 3}, {4, 5, 6}};
    TiledRange tr{{0, 1, 2}, {0, 1, 3}};
    TArray<double> a_matrix(world, tr, il);
    BOOST_CHECK_EQUAL(a_matrix.size(), 4);
  }

  {
    detail::tensor3_il<double> il{{{1, 2, 3}, {4, 5, 6}},
                                  {{7, 8, 9}, {10, 11, 12}}};
    TiledRange tr{{0, 1, 2}, {0, 1, 2}, {0, 1, 3}};
    TArray<double> a_tensor(world, tr, il);
    BOOST_CHECK_EQUAL(a_tensor.size(), 8);
  }

  {
    detail::tensor4_il<double> il{
        {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},

        {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    TiledRange tr{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 3}};
    TArray<double> a_tensor(world, tr, il);
    BOOST_CHECK_EQUAL(a_tensor.size(), 16);
  }

  {
    detail::tensor5_il<double> il{
        {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},
         {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}},

        {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},
         {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}}};
    TiledRange tr{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 3}};
    TArray<double> a_tensor(world, tr, il);
    BOOST_CHECK_EQUAL(a_tensor.size(), 32);
  }

  {
    detail::tensor6_il<double> il{
        {{{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},
          {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}},
         {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},
          {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}}},

        {{{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},
          {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}},
         {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}},
          {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}}}};
    TiledRange tr{{0, 1, 2}, {0, 1, 2}, {0, 1, 2},
                  {0, 1, 2}, {0, 1, 2}, {0, 1, 3}};
    TArray<double> a_tensor(world, tr, il);
    BOOST_CHECK_EQUAL(a_tensor.size(), 64);
  }
}

BOOST_AUTO_TEST_CASE(all_owned) {
  std::size_t count = 0ul;
  for (std::size_t i = 0ul; i < tr.tiles_range().volume(); ++i)
    if (a.owner(i) == GlobalFixture::world->rank()) ++count;
  world.gop.sum(count);

  // Check that all tiles are in the array
  BOOST_CHECK_EQUAL(tr.tiles_range().volume(), count);
}

BOOST_AUTO_TEST_CASE(owner) {
  // Test to make sure everyone agrees who owns which tiles.
  std::shared_ptr<ProcessID> group_owner(new ProcessID[world.size()],
                                         std::default_delete<ProcessID[]>());

  ordinal_type o = 0;
  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it, ++o) {
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
    BOOST_CHECK(
        (std::find_if(group_owner.get(), group_owner.get() + world.size(),
                      std::bind(std::not_equal_to<ProcessID>(), owner,
                                std::placeholders::_1)) ==
         (group_owner.get() + world.size())));
  }
}

BOOST_AUTO_TEST_CASE(is_local) {
  // Test to make sure everyone agrees who owns which tiles.

  ordinal_type o = 0;
  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it, ++o) {
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

BOOST_AUTO_TEST_CASE(find_local) {
  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it) {
    if (a.is_local(*it)) {
      Future<ArrayN::value_type> tile = a.find(*it);

      BOOST_CHECK(tile.probe());

      const int value = world.rank() + 1;
      for (ArrayN::value_type::iterator it = tile.get().begin();
           it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, value);
    }
  }

  for (auto&& tile_idx : a.range()) {
    if (a.is_local(tile_idx)) {
      const Future<ArrayN::value_type>& const_tile_fut = a.find_local(tile_idx);
      Future<ArrayN::value_type>& nonconst_tile_fut = a.find_local(tile_idx);

      const int value = world.rank() + 1;
      BOOST_CHECK(const_tile_fut.probe());
      const auto& const_tile_ref = const_tile_fut.get();
      for (auto&& val : const_tile_ref) {
        BOOST_CHECK_EQUAL(val, value);
      }

      BOOST_CHECK(nonconst_tile_fut.probe());
      const auto& nonconst_tile_ref = nonconst_tile_fut.get();
      for (auto&& val : nonconst_tile_ref) {
        BOOST_CHECK_EQUAL(val, value);
      }

    } else {
#ifdef TA_EXCEPTION_ERROR
      // Check that an exception is thrown when using a default constructed
      // object
      BOOST_CHECK_THROW(a.find_local(tile_idx), TiledArray::Exception);
#endif  // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_CASE(find_remote) {
  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it) {
    if (!a.is_local(*it)) {
      Future<ArrayN::value_type> tile = a.find(*it);

      const int owner = a.owner(*it);
      for (ArrayN::value_type::iterator it = tile.get().begin();
           it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, owner + 1);
    }
  }
}

BOOST_AUTO_TEST_CASE(fill_tiles) {
  ArrayN a(world, tr);

  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it) {
    if (a.is_local(*it)) {
      a.set(*it, 0);  // Fill the tile at *it (the index) with 0

      Future<ArrayN::value_type> tile = a.find(*it);

      // Check that the range for the constructed tile is correct.
      BOOST_CHECK_EQUAL(tile.get().range(), tr.make_tile_range(*it));

      for (ArrayN::value_type::iterator it = tile.get().begin();
           it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(assign_tiles) {
  std::vector<int> data;
  ArrayN a(world, tr);

  for (ArrayN::range_type::const_iterator it = a.range().begin();
       it != a.range().end(); ++it) {
    ArrayN::trange_type::range_type range = a.trange().make_tile_range(*it);
    if (a.is_local(*it)) {
      if (data.size() < range.volume()) data.resize(range.volume(), 1);
      a.set(*it, data.begin());

      Future<ArrayN::value_type> tile = a.find(*it);
      BOOST_CHECK(tile.probe());

      // Check that the range for the constructed tile is correct.
      BOOST_CHECK_EQUAL(tile.get().range(), tr.make_tile_range(*it));

      for (ArrayN::value_type::iterator it = tile.get().begin();
           it != tile.get().end(); ++it)
        BOOST_CHECK_EQUAL(*it, 1);
    }
  }
}

BOOST_AUTO_TEST_CASE(clone) {
  std::vector<int> data;
  ArrayN a(world, tr);

  // Init tiles with random data
  a.init_tiles([](const Range& range) -> TensorI {
    std::default_random_engine generator(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(0, 100);

    TensorI tile(range);
    tile.inplace_unary([&](int& value) { value = distribution(generator); });
    return tile;
  });

  ArrayN ca;
  BOOST_REQUIRE_NO_THROW(ca = TiledArray::clone(a));

  // Check array data are equal
  BOOST_CHECK(!(ca.id() == a.id()));
  BOOST_CHECK_EQUAL(ca.world().id(), a.world().id());
  BOOST_CHECK_EQUAL(ca.trange(), a.trange());
  BOOST_CHECK_EQUAL_COLLECTIONS(ca.pmap()->begin(), ca.pmap()->end(),
                                a.pmap()->begin(), a.pmap()->end());

  // Check that array tiles are equal
  for (typename ArrayN::ordinal_type index = 0ul; index < a.size(); ++index) {
    // Check that per-tile information is the same
    BOOST_CHECK_EQUAL(ca.owner(index), a.owner(index));
    BOOST_CHECK_EQUAL(ca.is_local(index), a.is_local(index));
    BOOST_CHECK_EQUAL(ca.is_zero(index), a.is_zero(index));

    // Skip non-local tiles
    if (!a.is_local(index)) continue;

    const TensorI t = a.find(index).get();
    const TensorI ct = ca.find(index).get();

    // Check that tile data is the same but held in different memory locations
    BOOST_CHECK_NE(ct.data(), t.data());
    BOOST_CHECK_EQUAL(ct.range(), t.range());
    BOOST_CHECK_EQUAL_COLLECTIONS(ct.begin(), ct.end(), t.begin(), t.end());
  }
}

BOOST_AUTO_TEST_CASE(truncate) {
  auto b_trunc0 = b.clone();
  BOOST_CHECK_NO_THROW(b_trunc0.truncate());
  auto b_trunc1 = b.clone();
  BOOST_CHECK_NO_THROW(
      b_trunc1.truncate(std::numeric_limits<typename decltype(
                            b)::shape_type::value_type>::max()));
  BOOST_CHECK(std::distance(b_trunc1.begin(), b_trunc1.end()) == 0);
}

BOOST_AUTO_TEST_CASE(make_replicated) {
  // Get a copy of the original process map
  std::shared_ptr<ArrayN::pmap_interface> distributed_pmap = a.pmap();

  // Convert array to a replicated array.
  BOOST_REQUIRE_NO_THROW(a.make_replicated());

  if (GlobalFixture::world->size() == 1)
    BOOST_CHECK(!a.pmap()->is_replicated());
  else
    BOOST_CHECK(a.pmap()->is_replicated());

  // Check that all the data is local
  for (std::size_t i = 0; i < a.size(); ++i) {
    BOOST_CHECK(a.is_local(i));
    BOOST_CHECK_EQUAL(a.pmap()->owner(i), GlobalFixture::world->rank());
    Future<ArrayN::value_type> tile = a.find(i);
    BOOST_CHECK_EQUAL(tile.get().range(), a.trange().make_tile_range(i));
    for (ArrayN::value_type::const_iterator it = tile.get().begin();
         it != tile.get().end(); ++it)
      BOOST_CHECK_EQUAL(*it, distributed_pmap->owner(i) + 1);
  }
}

BOOST_AUTO_TEST_CASE(serialization_by_tile) {
  decltype(a) acopy(a.world(), a.trange(), a.shape());

  const auto nproc = world.size();
  if (nproc > 1) {  // use BufferOutputArchive if more than 1 node ...
    const std::size_t buf_size = 1000000;  // "big enough" buffer
    auto buf = std::make_unique<unsigned char[]>(buf_size);
    madness::archive::BufferOutputArchive oar(buf.get(), buf_size);

    for (auto tile : a) {
      BOOST_REQUIRE_NO_THROW(oar & tile.get());
    }

    std::size_t nbyte = oar.size();
    BOOST_REQUIRE(nbyte < buf_size);
    oar.close();

    madness::archive::BufferInputArchive iar(buf.get(), nbyte);
    for (auto tile : acopy) {
      decltype(acopy)::value_type tile_value;
      BOOST_REQUIRE_NO_THROW(iar & tile_value);
      tile.future().set(std::move(tile_value));
    }
    iar.close();

    buf.reset();
  } else {  // ... else use TextFstreamOutputArchive
    char archive_file_name[] = "tmp.XXXXXX";
    mktemp(archive_file_name);
    madness::archive::TextFstreamOutputArchive oar(archive_file_name);

    for (auto tile : a) {
      BOOST_REQUIRE_NO_THROW(oar & tile.get());
    }

    oar.close();

    madness::archive::TextFstreamInputArchive iar(archive_file_name);
    for (auto tile : acopy) {
      decltype(acopy)::value_type tile_value;
      BOOST_REQUIRE_NO_THROW(iar & tile_value);
      tile.future().set(std::move(tile_value));
    }
    iar.close();

    std::remove(archive_file_name);
  }

  BOOST_CHECK_EQUAL(a.trange(), acopy.trange());
  BOOST_REQUIRE(a.shape() == acopy.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), acopy.begin(), acopy.end());
}

BOOST_AUTO_TEST_CASE(dense_serialization) {
  char archive_file_name[] = "tmp.XXXXXX";
  mktemp(archive_file_name);
  madness::archive::BinaryFstreamOutputArchive oar(archive_file_name);
  a.serialize(oar);

  oar.close();

  madness::archive::BinaryFstreamInputArchive iar(archive_file_name);
  decltype(a) aread;
  aread.serialize(iar);

  BOOST_CHECK_EQUAL(aread.trange(), a.trange());
  BOOST_REQUIRE(aread.shape() == a.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(aread.begin(), aread.end(), a.begin(), a.end());
  std::remove(archive_file_name);
}

BOOST_AUTO_TEST_CASE(sparse_serialization) {
  char archive_file_name[] = "tmp.XXXXXX";
  mktemp(archive_file_name);
  madness::archive::BinaryFstreamOutputArchive oar(archive_file_name);
  b.serialize(oar);

  oar.close();

  madness::archive::BinaryFstreamInputArchive iar(archive_file_name);
  decltype(b) bread;
  bread.serialize(iar);

  BOOST_CHECK_EQUAL(bread.trange(), b.trange());
  BOOST_REQUIRE(bread.shape() == b.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(bread.begin(), bread.end(), b.begin(), b.end());
  std::remove(archive_file_name);
}

BOOST_AUTO_TEST_CASE(parallel_serialization) {
  const int nio = 1;  // use 1 rank for I/O
  char archive_file_prefix_name[] = "tmp.XXXXXX";
  mktemp(archive_file_prefix_name);
  madness::archive::ParallelOutputArchive oar(world, archive_file_prefix_name,
                                              nio);
  oar& a;
  oar.close();

  madness::archive::ParallelInputArchive iar(world, archive_file_prefix_name,
                                             nio);
  decltype(a) aread;
  aread.load(world, iar);

  BOOST_CHECK_EQUAL(aread.trange(), a.trange());
  BOOST_REQUIRE(aread.shape() == a.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(aread.begin(), aread.end(), a.begin(), a.end());
  if (world.rank() < nio) {
    std::remove(
        to_parallel_archive_file_name(archive_file_prefix_name, world.rank())
            .c_str());
  }
}

BOOST_AUTO_TEST_CASE(parallel_sparse_serialization) {
  const int nio = 1;  // use 1 rank for 1
  char archive_file_prefix_name[] = "tmp.XXXXXX";
  mktemp(archive_file_prefix_name);
  madness::archive::ParallelOutputArchive oar(world, archive_file_prefix_name,
                                              nio);
  oar& b;
  oar.close();

  madness::archive::ParallelInputArchive iar(world, archive_file_prefix_name,
                                             nio);
  decltype(b) bread;
  bread.load(world, iar);

  BOOST_CHECK_EQUAL(bread.trange(), b.trange());
  BOOST_REQUIRE(bread.shape() == b.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(bread.begin(), bread.end(), b.begin(), b.end());
  if (world.rank() < nio) {
    std::remove(
        to_parallel_archive_file_name(archive_file_prefix_name, world.rank())
            .c_str());
  }
}

BOOST_AUTO_TEST_CASE(issue_225) {
  TiledRange1 TR0{0, 3, 8, 10};
  TiledRange1 TR1{0, 4, 7, 10};
  TiledRange TR{TR0, TR1};
  Tensor<float> shape_tensor(TR.tiles_range(), 0.0);
  shape_tensor(0, 0) = 1.0;
  shape_tensor(0, 1) = 1.0;
  shape_tensor(1, 1) = 1.0;
  shape_tensor(2, 2) = 1.0;
  SparseShape<float> shape(shape_tensor, TR);
  TSpArrayD S(world, TR, shape);
  TSpArrayD St;
  S.fill(1.0);

  char archive_file_name[] = "tmp.XXXXXX";
  mktemp(archive_file_name);
  madness::archive::BinaryFstreamOutputArchive oar(archive_file_name);
  St("i,j") = S("j,i");
  BOOST_REQUIRE_NO_THROW(oar & S);
  BOOST_REQUIRE_NO_THROW(oar & St);
  oar.close();

  madness::archive::BinaryFstreamInputArchive iar(archive_file_name);
  decltype(S) S_read;
  decltype(St) St_read;
  iar& S_read& St_read;

  BOOST_CHECK_EQUAL(S_read.trange(), S.trange());
  BOOST_REQUIRE(S_read.shape() == S.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(S_read.begin(), S_read.end(), S.begin(),
                                S.end());
  BOOST_CHECK_EQUAL(St_read.trange(), St.trange());
  BOOST_REQUIRE(St_read.shape() == St.shape());
  BOOST_CHECK_EQUAL_COLLECTIONS(St_read.begin(), St_read.end(), St.begin(),
                                St.end());
  std::remove(archive_file_name);
}

BOOST_AUTO_TEST_SUITE_END()
