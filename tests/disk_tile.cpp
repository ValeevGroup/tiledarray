/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 *  disk_tile.cpp
 *
 *  Tests of disk-backed (lazy) tiles and arrays.
 */

#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace TA = TiledArray;

namespace test {

/// A lazy tile that simply holds its evaluated tile in memory: the minimal
/// tile type whose `eval_type` differs from itself. Reference point for the
/// behavior of lazy tiles that is independent of where the data lives.
template <typename Tile>
class InMemLazyTile {
 public:
  using eval_type = Tile;
  using value_type = typename Tile::value_type;
  using numeric_type = typename Tile::numeric_type;
  using scalar_type = typename Tile::scalar_type;
  using range_type = typename Tile::range_type;

  InMemLazyTile() = default;
  explicit InMemLazyTile(const Tile& tile)
      : tile_(std::make_shared<Tile>(tile)) {}

  explicit operator Tile() const { return *tile_; }

  template <typename Archive>
  void serialize(Archive& ar) {
    if constexpr (madness::is_output_archive_v<Archive>) {
      ar&* tile_;
    } else {
      Tile t;
      ar & t;
      tile_ = std::make_shared<Tile>(std::move(t));
    }
  }

 private:
  std::shared_ptr<Tile> tile_;
};

}  // namespace test

namespace {

using ArenaInner = TA::ArenaTensor<double, TA::Range>;
using ArenaToT = TA::Tensor<ArenaInner>;
using ToTArray = TA::DistArray<ArenaToT, TA::SparsePolicy>;
using FlatArray = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;

/// Arena tensor-of-tensors array with deterministic, seed-shifted values.
ToTArray make_tot(TA::World& world, const TA::TiledRange& tr,
                  std::function<TA::Range(std::size_t)> inner, double seed) {
  ToTArray tot(world, tr);
  tot.init_tiles([=](TA::Range const& r) {
    ArenaToT t = TA::detail::arena_outer_init<ArenaToT>(r, 1, inner);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto n = c.range().volume();
      for (std::size_t a = 0; a < n; ++a)
        c.data()[a] = seed + 0.001 * static_cast<double>(
                                         (o * 7 + a + r.lobound()[0]) % 13);
    }
    return t;
  });
  world.gop.fence();
  return tot;
}

/// Copies \p array into an array of tiles wrapped by \p Lazy, with the same
/// TiledRange, shape and process map.
template <template <typename> class Lazy, typename Tile>
TA::DistArray<Lazy<Tile>, TA::SparsePolicy> wrap_tiles(
    const TA::DistArray<Tile, TA::SparsePolicy>& array) {
  TA::DistArray<Lazy<Tile>, TA::SparsePolicy> result(
      array.world(), array.trange(), array.shape(), array.pmap());
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (array.is_zero(*it)) continue;
    result.set(*it, Lazy<Tile>(array.find_local(*it).get()));
  }
  array.world().gop.fence();
  return result;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(disk_tile_suite)

BOOST_AUTO_TEST_CASE(lazy_tile_nestedness) {
  // a lazy tile is as nested as the tile it evaluates to ...
  static_assert(
      TA::detail::is_tensor_of_tensor_v<test::InMemLazyTile<ArenaToT>>);
  static_assert(!TA::detail::is_tensor_of_tensor_v<
                test::InMemLazyTile<TA::Tensor<double>>>);
  // ... but it is not itself a tensor
  static_assert(
      !TA::detail::is_tensor_v<test::InMemLazyTile<TA::Tensor<double>>>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(lazy_tot_annotation) {
  auto& world = TA::get_default_world();
  TA::TiledRange const tr{{0, 2, 4}, {0, 3, 6}};
  auto const tot = make_tot(
      world, tr, [](std::size_t) { return TA::Range{3}; }, 1.0);
  auto const lazy = wrap_tiles<test::InMemLazyTile>(tot);

  // a nested annotation on an array of lazy nested tiles is accepted
  ToTArray r;
  BOOST_REQUIRE_NO_THROW(r("i,j;a") = lazy("i,j;a"));
  ToTArray d;
  d("i,j;a") = r("i,j;a") - tot("i,j;a");
  BOOST_CHECK_EQUAL(TA::norm2(d), 0.0);
  BOOST_CHECK(TA::norm2(tot) > 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
