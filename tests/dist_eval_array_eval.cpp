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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  dist_eval_array_eval.cpp
 *  Sep 15, 2013
 *
 */

#include <array_fixture.h>

#include "TiledArray/dist_eval/array_eval.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <boost/mpl/list.hpp>

#include <chrono>
#include <thread>

#include <utility>

using namespace TiledArray;
using TiledArray::detail::Noop;
using TiledArray::detail::Scal;
using TiledArray::detail::UnaryWrapper;

// Lazy tiles that generate an integer tile on demand: every element is the
// tile's first element lobound plus one. `SyncLazyTile` converts to its tile
// directly, `AsyncLazyTile` to a Future of it (the pattern of a lazy tile whose
// evaluation is a task).
template <bool Async>
class GeneratedTile {
 public:
  using eval_type = TensorI;
  using value_type = TensorI::value_type;
  using numeric_type = TensorI::numeric_type;
  using scalar_type = TensorI::scalar_type;
  using range_type = TensorI::range_type;

  GeneratedTile() = default;
  explicit GeneratedTile(const range_type& range) : range_(range) {}

  const range_type& range() const { return range_; }
  bool empty() const { return range_.volume() == 0; }

  static value_type value(const range_type& range) {
    return range.lobound()[0] + 1;
  }

  explicit operator TensorI() const
    requires(!Async)
  {
    return TensorI(range_, value(range_));
  }
  // the Future is fulfilled by a task that finishes late, after the
  // evaluator that requested it is done, so that whatever the request
  // chains onto the Future must not depend on the evaluator still existing
  explicit operator madness::Future<TensorI>() const
    requires Async
  {
    return TiledArray::get_default_world().taskq.add([range = range_] {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      return TensorI(range, value(range));
    });
  }

  template <typename Archive>
  void serialize(Archive& ar) {
    ar & range_;
  }

 private:
  range_type range_;
};
using SyncLazyTile = GeneratedTile<false>;
using AsyncLazyTile = GeneratedTile<true>;
using LazyTiles = boost::mpl::list<SyncLazyTile, AsyncLazyTile>;

// Array evaluator fixture
struct ArrayEvalImplFixture : public TiledRangeFixture {
  ArrayEvalImplFixture() : array(*GlobalFixture::world, tr) {
    // Fill array with random data
    for (TArrayI::iterator it = array.begin(); it != array.end(); ++it) {
      TArrayI::value_type tile(array.trange().make_tile_range(it.index()));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
  }

  ~ArrayEvalImplFixture() {}

  static UnaryWrapper<Noop<TensorI, TensorI, true>> make_array_noop(
      const Permutation& perm = Permutation()) {
    return UnaryWrapper<Noop<TensorI, TensorI, true>>(
        Noop<TensorI, TensorI, true>(), perm);
  }

  static UnaryWrapper<Scal<TensorI, TensorI, int, true>> make_array_scal(
      const int factor, const Permutation& perm = Permutation()) {
    return UnaryWrapper<Scal<TensorI, TensorI, int, true>>(
        Scal<TensorI, TensorI, int, true>(factor), perm);
  }

  static UnaryWrapper<Scal<TensorI, TensorI, int, false>> make_scal(
      const int factor, const Permutation& perm = Permutation()) {
    return UnaryWrapper<Scal<TensorI, TensorI, int, false>>(
        Scal<TensorI, TensorI, int, false>(factor), perm);
  }

  template <typename Tile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<
      TiledArray::detail::LazyArrayTile<
          typename DistArray<Tile, Policy>::value_type, Op>,
      Policy>
  make_array_eval(
      const DistArray<Tile, Policy>& array, TiledArray::World& world,
      const typename TiledArray::detail::DistEval<Tile, Policy>::shape_type&
          shape,
      const std::shared_ptr<const typename TiledArray::detail::DistEval<
          Tile, Policy>::pmap_interface>& pmap,
      const Permutation& perm, const Op& op) {
    typedef TiledArray::detail::ArrayEvalImpl<DistArray<Tile, Policy>, Op,
                                              Policy>
        impl_type;
    return TiledArray::detail::DistEval<
        TiledArray::detail::LazyArrayTile<
            typename TiledArray::DistArray<Tile, Policy>::value_type, Op>,
        Policy>(std::shared_ptr<impl_type>(new impl_type(
        array, world, (perm ? perm * array.trange() : array.trange()), shape,
        pmap, perm, op)));
  }

  TArrayI array;
};  // ArrayEvalFixture

BOOST_FIXTURE_TEST_SUITE(array_eval_suite, ArrayEvalImplFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  BOOST_REQUIRE_NO_THROW(make_array_eval(array, array.world(), DenseShape(),
                                         array.pmap(), Permutation(),
                                         make_array_scal(3)));

  auto dist_eval =
      make_array_eval(array, array.world(), DenseShape(), array.pmap(),
                      Permutation(), make_array_scal(3));

  BOOST_CHECK_EQUAL(&dist_eval.world(), GlobalFixture::world);
  BOOST_CHECK(dist_eval.pmap() == array.pmap());
  BOOST_CHECK_EQUAL(dist_eval.range(), tr.tiles_range());
  BOOST_CHECK_EQUAL(dist_eval.trange(), tr);
  BOOST_CHECK_EQUAL(dist_eval.size(), tr.tiles_range().volume());
  BOOST_CHECK(dist_eval.is_dense());
  for (std::size_t i = 0; i < tr.tiles_range().volume(); ++i)
    BOOST_CHECK(!dist_eval.is_zero(i));
}

BOOST_AUTO_TEST_CASE(eval_scale) {
  auto dist_eval = make_array_eval(array, array.world(), DenseShape(),
                                   array.pmap(), Permutation(), make_scal(3));
  using dist_eval_type = decltype(dist_eval);

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());

  // Check that each tile has been properly scaled.
  for (auto index : *dist_eval.pmap()) {
    // Get the original type
    TArrayI::value_type array_tile = array.find(index);

    // Get the array evaluator tile.
    Future<dist_eval_type::value_type> impl_tile;
    BOOST_REQUIRE_NO_THROW(impl_tile = dist_eval.get(index));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(
        eval_tile = static_cast<dist_eval_type::eval_type>(impl_tile.get()));

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), array_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], 3 * array_tile[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(eval_permute) {
  // Create permutation to be applied in the array evaluations
  std::array<std::size_t, GlobalFixture::dim> p;
  for (std::size_t i = 0; i < p.size(); ++i)
    p[i] = (i + p.size() - 1) % p.size();
  const Permutation perm(p.begin(), p.end());

  // Construct and evaluate
  auto dist_eval = make_array_eval(array, array.world(), DenseShape(),
                                   array.pmap(), perm, make_array_noop(perm));
  using dist_eval_type = decltype(dist_eval);

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());

  // Check that each tile has been moved to the correct location and has been
  // properly permuted.
  const Permutation inv_perm = -perm;
  for (auto index : *dist_eval.pmap()) {
    // Get the original type
    TArrayI::value_type array_tile =
        array.find(inv_perm * dist_eval.range().idx(index));

    // Get the corresponding array evaluator tile.
    Future<dist_eval_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(index));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(
        eval_tile = static_cast<dist_eval_type::eval_type>(tile.get()););

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), perm * array_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[perm * array_tile.range().idx(i)],
                        array_tile[i]);
    }
  }
}

// Evaluates the block expression \p expr as a consuming expression (a
// product, a sum, an assignment) would: through its engine's distributed
// evaluator, converting each lazy tile to the eval tile.
template <typename BlkExpr>
auto block_eval_tiles(BlkExpr const& expr, World& world) {
  typename BlkExpr::engine_type engine(expr);
  engine.init(world, {},
              TiledArray::expressions::BipartiteIndexList(
                  TiledArray::detail::dummy_annotation(GlobalFixture::dim)));
  auto dist_eval = engine.make_dist_eval();
  dist_eval.eval();
  using dist_eval_type = decltype(dist_eval);
  std::vector<std::pair<typename dist_eval_type::ordinal_type,
                        typename dist_eval_type::eval_type>>
      tiles;
  for (auto ord : *dist_eval.pmap())
    tiles.emplace_back(ord, static_cast<typename dist_eval_type::eval_type>(
                                dist_eval.get(ord).get()));
  dist_eval.wait();
  return std::make_pair(dist_eval.trange(), std::move(tiles));
}

// A block of a CONST array that keeps the parent's element coordinates
// (preserve_lobound) relocates nothing: a consumer receives the parent's
// tiles themselves, not copies, exactly as it does for the plain (non-block)
// array operand. The array being const is the promise that nobody mutates
// those tiles through tile-level access while the expression is alive.
// Only a tile the parent holds on this rank can be shared; a tile fetched
// from another rank is a copy by construction, and is checked by value.
BOOST_AUTO_TEST_CASE(const_block_preserve_lobound_shares_tiles) {
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;  // a proper sub-block, so block and parent tile indices differ
  auto const annot = TiledArray::detail::dummy_annotation(GlobalFixture::dim);
  auto const [block_trange, tiles] = block_eval_tiles(
      std::as_const(array)(annot).block(lo, hi, TiledArray::preserve_lobound),
      array.world());
  for (auto const& [ord, eval_tile] : tiles) {
    auto parent_index = block_trange.tiles_range().idx(ord);
    for (std::size_t d = 0; d < parent_index.size(); ++d)
      parent_index[d] += lo[d];
    TArrayI::value_type const parent_tile = array.find(parent_index);
    BOOST_CHECK_EQUAL(eval_tile.range(), parent_tile.range());
    if (array.is_local(parent_index)) {
      // same buffer: no copy was made
      BOOST_CHECK_EQUAL(static_cast<void const*>(eval_tile.data()),
                        static_cast<void const*>(parent_tile.data()));
    } else {
      BOOST_REQUIRE_EQUAL(eval_tile.size(), parent_tile.size());
      for (std::size_t i = 0ul; i < eval_tile.size(); ++i)
        BOOST_CHECK_EQUAL(eval_tile[i], parent_tile[i]);
    }
  }
}

// The same block of a MUTABLE array keeps producing copies: its consumers
// may take its tiles over (consume them in place), which must not reach the
// array.
BOOST_AUTO_TEST_CASE(mutable_block_preserve_lobound_copies_tiles) {
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;
  auto const annot = TiledArray::detail::dummy_annotation(GlobalFixture::dim);
  auto const [block_trange, tiles] = block_eval_tiles(
      array(annot).block(lo, hi, TiledArray::preserve_lobound), array.world());
  for (auto const& [ord, eval_tile] : tiles) {
    auto parent_index = block_trange.tiles_range().idx(ord);
    for (std::size_t d = 0; d < parent_index.size(); ++d)
      parent_index[d] += lo[d];
    TArrayI::value_type const parent_tile = array.find(parent_index);
    BOOST_CHECK_EQUAL(eval_tile.range(), parent_tile.range());
    BOOST_CHECK_NE(static_cast<void const*>(eval_tile.data()),
                   static_cast<void const*>(parent_tile.data()));
    BOOST_REQUIRE_EQUAL(eval_tile.size(), parent_tile.size());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i)
      BOOST_CHECK_EQUAL(eval_tile[i], parent_tile[i]);
  }
}

// Assigning a const block to an array gives that array its own tiles: the
// shared tiles the block hands its consumers must not become the tiles of a
// mutable array.
BOOST_AUTO_TEST_CASE(assignment_of_const_block_does_not_alias) {
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;
  auto const annot = TiledArray::detail::dummy_annotation(GlobalFixture::dim);
  TArrayI result;
  BOOST_REQUIRE_NO_THROW(result(annot) = std::as_const(array)(annot).block(
                             lo, hi, TiledArray::preserve_lobound));
  for (auto const& ord : *result.pmap()) {
    auto parent_index = result.trange().tiles_range().idx(ord);
    for (std::size_t d = 0; d < parent_index.size(); ++d)
      parent_index[d] += lo[d];
    TArrayI::value_type const result_tile = result.find(ord);
    TArrayI::value_type const parent_tile = array.find(parent_index);
    BOOST_CHECK_EQUAL(result_tile.range(), parent_tile.range());
    if (array.is_local(parent_index))  // a remote find is a copy anyway
      BOOST_CHECK_NE(static_cast<void const*>(result_tile.data()),
                     static_cast<void const*>(parent_tile.data()));
    BOOST_REQUIRE_EQUAL(result_tile.size(), parent_tile.size());
    for (std::size_t i = 0ul; i < result_tile.size(); ++i)
      BOOST_CHECK_EQUAL(result_tile[i], parent_tile[i]);
  }
}

// A block rebased to the origin relocates every tile's range; the parent's
// own tiles keep their ranges and the values are unchanged.
BOOST_AUTO_TEST_CASE(block_rebased_relocates_tiles) {
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;
  auto const annot = TiledArray::detail::dummy_annotation(GlobalFixture::dim);
  auto const [block_trange, tiles] =
      block_eval_tiles(array(annot).block(lo, hi), array.world());
  for (auto const& [ord, eval_tile] : tiles) {
    auto const block_index = block_trange.tiles_range().idx(ord);
    auto parent_index = block_index;
    for (std::size_t d = 0; d < parent_index.size(); ++d)
      parent_index[d] += lo[d];
    TArrayI::value_type const parent_tile = array.find(parent_index);
    BOOST_CHECK_EQUAL(eval_tile.range(),
                      block_trange.make_tile_range(block_index));
    BOOST_CHECK_EQUAL(parent_tile.range(), tr.make_tile_range(parent_index));
    BOOST_REQUIRE_EQUAL(eval_tile.size(), parent_tile.size());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i)
      BOOST_CHECK_EQUAL(eval_tile[i], parent_tile[i]);
  }
}

// Assigning a const block into a block of another array also gives that array
// its own tiles (this assignment converts the operand's tiles through its own
// shift op, not through the plain assignment's conversion).
BOOST_AUTO_TEST_CASE(block_assignment_of_const_block_does_not_alias) {
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;
  auto const annot = TiledArray::detail::dummy_annotation(GlobalFixture::dim);
  TArrayI result = array.clone();
  BOOST_REQUIRE_NO_THROW(result(annot).block(lo, hi) =
                             std::as_const(array)(annot).block(
                                 lo, hi, TiledArray::preserve_lobound));
  BlockRange const blk_range(tiles_range, lo, hi);
  for (auto const& ord : *result.pmap()) {
    auto const index = tiles_range.idx(ord);
    if (!blk_range.includes(index)) continue;
    TArrayI::value_type const result_tile = result.find(ord);
    TArrayI::value_type const parent_tile = array.find(ord);
    BOOST_CHECK_EQUAL(result_tile.range(), parent_tile.range());
    if (array.is_local(ord))  // a remote find is a copy anyway
      BOOST_CHECK_NE(static_cast<void const*>(result_tile.data()),
                     static_cast<void const*>(parent_tile.data()));
    BOOST_REQUIRE_EQUAL(result_tile.size(), parent_tile.size());
    for (std::size_t i = 0ul; i < result_tile.size(); ++i)
      BOOST_CHECK_EQUAL(result_tile[i], parent_tile[i]);
  }
}

// The assignment of a const block of a tensor-of-tensors array copies the
// inner tensors too: the result shares neither outer nor inner data with the
// operand.
BOOST_AUTO_TEST_CASE(assignment_of_const_tot_block_does_not_alias) {
  using inner_type = TA::Tensor<double>;
  using tile_type = TA::Tensor<inner_type>;
  DistArray<tile_type> tot(*GlobalFixture::world, tr);
  for (auto it = tot.begin(); it != tot.end(); ++it) {
    tile_type tile(tot.trange().make_tile_range(it.index()));
    for (auto& inner : tile)
      inner = inner_type(TA::Range{2}, GlobalFixture::world->rand() % 101);
    *it = tile;
  }

  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;
  auto const annot =
      TiledArray::detail::dummy_annotation(GlobalFixture::dim) + ";x";
  DistArray<tile_type> result;
  BOOST_REQUIRE_NO_THROW(result(annot) = std::as_const(tot)(annot).block(
                             lo, hi, TiledArray::preserve_lobound));
  for (auto const& ord : *result.pmap()) {
    auto parent_index = result.trange().tiles_range().idx(ord);
    for (std::size_t d = 0; d < parent_index.size(); ++d)
      parent_index[d] += lo[d];
    tile_type const result_tile = result.find(ord);
    tile_type const parent_tile = tot.find(parent_index);
    BOOST_CHECK_EQUAL(result_tile.range(), parent_tile.range());
    BOOST_REQUIRE_EQUAL(result_tile.size(), parent_tile.size());
    bool const local = tot.is_local(parent_index);
    if (local)  // a remote find is a copy anyway
      BOOST_CHECK_NE(static_cast<void const*>(result_tile.data()),
                     static_cast<void const*>(parent_tile.data()));
    for (std::size_t i = 0ul; i < result_tile.size(); ++i) {
      if (local)
        BOOST_CHECK_NE(static_cast<void const*>(result_tile[i].data()),
                       static_cast<void const*>(parent_tile[i].data()));
      BOOST_CHECK_EQUAL(result_tile[i], parent_tile[i]);
    }
  }
}

// An array of lazy tiles, synchronous or asynchronous, can be assigned (whole
// or as a const block) to an array of the evaluated tiles. The lazy tile
// conversion produces a fresh tile, so nothing is shared with the operand.
BOOST_AUTO_TEST_CASE_TEMPLATE(assignment_of_lazy_tiles, LazyTile, LazyTiles) {
  DistArray<LazyTile> lazy(*GlobalFixture::world, tr);
  for (auto it = lazy.begin(); it != lazy.end(); ++it)
    *it = LazyTile(lazy.trange().make_tile_range(it.index()));

  auto check = [](TArrayI const& result) {
    for (auto const& ord : *result.pmap()) {
      TArrayI::value_type const tile = result.find(ord);
      auto const expected = LazyTile::value(tile.range());
      for (std::size_t i = 0ul; i < tile.size(); ++i)
        BOOST_CHECK_EQUAL(tile[i], expected);
    }
  };

  auto const annot = TiledArray::detail::dummy_annotation(GlobalFixture::dim);
  {
    TArrayI result;
    BOOST_REQUIRE_NO_THROW(result(annot) = lazy(annot));
    BOOST_CHECK_EQUAL(result.trange(), tr);
    check(result);
  }
  {
    auto const& tiles_range = tr.tiles_range();
    container::svector<long> lo(tiles_range.lobound().begin(),
                                tiles_range.lobound().end());
    container::svector<long> hi(tiles_range.upbound().begin(),
                                tiles_range.upbound().end());
    lo[0] += 1;
    TArrayI result;
    BOOST_REQUIRE_NO_THROW(result(annot) = std::as_const(lazy)(annot).block(
                               lo, hi, TiledArray::preserve_lobound));
    check(result);
  }
  // lazy tiles as the operands of a binary op
  {
    TArrayI result;
    BOOST_REQUIRE_NO_THROW(result(annot) = lazy(annot) + lazy(annot));
    for (auto const& ord : *result.pmap()) {
      TArrayI::value_type const tile = result.find(ord);
      auto const expected = 2 * LazyTile::value(tile.range());
      for (std::size_t i = 0ul; i < tile.size(); ++i)
        BOOST_CHECK_EQUAL(tile[i], expected);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
