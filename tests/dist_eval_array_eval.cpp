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

using namespace TiledArray;
using TiledArray::detail::Noop;
using TiledArray::detail::Scal;
using TiledArray::detail::UnaryWrapper;

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

// With no process map imposed by a consumer, a block expression evaluates
// where the parent holds its tiles: the evaluator owns, on each rank, exactly
// the tiles whose parent tile that rank holds, so no tile is fetched from
// another rank. (A map a consumer imposes, such as SUMMA's, still wins.) The
// ownership follows the parent through a permutation of the result too.
BOOST_AUTO_TEST_CASE(block_evaluates_where_the_parent_holds_its_tiles) {
  using TiledArray::detail::BlockPmap;
  auto& world = array.world();
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;  // a proper sub-block, so block and parent tile indices differ
  BlockRange const block(tiles_range, lo, hi);

  auto label = [](unsigned int i) { return "a" + std::to_string(i); };
  // the annotation of the result whose mode d is the parent's mode
  // source_mode[d]
  auto annotation = [&](container::svector<unsigned int> const& source_mode) {
    std::string result;
    for (unsigned int d = 0; d < source_mode.size(); ++d)
      result += (d ? "," : "") + label(source_mode[d]);
    return result;
  };
  container::svector<unsigned int> identity(GlobalFixture::dim);
  for (unsigned int d = 0; d < identity.size(); ++d) identity[d] = d;
  auto const annot = annotation(identity);

  // the evaluator of block expression expr, for the result annotated by the
  // parent's modes source_mode, with no map imposed
  auto make_dist_eval = [&](auto const& expr, auto const& source_mode) {
    typename TiledArray::expressions::ExprTrait<
        std::decay_t<decltype(expr)>>::engine_type engine(expr);
    engine.init(
        world, {},
        TiledArray::expressions::BipartiteIndexList(annotation(source_mode)));
    return engine.make_dist_eval();
  };

  // the parent's tile index of the result's tile with ordinal ord; the
  // permutation is applied by hand here, independently of the engine's
  auto parent_index = [&](auto const& dist_eval, std::size_t ord,
                          auto const& source_mode) {
    auto const result_index = dist_eval.trange().tiles_range().idx(ord);
    auto index = result_index;
    for (std::size_t d = 0; d < index.size(); ++d)
      index[source_mode[d]] = result_index[d] + lo[source_mode[d]];
    return index;
  };

  auto check_follows_parent = [&](auto const& parent, auto const& expr,
                                  auto const& source_mode) {
    auto dist_eval = make_dist_eval(expr, source_mode);
    BOOST_CHECK(std::dynamic_pointer_cast<const BlockPmap>(dist_eval.pmap()));
    std::size_t local = 0;
    for (auto ord : *dist_eval.pmap()) {
      ++local;
      BOOST_CHECK(parent.is_local(parent_index(dist_eval, ord, source_mode)));
    }
    std::size_t parent_local_in_block = 0;
    for (auto const& ord : block)
      if (parent.is_local(ord)) ++parent_local_in_block;
    BOOST_CHECK_EQUAL(local, parent_local_in_block);
  };

  check_follows_parent(array, array(annot).block(lo, hi), identity);
  check_follows_parent(
      array,
      std::as_const(array)(annot).block(lo, hi, TiledArray::preserve_lobound),
      identity);
  check_follows_parent(array, 2 * array(annot).block(lo, hi), identity);

  // a permuted result: with modes 0 and 1 swapped the permutation is its own
  // inverse, so also cycle 3 modes, which tells a permutation from its inverse
  if constexpr (GlobalFixture::dim >= 2) {
    auto swapped = identity;
    std::swap(swapped[0], swapped[1]);
    check_follows_parent(array, array(annot).block(lo, hi), swapped);
  }
  if constexpr (GlobalFixture::dim >= 3) {
    auto cycled = identity;
    std::rotate(cycled.begin(), cycled.begin() + 1, cycled.begin() + 3);
    check_follows_parent(array, array(annot).block(lo, hi), cycled);
    check_follows_parent(array, 2 * array(annot).block(lo, hi), cycled);
  }

  // a sparse parent: zero tiles have owners too
  TSpArrayI sparse;
  {
    Tensor<float> norms(tiles_range, 0.0f);
    for (std::size_t i = 0; i < norms.size(); ++i)
      if (i % 3) norms[i] = 1.0f;
    sparse = TSpArrayI(world, tr, SparseShape<float>(norms, tr));
    sparse.fill(1);
  }
  check_follows_parent(sparse, sparse(annot).block(lo, hi), identity);

  // each tile of an evaluator not following the parent has exactly one owner
  auto check_one_owner_each = [&](auto const& dist_eval) {
    BOOST_CHECK(!std::dynamic_pointer_cast<const BlockPmap>(dist_eval.pmap()));
    BOOST_CHECK(!dist_eval.pmap()->is_replicated());
    std::size_t local = 0;
    for ([[maybe_unused]] auto ord : *dist_eval.pmap()) ++local;
    world.gop.sum(local);
    BOOST_CHECK_EQUAL(local, block.volume());
  };

  // a replicated parent: no rank is the owner of a tile, so the default map
  // over the block's tiles
  {
    TArrayI replicated(world, tr,
                       std::make_shared<TiledArray::detail::ReplicatedPmap>(
                           world, tiles_range.volume()));
    replicated.fill(1);
    check_one_owner_each(
        make_dist_eval(replicated(annot).block(lo, hi), identity));
    // each element of the block counted once
    std::size_t block_elements = 1;
    for (std::size_t d = 0; d < GlobalFixture::dim; ++d)
      block_elements *=
          tr.dim(d).tile(hi[d] - 1).second - tr.dim(d).tile(lo[d]).first;
    BOOST_CHECK_EQUAL(
        replicated(annot).block(lo, hi).squared_norm().get(),
        static_cast<decltype(replicated(annot).squared_norm().get())>(
            block_elements));
  }

  // a parent in another World: its map's ranks are not ranks of this world,
  // so the default map over the block's tiles
  {
    // the split is a blocking MPI collective: let pending work finish first
    world.gop.fence();
    World this_rank(madness::blocking_invoke(
        &SafeMPI::Intracomm::Split, world.mpi.comm(), world.rank(), 0));
    {
      TArrayI other(this_rank, tr);
      other.fill(1);
      check_one_owner_each(
          make_dist_eval(other(annot).block(lo, hi), identity));
    }
    this_rank.gop.fence();
  }
  world.gop.fence();
}

// An array assigned from a block expression gets the default map for its
// tiled range, not the map the block evaluated on (which follows the parent's),
// and holds the parent's tiles.
BOOST_AUTO_TEST_CASE(block_assignment_result_has_the_default_map) {
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;
  std::string annot;
  for (unsigned int d = 0; d < GlobalFixture::dim; ++d)
    annot += (d ? ",a" : "a") + std::to_string(d);

  auto check = [&](auto const& parent) {
    std::decay_t<decltype(parent)> result;
    result(annot) = parent(annot).block(lo, hi);
    BOOST_CHECK(
        std::dynamic_pointer_cast<const TiledArray::detail::BlockedPmap>(
            result.pmap()));
    for (auto ord : *result.pmap()) {
      auto index = result.trange().tiles_range().idx(ord);
      for (std::size_t d = 0; d < index.size(); ++d) index[d] += lo[d];
      BOOST_CHECK_EQUAL(result.is_zero(ord), parent.is_zero(index));
      if (result.is_zero(ord)) continue;
      auto const tile = result.find(ord).get();
      auto const parent_tile = parent.find(index).get();
      BOOST_CHECK(std::equal(tile.begin(), tile.end(), parent_tile.begin(),
                             parent_tile.end()));
    }
  };

  check(array);

  Tensor<float> norms(tiles_range, 0.0f);
  for (std::size_t i = 0; i < norms.size(); ++i)
    if (i % 3) norms[i] = 1.0f;
  TSpArrayI sparse(array.world(), tr, SparseShape<float>(norms, tr));
  sparse.fill(1);
  check(sparse);
  array.world().gop.fence();
}

BOOST_AUTO_TEST_SUITE_END()
