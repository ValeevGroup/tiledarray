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
  auto const& tiles_range = tr.tiles_range();
  container::svector<long> lo(tiles_range.lobound().begin(),
                              tiles_range.lobound().end());
  container::svector<long> hi(tiles_range.upbound().begin(),
                              tiles_range.upbound().end());
  lo[0] += 1;  // a proper sub-block, so block and parent tile indices differ
  // annotations "a0,a1,..." and the same with the first two labels swapped
  std::string annot, swapped_annot;
  for (unsigned int d = 0; d < GlobalFixture::dim; ++d) {
    auto label = [](unsigned int i) { return "a" + std::to_string(i); };
    unsigned int const sd = (d == 0 ? 1 : (d == 1 ? 0 : d));
    annot += (d ? "," : "") + label(d);
    swapped_annot += (d ? "," : "") + label(sd);
  }
  // the parent's tiles inside the block that this rank holds
  std::size_t parent_local_in_block = 0;
  {
    BlockRange const blk(tiles_range, lo, hi);
    for (auto const& ord : blk)
      if (array.is_local(ord)) ++parent_local_in_block;
  }

  auto check_follows_parent = [&](auto const& expr, auto const& target_annot,
                                  Permutation const& result_to_source) {
    typename TiledArray::expressions::ExprTrait<
        std::decay_t<decltype(expr)>>::engine_type engine(expr);
    engine.init(array.world(), {},
                TiledArray::expressions::BipartiteIndexList(target_annot));
    auto dist_eval = engine.make_dist_eval();
    std::size_t local = 0;
    for (auto ord : *dist_eval.pmap()) {
      ++local;
      auto block_index = dist_eval.trange().tiles_range().idx(ord);
      if (result_to_source) block_index = result_to_source * block_index;
      auto parent_index = block_index;
      for (std::size_t d = 0; d < parent_index.size(); ++d)
        parent_index[d] += lo[d];
      BOOST_CHECK(array.is_local(parent_index));
    }
    BOOST_CHECK_EQUAL(local, parent_local_in_block);
  };

  // the block as a consumer with no map of its own sees it
  check_follows_parent(array(annot).block(lo, hi), annot, Permutation());
  check_follows_parent(
      std::as_const(array)(annot).block(lo, hi, TiledArray::preserve_lobound),
      annot, Permutation());

  // a permuted result (the first two modes swapped): the result's tile
  // (i1, i0, ...) is the source's (i0, i1, ...), and must live where the
  // parent holds the source tile
  if constexpr (GlobalFixture::dim >= 2) {
    container::svector<unsigned int> p(GlobalFixture::dim);
    for (std::size_t d = 0; d < p.size(); ++d) p[d] = d;
    std::swap(p[0], p[1]);
    Permutation const swap01(p.begin(), p.end());
    check_follows_parent(array(annot).block(lo, hi), swapped_annot,
                         swap01.inv());
  }
}

BOOST_AUTO_TEST_SUITE_END()
