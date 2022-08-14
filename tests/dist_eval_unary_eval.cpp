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
 *  unary_eval.cpp
 *  August 8, 2013
 *
 */

#include <array_fixture.h>

#include "TiledArray/dist_eval/unary_eval.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;
using TiledArray::detail::Noop;
using TiledArray::detail::Scal;
using TiledArray::detail::UnaryWrapper;

// Array evaluator fixture
struct UnaryEvalImplFixture : public TiledRangeFixture {
  typedef Noop<TArrayI::value_type, TArrayI::value_type, true>
      array_op_base_type;
  typedef UnaryWrapper<array_op_base_type> array_op_type;
  typedef TiledArray::detail::DistEval<
      TiledArray::detail::LazyArrayTile<TArrayI::value_type, array_op_type>,
      DensePolicy>
      dist_eval_type;

  UnaryEvalImplFixture()
      : array(*GlobalFixture::world, tr),
        arg(make_array_eval(array, array.world(), DenseShape(), array.pmap(),
                            Permutation(),
                            array_op_type(array_op_base_type()))) {
    // Fill array with random data
    for (TArrayI::iterator it = array.begin(); it != array.end(); ++it) {
      TensorI tile(array.trange().make_tile_range(it.index()));
      for (TensorI::iterator tile_it = tile.begin(); tile_it != tile.end();
           ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
  }

  ~UnaryEvalImplFixture() {}

  static UnaryWrapper<Noop<TensorI, TensorI, true> > make_array_noop(
      const Permutation& perm = Permutation()) {
    return UnaryWrapper<Noop<TensorI, TensorI, true> >(
        Noop<TensorI, TensorI, true>(), perm);
  }

  static UnaryWrapper<Scal<TensorI, TensorI, int, true> > make_scal1(
      const int factor, const Permutation& perm = Permutation()) {
    return UnaryWrapper<Scal<TensorI, TensorI, int, true> >(
        Scal<TensorI, TensorI, int, true>(factor), perm);
  }

  static UnaryWrapper<Scal<TensorI, TensorI, int, false> > make_scal0(
      const int factor, const Permutation& perm = Permutation()) {
    return UnaryWrapper<Scal<TensorI, TensorI, int, false> >(
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

  template <typename Tile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<typename Op::result_type, Policy>
  make_unary_eval(
      const TiledArray::detail::DistEval<Tile, Policy>& arg,
      TiledArray::World& world,
      const typename TiledArray::detail::DistEval<Tile, Policy>::shape_type&
          shape,
      const std::shared_ptr<const typename TiledArray::detail::DistEval<
          Tile, Policy>::pmap_interface>& pmap,
      const Permutation& perm, const Op& op) {
    typedef TiledArray::detail::UnaryEvalImpl<
        TiledArray::detail::DistEval<Tile, Policy>, Op, Policy>
        impl_type;
    return TiledArray::detail::DistEval<typename Op::result_type, Policy>(
        std::shared_ptr<impl_type>(new impl_type(
            arg, world, (perm ? perm * arg.trange() : arg.trange()), shape,
            pmap, perm, op)));
  }

  TArrayI array;
  dist_eval_type arg;
};  // ArrayEvalFixture

BOOST_FIXTURE_TEST_SUITE(unary_eval_suite, UnaryEvalImplFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  BOOST_REQUIRE_NO_THROW(make_unary_eval(arg, arg.world(), DenseShape(),
                                         arg.pmap(), Permutation(),
                                         make_scal0(3)));

  auto unary = make_unary_eval(arg, arg.world(), DenseShape(), arg.pmap(),
                               Permutation(), make_scal0(3));

  BOOST_CHECK_EQUAL(&unary.world(), GlobalFixture::world);
  BOOST_CHECK(unary.pmap() == arg.pmap());
  BOOST_CHECK_EQUAL(unary.range(), tr.tiles_range());
  BOOST_CHECK_EQUAL(unary.trange(), tr);
  BOOST_CHECK_EQUAL(unary.size(), tr.tiles_range().volume());
  BOOST_CHECK(unary.is_dense());
  for (std::size_t i = 0; i < tr.tiles_range().volume(); ++i)
    BOOST_CHECK(!unary.is_zero(i));

  BOOST_REQUIRE_NO_THROW(make_unary_eval(unary, unary.world(), DenseShape(),
                                         arg.pmap(), Permutation(),
                                         make_scal0(5)));

  auto unary2 = make_unary_eval(unary, unary.world(), DenseShape(),
                                unary.pmap(), Permutation(), make_scal0(5));

  BOOST_CHECK_EQUAL(&unary2.world(), GlobalFixture::world);
  BOOST_CHECK(unary2.pmap() == arg.pmap());
  BOOST_CHECK_EQUAL(unary2.range(), tr.tiles_range());
  BOOST_CHECK_EQUAL(unary2.trange(), tr);
  BOOST_CHECK_EQUAL(unary2.size(), tr.tiles_range().volume());
  BOOST_CHECK(unary2.is_dense());
  for (std::size_t i = 0; i < tr.tiles_range().volume(); ++i)
    BOOST_CHECK(!unary2.is_zero(i));
}

BOOST_AUTO_TEST_CASE(eval) {
  auto dist_eval = make_unary_eval(arg, arg.world(), DenseShape(), arg.pmap(),
                                   Permutation(), make_scal0(3));

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval.wait());

  // Check that each tile has been properly scaled.
  for (auto index : *dist_eval.pmap()) {
    // Get the original type
    TensorI array_tile = array.find(index);

    // Get the array evaluator tile.
    Future<TensorI> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(index));

    // Force the evaluation of the tile
    TensorI eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(),
                      dist_eval.trange().make_tile_range(index));
    BOOST_CHECK_EQUAL(eval_tile.range(), array_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], 3 * array_tile[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(double_eval) {
  /// Construct a scaling unary evaluator
  auto dist_eval = make_unary_eval(arg, arg.world(), DenseShape(), arg.pmap(),
                                   Permutation(), make_scal0(3));

  /// Construct a two-step, scaling unary evaluator
  auto dist_eval2 =
      make_unary_eval(dist_eval, dist_eval.world(), DenseShape(),
                      dist_eval.pmap(), Permutation(), make_scal1(5));

  BOOST_REQUIRE_NO_THROW(dist_eval2.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval2.wait());

  // Check that each tile has been properly scaled.
  for (auto index : *dist_eval2.pmap()) {
    // Get the original type
    TensorI array_tile = array.find(index);

    // Get the array evaluator tile.
    Future<TensorI> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval2.get(index));

    // Wait the evaluation of the tile
    TensorI eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(),
                      dist_eval2.trange().make_tile_range(index));
    BOOST_CHECK_EQUAL(eval_tile.range(), array_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], 5 * 3 * array_tile[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(perm_eval) {
  // Create permutation to be applied in the array evaluations
  std::array<std::size_t, GlobalFixture::dim> p;
  for (std::size_t i = 0; i < p.size(); ++i)
    p[i] = (i + p.size() - 1) % p.size();
  const Permutation perm(p.begin(), p.end());

  /// Construct a scaling unary evaluator
  auto dist_eval = make_unary_eval(arg, arg.world(), DenseShape(), arg.pmap(),
                                   Permutation(), make_scal0(3));

  /// Construct a two-step, scaling unary evaluator
  auto dist_eval2 =
      make_unary_eval(dist_eval, dist_eval.world(), DenseShape(),
                      dist_eval.pmap(), perm, make_scal1(5, perm));

  BOOST_REQUIRE_NO_THROW(dist_eval2.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval2.wait());

  // Check that each tile has been properly scaled and permuted.
  const Permutation inv_perm = -perm;
  for (auto index : *dist_eval2.pmap()) {
    // Get the original type
    const TensorI array_tile =
        array.find(inv_perm * dist_eval2.range().idx(index));

    // Get the array evaluator tile.
    Future<TensorI> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval2.get(index));

    // Force the evaluation of the tile
    TensorI eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(),
                      dist_eval2.trange().make_tile_range(index));
    BOOST_CHECK_EQUAL(eval_tile.range(), perm * array_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[perm * array_tile.range().idx(i)],
                        5 * 3 * array_tile[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
