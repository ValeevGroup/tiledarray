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

  static UnaryWrapper<Noop<TensorI, TensorI, true> > make_array_noop(
      const Permutation& perm = Permutation()) {
    return UnaryWrapper<Noop<TensorI, TensorI, true> >(
        Noop<TensorI, TensorI, true>(), perm);
  }

  static UnaryWrapper<Scal<TensorI, TensorI, int, true> > make_array_scal(
      const int factor, const Permutation& perm = Permutation()) {
    return UnaryWrapper<Scal<TensorI, TensorI, int, true> >(
        Scal<TensorI, TensorI, int, true>(factor), perm);
  }

  static UnaryWrapper<Scal<TensorI, TensorI, int, false> > make_scal(
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

BOOST_AUTO_TEST_SUITE_END()
