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
 *  dist_eval_binary_eval.cpp
 *  Oct 1, 2013
 *
 */

#include <array_fixture.h>

#include "TiledArray/dist_eval/binary_eval.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using TiledArray::detail::Add;
using TiledArray::detail::BinaryWrapper;
using TiledArray::detail::Noop;
using TiledArray::detail::UnaryWrapper;

struct BinaryEvalFixture : public TiledRangeFixture {
  BinaryEvalFixture()
      : left(*GlobalFixture::world, tr), right(*GlobalFixture::world, tr) {
    // Fill array with random data
    for (TArrayI::iterator it = left.begin(); it != left.end(); ++it) {
      TArrayI::value_type tile(left.trange().make_tile_range(it.index()));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
    for (TArrayI::iterator it = right.begin(); it != right.end(); ++it) {
      TArrayI::value_type tile(right.trange().make_tile_range(it.index()));
      for (TArrayI::value_type::iterator tile_it = tile.begin();
           tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
  }

  ~BinaryEvalFixture() {}

  static UnaryWrapper<Noop<TensorI, TensorI, true> > make_array_noop(
      const Permutation& perm = Permutation()) {
    return UnaryWrapper<Noop<TensorI, TensorI, true> >(
        Noop<TensorI, TensorI, true>(), perm);
  }

  static BinaryWrapper<
      TiledArray::detail::Add<TArrayI::value_type, TArrayI::value_type,
                              TArrayI::value_type, false, false> >
  make_add(const Permutation& perm = Permutation()) {
    return BinaryWrapper<
        TiledArray::detail::Add<TArrayI::value_type, TArrayI::value_type,
                                TArrayI::value_type, false, false> >(
        TiledArray::detail::Add<TArrayI::value_type, TArrayI::value_type,
                                TArrayI::value_type, false, false>(),
        perm);
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
      const std::shared_ptr<
          typename TiledArray::detail::DistEval<Tile, Policy>::pmap_interface>&
          pmap,
      const Permutation& perm, const Op& op) {
    typedef TiledArray::detail::ArrayEvalImpl<DistArray<Tile, Policy>, Op,
                                              Policy>
        impl_type;
    return TiledArray::detail::DistEval<
        TiledArray::detail::LazyArrayTile<
            typename DistArray<Tile, Policy>::value_type, Op>,
        Policy>(std::shared_ptr<impl_type>(new impl_type(
        array, world, (perm ? perm * array.trange() : array.trange()), shape,
        pmap, perm, op)));
  }

  template <typename LeftTile, typename RightTile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<typename Op::result_type, Policy>
  make_binary_eval(const TiledArray::detail::DistEval<LeftTile, Policy>& left,
                   const TiledArray::detail::DistEval<RightTile, Policy>& right,
                   TiledArray::World& world,
                   const typename TiledArray::detail::DistEval<
                       typename Op::result_type, Policy>::shape_type& shape,
                   const std::shared_ptr<typename TiledArray::detail::DistEval<
                       typename Op::result_type, Policy>::pmap_interface>& pmap,
                   const Permutation& perm, const Op& op) {
    typedef TiledArray::detail::BinaryEvalImpl<
        TiledArray::detail::DistEval<LeftTile, Policy>,
        TiledArray::detail::DistEval<RightTile, Policy>, Op, Policy>
        impl_type;
    return TiledArray::detail::DistEval<typename Op::result_type, Policy>(
        std::shared_ptr<impl_type>(new impl_type(
            left, right, world, (perm ? perm * left.trange() : left.trange()),
            shape, pmap, perm, op)));
  }

  TArrayI left;
  TArrayI right;
};  // Fixture

BOOST_FIXTURE_TEST_SUITE(dist_eval_binary_eval_suite, BinaryEvalFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  auto left_arg = make_array_eval(left, left.world(), DenseShape(), left.pmap(),
                                  Permutation(), make_array_noop());
  auto right_arg =
      make_array_eval(right, right.world(), DenseShape(), left.pmap(),
                      Permutation(), make_array_noop());

  BOOST_REQUIRE_NO_THROW(make_binary_eval(left_arg, right_arg, left.world(),
                                          DenseShape(), left_arg.pmap(),
                                          Permutation(), make_add()));

  auto binary =
      make_binary_eval(left_arg, right_arg, left_arg.world(), DenseShape(),
                       left_arg.pmap(), Permutation(), make_add());

  BOOST_CHECK_EQUAL(&binary.world(), GlobalFixture::world);
  BOOST_CHECK(binary.pmap() == left_arg.pmap());
  BOOST_CHECK_EQUAL(binary.range(), tr.tiles_range());
  BOOST_CHECK_EQUAL(binary.trange(), tr);
  BOOST_CHECK_EQUAL(binary.size(), tr.tiles_range().volume());
  BOOST_CHECK(binary.is_dense());
  for (std::size_t i = 0; i < tr.tiles_range().volume(); ++i)
    BOOST_CHECK(!binary.is_zero(i));
}

BOOST_AUTO_TEST_CASE(eval) {
  auto left_arg = make_array_eval(left, left.world(), DenseShape(), left.pmap(),
                                  Permutation(), make_array_noop());
  auto right_arg =
      make_array_eval(right, right.world(), DenseShape(), left.pmap(),
                      Permutation(), make_array_noop());

  auto dist_eval =
      make_binary_eval(left_arg, right_arg, left_arg.world(), DenseShape(),
                       left_arg.pmap(), Permutation(), make_add());
  using dist_eval_type = decltype(dist_eval);

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval.wait());

  // Check that each tile has been properly scaled.
  for (auto index : *dist_eval.pmap()) {
    // Get the original tiles
    const TArrayI::value_type left_tile = left.find(index);
    const TArrayI::value_type right_tile = right.find(index);

    // Get the array evaluator tile.
    Future<dist_eval_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(index));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(),
                      dist_eval.trange().make_tile_range(index));
    BOOST_CHECK_EQUAL(eval_tile.range(), left_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], left_tile[i] + right_tile[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(perm_eval) {
  auto left_arg = make_array_eval(left, left.world(), DenseShape(), left.pmap(),
                                  Permutation(), make_array_noop());
  auto right_arg =
      make_array_eval(right, right.world(), DenseShape(), left.pmap(),
                      Permutation(), make_array_noop());

  // Create permutation to be applied in the array evaluations
  std::array<std::size_t, GlobalFixture::dim> p;
  for (std::size_t i = 0; i < p.size(); ++i)
    p[i] = (i + p.size() - 1) % p.size();
  const Permutation perm(p.begin(), p.end());

  auto dist_eval =
      make_binary_eval(left_arg, right_arg, left_arg.world(), DenseShape(),
                       left_arg.pmap(), perm, make_add(perm));

  using dist_eval_type = decltype(dist_eval);

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval.wait());

  // Check that each tile has been properly scaled.
  const Permutation inv_perm = -perm;
  for (auto index : *dist_eval.pmap()) {
    // Get the original tiles
    const std::size_t arg_index =
        left.tiles_range().ordinal(inv_perm * dist_eval.range().idx(index));
    const TArrayI::value_type left_tile = left.find(arg_index);
    const TArrayI::value_type right_tile = right.find(arg_index);

    // Get the array evaluator tile.
    Future<dist_eval_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(index));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(),
                      dist_eval.trange().make_tile_range(index));
    BOOST_CHECK_EQUAL(eval_tile.range(), perm * left_tile.range());
    for (std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[perm * left_tile.range().idx(i)],
                        left_tile[i] + right_tile[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
