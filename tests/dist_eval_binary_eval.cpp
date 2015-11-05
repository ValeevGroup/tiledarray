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

struct BinaryEvalFixture : public TiledRangeFixture {
  typedef TArrayI ArrayN;
  typedef math::Noop<ArrayN::value_type,
      ArrayN::value_type, true> array_op_type;
  typedef detail::DistEval<detail::LazyArrayTile<ArrayN::value_type, array_op_type>,
      DensePolicy> array_eval_type;
  typedef math::Add<ArrayN::value_type, ArrayN::value_type, ArrayN::value_type,
      false, false> op_type;
  typedef detail::BinaryEvalImpl<array_eval_type, array_eval_type, op_type, DensePolicy> impl_type;

  BinaryEvalFixture() :
    left(*GlobalFixture::world, tr),
    right(*GlobalFixture::world, tr),
    left_arg(make_array_eval(left, left.get_world(), DenseShape(),
        left.get_pmap(), Permutation(), array_op_type())),
    right_arg(make_array_eval(right, right.get_world(), DenseShape(),
        left.get_pmap(), Permutation(), array_op_type()))
  {
    // Fill array with random data
    for(ArrayN::iterator it = left.begin(); it != left.end(); ++it) {
      ArrayN::value_type tile(left.trange().make_tile_range(it.index()));
      for(ArrayN::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
    for(ArrayN::iterator it = right.begin(); it != right.end(); ++it) {
      ArrayN::value_type tile(right.trange().make_tile_range(it.index()));
      for(ArrayN::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
  }

  ~BinaryEvalFixture() { }

  template <typename Tile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<typename DistArray<Tile, Policy>::value_type, Op>, Policy>
  make_array_eval(
      const DistArray<Tile, Policy>& array,
      TiledArray::World& world,
      const typename TiledArray::detail::DistEval<Tile, Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<Tile, Policy>::pmap_interface>& pmap,
      const Permutation& perm,
      const Op& op)
  {
    typedef TiledArray::detail::ArrayEvalImpl<DistArray<Tile, Policy>, Op, Policy> impl_type;
    return TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<typename DistArray<Tile, Policy>::value_type, Op>, Policy>(
        std::shared_ptr<impl_type>(new impl_type(array, world,
        (perm ? perm * array.trange() : array.trange()), shape, pmap, perm, op)));
  }

  template <typename LeftTile, typename RightTile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<typename Op::result_type, Policy> make_binary_eval(
      const TiledArray::detail::DistEval<LeftTile, Policy>& left,
      const TiledArray::detail::DistEval<RightTile, Policy>& right,
      TiledArray::World& world,
      const typename TiledArray::detail::DistEval<typename Op::result_type, Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<typename Op::result_type, Policy>::pmap_interface>& pmap,
      const Permutation& perm,
      const Op& op)
  {
    typedef TiledArray::detail::BinaryEvalImpl<
        TiledArray::detail::DistEval<LeftTile, Policy>,
        TiledArray::detail::DistEval<RightTile, Policy>, Op, Policy> impl_type;
    return TiledArray::detail::DistEval<typename Op::result_type, Policy>(
        std::shared_ptr<impl_type>(new impl_type(left, right, world,
        (perm ? perm * left.trange() : left.trange()), shape, pmap, perm, op)));
  }

   ArrayN left;
   ArrayN right;
   array_eval_type left_arg;
   array_eval_type right_arg;
}; // Fixture

BOOST_FIXTURE_TEST_SUITE( dist_eval_binary_eval_suite, BinaryEvalFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(impl_type(left_arg, right_arg, left.get_world(),
      left_arg.trange(), DenseShape(), left_arg.pmap(), Permutation(), op_type()));

  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  dist_eval_type1 binary = make_binary_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), left_arg.pmap(), Permutation(), op_type());

  BOOST_CHECK_EQUAL(& binary.get_world(), GlobalFixture::world);
  BOOST_CHECK(binary.pmap() == left_arg.pmap());
  BOOST_CHECK_EQUAL(binary.range(), tr.tiles());
  BOOST_CHECK_EQUAL(binary.trange(), tr);
  BOOST_CHECK_EQUAL(binary.size(), tr.tiles().volume());
  BOOST_CHECK(binary.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! binary.is_zero(i));
}

BOOST_AUTO_TEST_CASE( eval )
{
  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  dist_eval_type1 dist_eval = make_binary_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), left_arg.pmap(), Permutation(), op_type());

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval.wait());

  dist_eval_type1::pmap_interface::const_iterator it = dist_eval.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = dist_eval.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {
    // Get the original tiles
    const ArrayN::value_type left_tile = left.find(*it);
    const ArrayN::value_type right_tile = right.find(*it);

    // Get the array evaluator tile.
    Future<dist_eval_type1::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(*it));

    // Force the evaluation of the tile
    dist_eval_type1::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), dist_eval.trange().make_tile_range(*it));
    BOOST_CHECK_EQUAL(eval_tile.range(), left_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], left_tile[i] + right_tile[i]);
    }
  }

}

BOOST_AUTO_TEST_CASE( perm_eval )
{
  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  // Create permutation to be applied in the array evaluations
  std::array<std::size_t, GlobalFixture::dim> p;
  for(std::size_t i = 0; i < p.size(); ++i)
    p[i] = (i + p.size() - 1) % p.size();
  const Permutation perm(p.begin(), p.end());


  dist_eval_type1 dist_eval = make_binary_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), left_arg.pmap(), perm, op_type(perm));

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval.wait());

  dist_eval_type1::pmap_interface::const_iterator it = dist_eval.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = dist_eval.pmap()->end();

  // Check that each tile has been properly scaled.
  const Permutation inv_perm = -perm;
  for(; it != end; ++it) {
    // Get the original tiles
    const std::size_t arg_index = left.range().ordinal(inv_perm * dist_eval.range().idx(*it));
    const ArrayN::value_type left_tile = left.find(arg_index);
    const ArrayN::value_type right_tile = right.find(arg_index);

    // Get the array evaluator tile.
    Future<dist_eval_type1::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(*it));

    // Force the evaluation of the tile
    dist_eval_type1::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), dist_eval.trange().make_tile_range(*it));
    BOOST_CHECK_EQUAL(eval_tile.range(), perm * left_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[perm * left_tile.range().idx(i)], left_tile[i] + right_tile[i]);
    }
  }

}

BOOST_AUTO_TEST_SUITE_END()
