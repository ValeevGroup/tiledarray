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

#include "TiledArray/dist_eval/unary_eval.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

// Array evaluator fixture
struct UnaryEvalImplFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::dim> ArrayN;
  typedef math::Noop<ArrayN::value_type, ArrayN::value_type, true> array_op_type;
  typedef TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<ArrayN::value_type,
      array_op_type>, DensePolicy> dist_eval_type;
  typedef math::Scal<ArrayN::value_type, ArrayN::value_type, false> op_type;
  typedef TiledArray::detail::UnaryEvalImpl<dist_eval_type, op_type, DensePolicy> impl_type;


  UnaryEvalImplFixture() :
    array(*GlobalFixture::world, tr),
    arg(make_array_eval(array, array.get_world(), DenseShape(),
        array.get_pmap(), Permutation(), array_op_type()))
  {
    // Fill array with random data
    for(ArrayN::iterator it = array.begin(); it != array.end(); ++it) {
      ArrayN::value_type tile(array.trange().make_tile_range(it.index()));
      for(ArrayN::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
  }

  ~UnaryEvalImplFixture() { }

  template <typename T, unsigned int DIM, typename Tile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<typename Array<T, DIM, Tile, Policy>::value_type, Op>, Policy>
  make_array_eval(
      const Array<T, DIM, Tile, Policy>& array,
      madness::World& world,
      const typename TiledArray::detail::DistEval<Tile, Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<Tile, Policy>::pmap_interface>& pmap,
      const Permutation& perm,
      const Op& op)
  {
    typedef TiledArray::detail::ArrayEvalImpl<Array<T, DIM, Tile, Policy>, Op, Policy> impl_type;
    return TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<typename TiledArray::Array<T, DIM, Tile, Policy>::value_type, Op>, Policy>(
        std::shared_ptr<impl_type>( new impl_type(array, world,
        (perm ? perm ^ array.trange() : array.trange()), shape, pmap, perm, op)));
  }

  template <typename Tile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<typename Op::result_type, Policy> make_unary_eval(
      const TiledArray::detail::DistEval<Tile, Policy>& arg,
      madness::World& world,
      const typename TiledArray::detail::DistEval<Tile, Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<Tile, Policy>::pmap_interface>& pmap,
      const Permutation& perm,
      const Op& op)
  {
    typedef TiledArray::detail::UnaryEvalImpl<
        TiledArray::detail::DistEval<Tile, Policy>, Op, Policy> impl_type;
    return TiledArray::detail::DistEval<typename Op::result_type, Policy>(
        std::shared_ptr<impl_type>(new impl_type(arg, world,
        (perm ? perm ^ arg.trange() : arg.trange()), shape, pmap, perm, op)));
  }

   ArrayN array;
   dist_eval_type arg;
}; // ArrayEvalFixture

BOOST_FIXTURE_TEST_SUITE( unary_eval_suite, UnaryEvalImplFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(impl_type(arg, arg.get_world(), arg.trange(),
      DenseShape(), arg.pmap(), Permutation(), op_type(3)));

  typedef TiledArray::detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  dist_eval_type1 unary = make_unary_eval(arg, arg.get_world(),
      DenseShape(), arg.pmap(), Permutation(), op_type(3));

  BOOST_CHECK_EQUAL(& unary.get_world(), GlobalFixture::world);
  BOOST_CHECK(unary.pmap() == arg.pmap());
  BOOST_CHECK_EQUAL(unary.range(), tr.tiles());
  BOOST_CHECK_EQUAL(unary.trange(), tr);
  BOOST_CHECK_EQUAL(unary.size(), tr.tiles().volume());
  BOOST_CHECK(unary.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! unary.is_zero(i));

  typedef math::Scal<ArrayN::value_type, ArrayN::value_type, true> op_type2;
  typedef TiledArray::detail::DistEval<op_type2::result_type, DensePolicy> dist_eval_type2;
  typedef TiledArray::detail::UnaryEvalImpl<dist_eval_type2, op_type2, DensePolicy> impl_type2;


  BOOST_REQUIRE_NO_THROW(impl_type2(unary, unary.get_world(), unary.trange(),
      DenseShape(), arg.pmap(), Permutation(), op_type2(5)));

  dist_eval_type2 unary2 = make_unary_eval(unary, unary.get_world(),
      DenseShape(), unary.pmap(), Permutation(), op_type2(5));


  BOOST_CHECK_EQUAL(& unary2.get_world(), GlobalFixture::world);
  BOOST_CHECK(unary2.pmap() == arg.pmap());
  BOOST_CHECK_EQUAL(unary2.range(), tr.tiles());
  BOOST_CHECK_EQUAL(unary2.trange(), tr);
  BOOST_CHECK_EQUAL(unary2.size(), tr.tiles().volume());
  BOOST_CHECK(unary2.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! unary2.is_zero(i));
}


BOOST_AUTO_TEST_CASE( eval )
{
  typedef TiledArray::detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  dist_eval_type1 dist_eval = make_unary_eval(arg, arg.get_world(),
      DenseShape(), arg.pmap(), Permutation(), op_type(3));

  BOOST_REQUIRE_NO_THROW(dist_eval.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval.wait());

  dist_eval_type1::pmap_interface::const_iterator it = dist_eval.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = dist_eval.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {
    // Get the original type
    ArrayN::value_type array_tile = array.find(*it);

    // Get the array evaluator tile.
    madness::Future<dist_eval_type1::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.get(*it));

    // Force the evaluation of the tile
    dist_eval_type1::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), dist_eval.trange().make_tile_range(*it));
    BOOST_CHECK_EQUAL(eval_tile.range(), array_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], 3 * array_tile[i]);
    }

  }
}

BOOST_AUTO_TEST_CASE( double_eval )
{
  typedef TiledArray::detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;
  typedef math::Scal<ArrayN::value_type, ArrayN::value_type, true> op_type2;
  typedef TiledArray::detail::DistEval<op_type2::result_type, DensePolicy> dist_eval_type2;


  /// Construct a scaling unary evaluator
  dist_eval_type1 dist_eval = make_unary_eval(arg, arg.get_world(),
      DenseShape(), arg.pmap(), Permutation(), op_type(3));

  /// Construct a two-step, scaling unary evaluator
  dist_eval_type2 dist_eval2 = make_unary_eval(dist_eval,
      dist_eval.get_world(), DenseShape(), dist_eval.pmap(), Permutation(), op_type2(5));

  BOOST_REQUIRE_NO_THROW(dist_eval2.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval2.wait());

  dist_eval_type2::pmap_interface::const_iterator it = dist_eval2.pmap()->begin();
  const impl_type::pmap_interface::const_iterator end = dist_eval2.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {
    // Get the original type
    ArrayN::value_type array_tile = array.find(*it);

    // Get the array evaluator tile.
    madness::Future<dist_eval_type2::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval2.get(*it));

    // Wait the evaluation of the tile
    dist_eval_type2::value_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), dist_eval2.trange().make_tile_range(*it));
    BOOST_CHECK_EQUAL(eval_tile.range(), array_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], 5 * 3 * array_tile[i]);
    }
  }

}

BOOST_AUTO_TEST_CASE( perm_eval )
{
  typedef TiledArray::detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;
  typedef math::Scal<ArrayN::value_type, ArrayN::value_type, true> op_type2;
  typedef TiledArray::detail::DistEval<op_type2::result_type, DensePolicy> dist_eval_type2;

  // Create permutation to be applied in the array evaluations
  std::array<std::size_t, GlobalFixture::dim> p;
  for(std::size_t i = 0; i < p.size(); ++i)
    p[i] = (i + p.size() - 1) % p.size();
  const Permutation perm(p.begin(), p.end());

  /// Construct a scaling unary evaluator
  dist_eval_type1 dist_eval = make_unary_eval(arg, arg.get_world(),
      DenseShape(), arg.pmap(), Permutation(), op_type(3));

  /// Construct a two-step, scaling unary evaluator
  dist_eval_type2 dist_eval2 = make_unary_eval(dist_eval,
      dist_eval.get_world(), DenseShape(), dist_eval.pmap(), perm, op_type2(perm, 5));

  BOOST_REQUIRE_NO_THROW(dist_eval2.eval());
  BOOST_REQUIRE_NO_THROW(dist_eval2.wait());

  // Check that each tile has been properly scaled and permuted.
  impl_type::pmap_interface::const_iterator it = dist_eval2.pmap()->begin();
  const impl_type::pmap_interface::const_iterator end = dist_eval2.pmap()->end();
  const Permutation inv_perm = -perm;
  for(; it != end; ++it) {
    // Get the original type
    const ArrayN::value_type array_tile =
        array.find(inv_perm ^ dist_eval2.range().idx(*it));

    // Get the array evaluator tile.
    madness::Future<impl_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval2.get(*it));

    // Force the evaluation of the tile
    impl_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), dist_eval2.trange().make_tile_range(*it));
    BOOST_CHECK_EQUAL(eval_tile.range(), perm ^ array_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[perm ^ array_tile.range().idx(i)], 5 * 3 * array_tile[i]);
    }
  }

}

BOOST_AUTO_TEST_SUITE_END()
