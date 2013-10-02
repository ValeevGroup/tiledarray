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

#include "TiledArray/dist_eval/array_eval.h"
#include "unit_test_config.h"
#include "array_fixture.h"
#include "TiledArray/tile_op/scal.h"
#include "TiledArray/shape.h"


using namespace TiledArray;

// Array evaluator fixture
struct ArrayEvalImplFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::dim> ArrayN;
  typedef math::Scal<ArrayN::value_type::eval_type,
      ArrayN::value_type::eval_type, false> op_type;
  typedef detail::ArrayEvalImpl<ArrayN, op_type, DensePolicy> impl_type;
  typedef detail::DistEval<detail::LazyArrayTile<typename ArrayN::value_type,
      op_type>, DensePolicy> dist_eval_type;


  ArrayEvalImplFixture() : op(3), array(*GlobalFixture::world, tr) {
    // Fill array with random data
    for(ArrayN::iterator it = array.begin(); it != array.end(); ++it) {
      ArrayN::value_type tile(array.trange().make_tile_range(it.index()));
      for(ArrayN::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 101;
      *it = tile;
    }
  }

  ~ArrayEvalImplFixture() { }

   op_type op;
   ArrayN array;
}; // ArrayEvalFixture

BOOST_FIXTURE_TEST_SUITE( array_eval_suite, ArrayEvalImplFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(impl_type(array, array.get_world(), DenseShape(), array.get_pmap(), Permutation(), op));

  dist_eval_type dist_eval = detail::make_array_eval(array, array.get_world(),
      DenseShape(), array.get_pmap(), Permutation(), op);

  BOOST_CHECK_EQUAL(& dist_eval.get_world(), GlobalFixture::world);
  BOOST_CHECK(dist_eval.pmap() == array.get_pmap());
  BOOST_CHECK_EQUAL(dist_eval.range(), tr.tiles());
  BOOST_CHECK_EQUAL(dist_eval.trange(), tr);
  BOOST_CHECK_EQUAL(dist_eval.size(), tr.tiles().volume());
  BOOST_CHECK(dist_eval.is_dense());
  for(std::size_t i = 0; i < tr.tiles().volume(); ++i)
    BOOST_CHECK(! dist_eval.is_zero(i));
}

BOOST_AUTO_TEST_CASE( eval_scale )
{
  dist_eval_type dist_eval = detail::make_array_eval(array, array.get_world(),
      DenseShape(), array.get_pmap(), Permutation(), op);
  BOOST_REQUIRE_NO_THROW(dist_eval.eval());

  dist_eval_type::pmap_interface::const_iterator it = dist_eval.pmap()->begin();
  const dist_eval_type::pmap_interface::const_iterator end = dist_eval.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {
    // Get the original type
    ArrayN::value_type array_tile = array.find(*it);

    // Get the array evaluator tile.
    madness::Future<dist_eval_type::value_type> impl_tile;
    BOOST_REQUIRE_NO_THROW(impl_tile = dist_eval.move(*it));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = impl_tile.get());

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), array_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[i], 3 * array_tile[i]);
    }
  }

}

BOOST_AUTO_TEST_CASE( eval_permute )
{
  // Create permutation to be applied in the array evaluations
  std::array<std::size_t, GlobalFixture::dim> p;
  for(std::size_t i = 0; i < p.size(); ++i)
    p[i] = (i + p.size() - 1) % p.size();
  const Permutation perm(p.begin(), p.end());

  // Redefine the types for the new operation.
  typedef math::Noop<ArrayN::value_type, ArrayN::value_type, false> op_type;
  typedef detail::DistEval<detail::LazyArrayTile<typename ArrayN::value_type,
      op_type>, DensePolicy> dist_eval_type;

  // Construct and evaluate
  dist_eval_type dist_eval = detail::make_array_eval(array, array.get_world(),
      DenseShape(), array.get_pmap(), perm, op_type(perm));
  BOOST_REQUIRE_NO_THROW(dist_eval.eval());

  // Check that each tile has been moved to the correct location and has been
  // properly permuted.
  dist_eval_type::pmap_interface::const_iterator it = dist_eval.pmap()->begin();
  const dist_eval_type::pmap_interface::const_iterator end = dist_eval.pmap()->end();
  const Permutation inv_perm = -perm;
  for(; it != end; ++it) {
    // Get the original type
    ArrayN::value_type array_tile = array.find(inv_perm ^ dist_eval.range().idx(*it));

    // Get the corresponding array evaluator tile.
    madness::Future<dist_eval_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = dist_eval.move(*it));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get(););

    // Check that the result tile is correctly modified.
    BOOST_CHECK_EQUAL(eval_tile.range(), perm ^ array_tile.range());
    for(std::size_t i = 0ul; i < eval_tile.size(); ++i) {
      BOOST_CHECK_EQUAL(eval_tile[perm ^ array_tile.range().idx(i)], array_tile[i]);
    }
  }

}

BOOST_AUTO_TEST_SUITE_END()
