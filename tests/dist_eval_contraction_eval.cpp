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
 *  dist_eval_contraction_eval.h
 *  Oct 8, 2013
 *
 */

#include "TiledArray/dist_eval/contraction_eval.h"
#include "unit_test_config.h"
#include "TiledArray/array.h"
#include "TiledArray/dist_eval/array_eval.h"
#include "TiledArray/dense_shape.h"
#include "TiledArray/tile_op/contract_reduce.h"
#include "TiledArray/eigen.h"
#include "array_fixture.h"

using namespace TiledArray;

struct ContractionEvalFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::dim> ArrayN;
  typedef math::Noop<ArrayN::value_type::eval_type,
      ArrayN::value_type::eval_type, true> array_op_type;
  typedef detail::DistEval<detail::LazyArrayTile<ArrayN::value_type, array_op_type>,
      DensePolicy> array_eval_type;
  typedef math::ContractReduce<ArrayN::value_type, ArrayN::value_type, ArrayN::value_type> op_type;
  typedef detail::ContractionEvalImpl<array_eval_type, array_eval_type, op_type, DensePolicy> impl_type;

  ContractionEvalFixture() :
    left(*GlobalFixture::world, tr),
    right(*GlobalFixture::world, tr),
    left_arg(detail::make_array_eval(left, left.get_world(), DenseShape(),
        left.get_pmap(), Permutation(), array_op_type())),
    right_arg(detail::make_array_eval(right, right.get_world(), DenseShape(),
        left.get_pmap(), Permutation(), array_op_type())),
    result_tr(),
    op(madness::cblas::NoTrans, madness::cblas::NoTrans, 1, 2u, tr.tiles().dim(), tr.tiles().dim())
  {
    // Fill array with random data
    for(ArrayN::iterator it = left.begin(); it != left.end(); ++it) {
      ArrayN::value_type tile(left.trange().make_tile_range(it.index()));
      for(ArrayN::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 27;
      *it = tile;
    }
    for(ArrayN::iterator it = right.begin(); it != right.end(); ++it) {
      ArrayN::value_type tile(right.trange().make_tile_range(it.index()));
      for(ArrayN::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 27;
      *it = tile;
    }

    std::array<TiledRange1, 2ul> tranges =
        {{ left.trange().data().front(), right.trange().data().back() }};
    result_tr = typename impl_type::trange_type(tranges.begin(), tranges.end());

    pmap.reset(new detail::BlockedPmap(* GlobalFixture::world, result_tr.tiles().volume()));
  }

  ~ContractionEvalFixture() {
    GlobalFixture::world->gop.fence();
  }

  ArrayN left;
  ArrayN right;
  array_eval_type left_arg;
  array_eval_type right_arg;
  typename impl_type::trange_type result_tr;
  std::shared_ptr<impl_type::pmap_interface> pmap;
  op_type op;

}; // ContractionEvalFixture

BOOST_FIXTURE_TEST_SUITE( dist_eval_contraction_eval_suite, ContractionEvalFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  BOOST_REQUIRE_NO_THROW(detail::make_contract_eval(left_arg, right_arg, left.get_world(),
      DenseShape(), pmap, Permutation(), op));


  dist_eval_type1 contract = detail::make_contract_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), pmap, Permutation(), op);

  BOOST_CHECK_EQUAL(& contract.get_world(), GlobalFixture::world);
  BOOST_CHECK(contract.pmap() == pmap);
  BOOST_CHECK_EQUAL(contract.range(), result_tr.tiles());
  BOOST_CHECK_EQUAL(contract.trange(), result_tr);
  BOOST_CHECK_EQUAL(contract.size(), result_tr.tiles().volume());
  BOOST_CHECK(contract.is_dense());
  for(std::size_t i = 0; i < result_tr.tiles().volume(); ++i)
    BOOST_CHECK(! contract.is_zero(i));

}


BOOST_AUTO_TEST_CASE( eval )
{
  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  dist_eval_type1 contract = detail::make_contract_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), pmap, Permutation(), op);

  // Check evaluation
  BOOST_REQUIRE_NO_THROW(contract.eval());

  // Compute matrix element dimensions.
  const std::size_t m = left.trange().elements().size().front();
  const std::size_t n = right.trange().elements().size().back();
  const std::size_t k = left.trange().elements().volume() / m;

  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;
  matrix_type l(m, k), r(k, n);

  // Copy all tiles of left into local l matrix.
  {
    const std::size_t left_middle = left.range().dim() - op.num_contract_ranks();
    for(std::size_t index = 0ul; index < left.size(); ++index) {
      const ArrayN::value_type tile = left.find(index);

      std::size_t m0 = 1ul, m = 1ul, k0 = 1ul, k = 1ul;
      std::size_t dim;
      for(dim = 0ul; dim < left_middle; ++dim) {
        m0 *= tile.range().start()[dim];
        m *= tile.range().size()[dim];
      }
      for(; dim < left.range().dim(); ++dim) {
        k0 *= tile.range().start()[dim];
        k *= tile.range().size()[dim];
      }

      l.block(m0, k0, m, k) = eigen_map(tile, m, k);
    }
  }

  // Copy all tiles of right into local r matrix.
  {
    const std::size_t right_middle = op.num_contract_ranks();
    for(std::size_t index = 0ul; index < right.size(); ++index) {
      const ArrayN::value_type tile = right.find(index);

      std::size_t k0 = 1ul, k = 1ul, n0 = 1ul, n = 1ul;
      std::size_t dim;
      for(dim = 0ul; dim < right_middle; ++dim) {
        k0 *= tile.range().start()[dim];
        k *= tile.range().size()[dim];
      }
      for(; dim < right.range().dim(); ++dim) {
        n0 *= tile.range().start()[dim];
        n *= tile.range().size()[dim];
      }

      r.block(k0, n0, k, n) = eigen_map(tile, k, n);
    }
  }

  // Compute the reference contraction
  const matrix_type reference = l * r;

  dist_eval_type1::pmap_interface::const_iterator it = contract.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = contract.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {

    // Get the array evaluator tile.
    madness::Future<dist_eval_type1::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = contract.move(*it));

    // Force the evaluation of the tile
    dist_eval_type1::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());
    BOOST_CHECK(! eval_tile.empty());

    if(!eval_tile.empty()) {
      // Check that the result tile is correctly modified.
      BOOST_CHECK_EQUAL(eval_tile.range(), contract.trange().make_tile_range(*it));
      BOOST_CHECK(eigen_map(eval_tile) == reference.block(eval_tile.range().start()[0],
          eval_tile.range().start()[1], eval_tile.range().size()[0], eval_tile.range().size()[1]));
    }
  }

}

BOOST_AUTO_TEST_SUITE_END()
