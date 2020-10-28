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

#include "array_fixture.h"

#include "../src/TiledArray/dist_eval/contraction_eval.h"
#include "../src/tiledarray.h"
#include "sparse_shape_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::detail::ContractReduce;
using TiledArray::detail::Noop;
using TiledArray::detail::UnaryWrapper;

struct ContractionEvalFixture : public SparseShapeFixture {
  typedef Noop<TensorI, TensorI, true> array_base_op_type;
  typedef UnaryWrapper<array_base_op_type> array_op_type;
  typedef detail::DistEval<detail::LazyArrayTile<TensorI, array_op_type>,
                           DensePolicy>
      array_eval_type;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrix_type;

  ContractionEvalFixture()
      : left(*GlobalFixture::world, tr),
        right(*GlobalFixture::world, tr),
        proc_grid(*GlobalFixture::world, tr.tiles_range().extent(0),
                  tr.tiles_range().extent(tr.tiles_range().rank() - 1),
                  tr.elements_range().extent(0),
                  tr.elements_range().extent(tr.elements_range().rank() - 1u)),
        left_arg(make_array_eval(
            left, left.world(), DenseShape(),
            proc_grid.make_row_phase_pmap(tr.tiles_range().volume() /
                                          tr.tiles_range().extent(0)),
            Permutation(), make_array_noop())),
        right_arg(make_array_eval(
            right, right.world(), DenseShape(),
            proc_grid.make_col_phase_pmap(
                tr.tiles_range().volume() /
                tr.tiles_range().extent(tr.tiles_range().rank() - 1u)),
            Permutation(), make_array_noop())),
        result_tr() {
    // Fill arrays with random data
    rand_fill_array(left);
    rand_fill_array(right);

    std::array<TiledRange1, 2ul> tranges = {
        {left.trange().data().front(), right.trange().data().back()}};
    result_tr = TiledRange(tranges.begin(), tranges.end());

    pmap.reset(new detail::BlockedPmap(*GlobalFixture::world,
                                       result_tr.tiles_range().volume()));
  }

  ~ContractionEvalFixture() { GlobalFixture::world->gop.fence(); }

  static ContractReduce<TensorI, TensorI, TensorI, int> make_contract(
      const unsigned int result_rank, const unsigned int left_rank,
      const unsigned int right_rank, const Permutation& perm = Permutation()) {
    return ContractReduce<TensorI, TensorI, TensorI, int>(
        madness::cblas::NoTrans, madness::cblas::NoTrans, 1, result_rank,
        left_rank, right_rank, perm);
  }

  static TiledArray::detail::UnaryWrapper<Noop<TensorI, TensorI, true> >
  make_array_noop(const Permutation& perm = Permutation()) {
    return TiledArray::detail::UnaryWrapper<Noop<TensorI, TensorI, true> >(
        Noop<TensorI, TensorI, true>(), perm);
  }

  template <typename Tile, typename Policy>
  static void rand_fill_array(DistArray<Tile, Policy>& array) {
    // Iterate over local, non-zero tiles
    for (typename DistArray<Tile, Policy>::iterator it = array.begin();
         it != array.end(); ++it) {
      // Construct a new tile with random data
      typename DistArray<Tile, Policy>::value_type tile(
          array.trange().make_tile_range(it.index()));
      for (typename DistArray<Tile, Policy>::value_type::iterator tile_it =
               tile.begin();
           tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 27;

      // Set array tile
      *it = tile;
    }
  }

  template <typename Tile, typename Policy>
  static matrix_type copy_to_matrix(const DistArray<Tile, Policy>& array,
                                    const int middle) {
    // Compute the number of rows and columns in the matrix, and a new weight
    // that is bisected the row and column dimensions.
    std::vector<std::size_t> weight(array.range().rank(), 0ul);
    std::size_t MN[2] = {1ul, 1ul};
    const int dim = array.range().rank();
    int i = dim - 1;
    for (; i >= middle; --i) {
      weight[i] = MN[1];
      MN[1] *= array.trange().elements_range().extent(i);
    }
    for (; i >= 0; --i) {
      weight[i] = MN[0];
      MN[0] *= array.trange().elements_range().extent(i);
    }

    // Construct the result matrix
    matrix_type matrix(MN[0], MN[1]);
    matrix.fill(0);

    // Copy tiles from array to matrix
    for (std::size_t index = 0ul; index < array.size(); ++index) {
      if (array.is_zero(index)) continue;

      // Get tile for index
      const TensorI tile = array.find(index);

      // Compute block start and size
      std::size_t start[2] = {0ul, 0ul}, size[2] = {1ul, 1ul};
      for (i = 0ul; i < middle; ++i) {
        start[0] += tile.range().lobound(i) * weight[i] * size[0];
        size[0] *= tile.range().extent(i);
      }
      for (; i < dim; ++i) {
        start[1] += tile.range().lobound(i) * weight[i] * size[1];
        size[1] *= tile.range().extent(i);
      }

      // Copy tile into matrix sub-block
      matrix.block(start[0], start[1], size[0], size[1]) =
          eigen_map(tile, size[0], size[1]);
    }

    return matrix;
  }

  /// Distributed contraction evaluator factory function

  /// Construct a distributed contraction evaluator, which constructs a new
  /// tensor by applying \c op to tiles of \c left and \c right.
  /// \tparam LeftTile Tile type of the left-hand argument
  /// \tparam RightTile Tile type of the right-hand argument
  /// \tparam Policy The policy type of the argument
  /// \tparam Op The unary tile operation
  /// \param left The left-hand argument
  /// \param right The right-hand argument
  /// \param world The world where the argument will be evaluated
  /// \param shape The shape of the evaluated tensor
  /// \param pmap The process map for the evaluated tensor
  /// \param perm The permutation applied to the tensor
  /// \param op The contraction/reduction tile operation
  template <typename LeftTile, typename RightTile, typename Policy, typename Op>
  TiledArray::detail::DistEval<typename Op::result_type, Policy>
  make_contract_eval(
      const TiledArray::detail::DistEval<LeftTile, Policy>& left,
      const TiledArray::detail::DistEval<RightTile, Policy>& right,
      TiledArray::World& world,
      const typename TiledArray::detail::DistEval<typename Op::result_type,
                                                  Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<
          typename Op::result_type, Policy>::pmap_interface>& pmap,
      const Permutation& perm, const Op& op) {
    TA_ASSERT(left.range().rank() == op.left_rank());
    TA_ASSERT(right.range().rank() == op.right_rank());
    TA_ASSERT((perm.size() == op.result_rank()) || !perm);

    // Define the impl type
    typedef TiledArray::detail::Summa<
        TiledArray::detail::DistEval<LeftTile, Policy>,
        TiledArray::detail::DistEval<RightTile, Policy>, Op, Policy>
        impl_type;

    // Precompute iteration range data
    const unsigned int num_contract_ranks = op.num_contract_ranks();
    const unsigned int left_end = op.left_rank();
    const unsigned int left_middle = left_end - num_contract_ranks;
    const unsigned int right_end = op.right_rank();

    // Construct a vector TiledRange1 objects from the left- and right-hand
    // arguments that will be used to construct the result TiledRange. Also,
    // compute the fused outer dimension sizes, number of tiles and elements,
    // for the contraction.
    typename impl_type::trange_type::Ranges ranges(op.result_rank());
    std::size_t M = 1ul, m = 1ul, N = 1ul, n = 1ul;
    std::size_t pi = 0ul;
    for (unsigned int i = 0ul; i < left_middle; ++i) {
      ranges[(perm ? perm[pi++] : pi++)] = left.trange().data()[i];
      M *= left.range().extent(i);
      m *= left.trange().elements_range().extent(i);
    }
    for (std::size_t i = num_contract_ranks; i < right_end; ++i) {
      ranges[(perm ? perm[pi++] : pi++)] = right.trange().data()[i];
      N *= right.range().extent(i);
      n *= right.trange().elements_range().extent(i);
    }

    // Compute the number of tiles in the inner dimension.
    std::size_t K = 1ul;
    for (std::size_t i = left_middle; i < left_end; ++i)
      K *= left.range().extent(i);

    // Construct the result range
    typename impl_type::trange_type trange(ranges.begin(), ranges.end());

    // Construct the process grid
    TiledArray::detail::ProcGrid proc_grid(world, M, N, m, n);

    return TiledArray::detail::DistEval<typename Op::result_type, Policy>(
        std::shared_ptr<impl_type>(new impl_type(
            left, right, world, trange, shape, pmap, perm, op, K, proc_grid)));
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
            typename TiledArray::DistArray<Tile, Policy>::value_type, Op>,
        Policy>(std::shared_ptr<impl_type>(new impl_type(
        array, world, (perm ? perm * array.trange() : array.trange()), shape,
        pmap, perm, op)));
  }

  TArrayI left;
  TArrayI right;
  detail::ProcGrid proc_grid;
  array_eval_type left_arg;
  array_eval_type right_arg;
  TiledRange result_tr;
  std::shared_ptr<Pmap> pmap;
};  // ContractionEvalFixture

BOOST_FIXTURE_TEST_SUITE(dist_eval_contraction_eval_suite,
                         ContractionEvalFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  BOOST_REQUIRE_NO_THROW(make_contract_eval(
      left_arg, right_arg, left.world(), DenseShape(), pmap, Permutation(),
      make_contract(2u, left_arg.trange().tiles_range().rank(),
                    right_arg.trange().tiles_range().rank())));

  auto contract = make_contract_eval(
      left_arg, right_arg, left_arg.world(), DenseShape(), pmap, Permutation(),
      make_contract(2u, left_arg.trange().tiles_range().rank(),
                    right_arg.trange().tiles_range().rank()));

  BOOST_CHECK_EQUAL(&contract.world(), GlobalFixture::world);
  BOOST_CHECK(contract.pmap() == pmap);
  BOOST_CHECK_EQUAL(contract.range(), result_tr.tiles_range());
  BOOST_CHECK_EQUAL(contract.trange(), result_tr);
  BOOST_CHECK_EQUAL(contract.size(), result_tr.tiles_range().volume());
  BOOST_CHECK(contract.is_dense());
  for (std::size_t i = 0; i < result_tr.tiles_range().volume(); ++i)
    BOOST_CHECK(!contract.is_zero(i));
}

BOOST_AUTO_TEST_CASE(eval) {
  auto contract = make_contract_eval(
      left_arg, right_arg, left_arg.world(), DenseShape(), pmap, Permutation(),
      make_contract(2u, left_arg.trange().tiles_range().rank(),
                    right_arg.trange().tiles_range().rank()));
  using dist_eval_type = decltype(contract);

  // Check evaluation
  BOOST_REQUIRE_NO_THROW(contract.eval());
  BOOST_REQUIRE_NO_THROW(contract.wait());

  // Compute the reference contraction
  const matrix_type l = copy_to_matrix(left, 1),
                    r = copy_to_matrix(right, GlobalFixture::dim - 1);
  const matrix_type reference = l * r;

  // Check that each tile has been properly scaled.
  for (auto index : *contract.pmap()) {
    // Get the array evaluator tile.
    Future<dist_eval_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = contract.get(index));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());
    BOOST_CHECK(!eval_tile.empty());

    if (!eval_tile.empty()) {
      // Check that the result tile is correctly modified.
      BOOST_CHECK_EQUAL(eval_tile.range(),
                        contract.trange().make_tile_range(index));
      BOOST_CHECK(eigen_map(eval_tile) ==
                  reference.block(eval_tile.range().lobound(0),
                                  eval_tile.range().lobound(1),
                                  eval_tile.range().extent(0),
                                  eval_tile.range().extent(1)));
    }
  }
}

BOOST_AUTO_TEST_CASE(perm_eval) {
  Permutation perm({1, 0});

  auto contract = make_contract_eval(
      left_arg, right_arg, left_arg.world(), DenseShape(), pmap, perm,
      make_contract(2u, left_arg.trange().tiles_range().rank(),
                    right_arg.trange().tiles_range().rank(), perm));
  using dist_eval_type = decltype(contract);

  // Check evaluation
  BOOST_REQUIRE_NO_THROW(contract.eval());
  BOOST_REQUIRE_NO_THROW(contract.wait());

  // Compute the reference contraction
  const matrix_type l = copy_to_matrix(left, 1),
                    r = copy_to_matrix(right, GlobalFixture::dim - 1);
  const matrix_type reference = (l * r).transpose();

  // Check that each tile has been properly scaled.
  for (auto index : *contract.pmap()) {
    // Get the array evaluator tile.
    Future<dist_eval_type::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = contract.get(index));

    // Force the evaluation of the tile
    dist_eval_type::eval_type eval_tile;
    BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());
    BOOST_CHECK(!eval_tile.empty());

    if (!eval_tile.empty()) {
      // Check that the result tile is correctly modified.
      BOOST_CHECK_EQUAL(eval_tile.range(),
                        contract.trange().make_tile_range(index));
      BOOST_CHECK(eigen_map(eval_tile) ==
                  reference.block(eval_tile.range().lobound(0),
                                  eval_tile.range().lobound(1),
                                  eval_tile.range().extent(0),
                                  eval_tile.range().extent(1)));
    }
  }
}

BOOST_AUTO_TEST_CASE(sparse_eval) {
  auto do_sparse_eval = [&](bool force_shape) -> void {
    TSpArrayI left(*GlobalFixture::world, tr, make_shape(tr, 0.1, 23));
    TSpArrayI right(*GlobalFixture::world, tr, make_shape(tr, 0.1, 42));

    // Fill arrays with random data
    rand_fill_array(left);
    left.truncate();
    rand_fill_array(right);
    right.truncate();

    auto left_arg = make_array_eval(
        left, left.world(), left.shape(),
        proc_grid.make_row_phase_pmap(tr.tiles_range().volume() /
                                      tr.tiles_range().extent(0)),
        Permutation(), make_array_noop());
    auto right_arg = make_array_eval(
        right, right.world(), right.shape(),
        proc_grid.make_col_phase_pmap(
            tr.tiles_range().volume() /
            tr.tiles_range().extent(tr.tiles_range().rank() - 1)),
        Permutation(), make_array_noop());
    auto op = make_contract(2u, left_arg.trange().tiles_range().rank(),
                            right_arg.trange().tiles_range().rank());

    SparseShape<float> result_shape =
        left_arg.shape().gemm(right_arg.shape(), 1, op.gemm_helper());

    // if needed, force the shape to have 1 tile only
    if (force_shape) {
      auto result_shape_data = result_shape.data().clone();
      bool found_nonzero = false;
      for (auto& i : result_shape_data) {
        if (i >= decltype(result_shape)::threshold() && !found_nonzero) {
          found_nonzero = true;
          i = std::numeric_limits<float>::max();
        } else
          i = 0.0;
      }
      TiledRange result_trange{dims[0], dims[0]};
      result_shape = decltype(result_shape)(result_shape_data, result_trange);
    }

    auto contract = make_contract_eval(left_arg, right_arg, left_arg.world(),
                                       result_shape, pmap, Permutation(), op);
    using dist_eval_type = decltype(contract);

    // Check evaluation
    BOOST_REQUIRE_NO_THROW(contract.eval());
    BOOST_REQUIRE_NO_THROW(contract.wait());

    // Compute the reference contraction
    const matrix_type l = copy_to_matrix(left, 1),
                      r = copy_to_matrix(right, GlobalFixture::dim - 1);
    const matrix_type reference = l * r;

    // Check that each tile has been properly scaled.
    for (auto index : *contract.pmap()) {
      // Skip zero tiles
      if (contract.is_zero(index)) {
        if (!force_shape) {  // can't distinguish forced zeroes, can only check
                             // if shape was not forced
          dist_eval_type::range_type range =
              contract.trange().make_tile_range(index);

          BOOST_CHECK((reference
                           .block(range.lobound(0), range.lobound(1),
                                  range.extent(0), range.extent(1))
                           .array() == 0)
                          .all());
        }
      } else {
        // Get the array evaluator tile.
        Future<dist_eval_type::value_type> tile;
        BOOST_REQUIRE_NO_THROW(tile = contract.get(index));

        // Force the evaluation of the tile
        dist_eval_type::eval_type eval_tile;
        BOOST_REQUIRE_NO_THROW(eval_tile = tile.get());
        BOOST_CHECK(!eval_tile.empty());

        if (!eval_tile.empty()) {
          // Check that the result tile is correctly modified.
          BOOST_CHECK_EQUAL(eval_tile.range(),
                            contract.trange().make_tile_range(index));
          BOOST_CHECK(eigen_map(eval_tile) ==
                      reference.block(eval_tile.range().lobound(0),
                                      eval_tile.range().lobound(1),
                                      eval_tile.range().extent(0),
                                      eval_tile.range().extent(1)));
        }
      }
    }
  };

  do_sparse_eval(false);
  do_sparse_eval(true);
}

BOOST_AUTO_TEST_SUITE_END()
