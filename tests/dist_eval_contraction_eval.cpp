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
#include "tiledarray.h"
#include "unit_test_config.h"
#include "array_fixture.h"
#include "sparse_shape_fixture.h"

using namespace TiledArray;

struct ContractionEvalFixture : public SparseShapeFixture {
  typedef Array<int, GlobalFixture::dim> ArrayN;
  typedef math::Noop<ArrayN::value_type,
      ArrayN::value_type, true> array_op_type;
  typedef detail::DistEval<detail::LazyArrayTile<ArrayN::value_type, array_op_type>,
      DensePolicy> array_eval_type;
  typedef math::ContractReduce<ArrayN::value_type, ArrayN::value_type, ArrayN::value_type> op_type;
  typedef detail::Summa<array_eval_type, array_eval_type, op_type, DensePolicy> impl_type;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;

  ContractionEvalFixture() :
    left(*GlobalFixture::world, tr),
    right(*GlobalFixture::world, tr),
    proc_grid(*GlobalFixture::world, tr.tiles().size().front(), tr.tiles().size().back(),
        tr.elements().size().front(), tr.elements().size().back()),
    left_arg(make_array_eval(left, left.get_world(), DenseShape(),
        proc_grid.make_row_phase_pmap(tr.tiles().volume() / tr.tiles().size().front()),
        Permutation(), array_op_type())),
    right_arg(make_array_eval(right, right.get_world(), DenseShape(),
        proc_grid.make_col_phase_pmap(tr.tiles().volume() / tr.tiles().size().back()),
        Permutation(), array_op_type())),
    result_tr(),
    op(madness::cblas::NoTrans, madness::cblas::NoTrans, 1, 2u, tr.tiles().dim(), tr.tiles().dim())
  {
    // Fill arrays with random data
    rand_fill_array(left);
    rand_fill_array(right);

    std::array<TiledRange1, 2ul> tranges =
        {{ left.trange().data().front(), right.trange().data().back() }};
    result_tr = impl_type::trange_type(tranges.begin(), tranges.end());

    pmap.reset(new detail::BlockedPmap(* GlobalFixture::world, result_tr.tiles().volume()));
  }

  ~ContractionEvalFixture() {
    GlobalFixture::world->gop.fence();
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  static void rand_fill_array(Array<T, DIM, Tile, Policy>& array) {
    // Iterate over local, non-zero tiles
    for(typename Array<T, DIM, Tile, Policy>::iterator it = array.begin(); it != array.end(); ++it) {
      // Construct a new tile with random data
      typename Array<T, DIM, Tile, Policy>::value_type tile(array.trange().make_tile_range(it.index()));
      for(typename Array<T, DIM, Tile, Policy>::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it)
        *tile_it = GlobalFixture::world->rand() % 27;

      // Set array tile
      *it = tile;
    }
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  static matrix_type copy_to_matrix(const Array<T, DIM, Tile, Policy>& array, const int middle) {

    // Compute the number of rows and columns in the matrix, and a new weight
    // that is bisected the row and column dimensions.
    std::vector<std::size_t> weight(array.range().dim(), 0ul);
    std::size_t MN[2] = { 1ul, 1ul };
    const int dim = array.range().dim();
    int i = dim - 1;
    for(; i >= middle; --i) {
      weight[i] = MN[1];
      MN[1] *= array.trange().elements().size()[i];
    }
    for(; i >= 0; --i) {
      weight[i] = MN[0];
      MN[0] *= array.trange().elements().size()[i];
    }

    // Construct the result matrix
    matrix_type matrix(MN[0], MN[1]);
    matrix.fill(0);

    // Copy tiles from array to matrix
    for(std::size_t index = 0ul; index < array.size(); ++index) {
      if(array.is_zero(index))
        continue;

      // Get tile for index
      const ArrayN::value_type tile = array.find(index);

      // Compute block start and size
      std::size_t start[2] = { 0ul, 0ul }, size[2] = { 1ul, 1ul };
      for(i = 0ul; i < middle; ++i) {
        start[0] += tile.range().start()[i] * weight[i] * size[0];
        size[0] *= tile.range().size()[i];
      }
      for(; i < dim; ++i) {
        start[1] += tile.range().start()[i] * weight[i] * size[1];
        size[1] *= tile.range().size()[i];
      }

      // Copy tile into matrix sub-block
      matrix.block(start[0], start[1], size[0], size[1]) = eigen_map(tile, size[0], size[1]);
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
  TiledArray::detail::DistEval<typename Op::result_type, Policy> make_contract_eval(
      const TiledArray::detail::DistEval<LeftTile, Policy>& left,
      const TiledArray::detail::DistEval<RightTile, Policy>& right,
      TiledArray::World& world,
      const typename TiledArray::detail::DistEval<typename Op::result_type, Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<typename Op::result_type, Policy>::pmap_interface>& pmap,
      const Permutation& perm,
      const Op& op)
  {
    TA_ASSERT(left.range().dim() == op.left_rank());
    TA_ASSERT(right.range().dim() == op.right_rank());
    TA_ASSERT((perm.dim() == op.result_rank()) || !perm);

    // Define the impl type
    typedef TiledArray::detail::Summa<
        TiledArray::detail::DistEval<LeftTile, Policy>,
        TiledArray::detail::DistEval<RightTile, Policy>, Op, Policy> impl_type;

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
    for(unsigned int i = 0ul; i < left_middle; ++i) {
      ranges[(perm ? perm[pi++] : pi++)] = left.trange().data()[i];
      M *= left.range().size()[i];
      m *= left.trange().elements().size()[i];
    }
    for(std::size_t i = num_contract_ranks; i < right_end; ++i) {
      ranges[(perm ? perm[pi++] : pi++)] = right.trange().data()[i];
      N *= right.range().size()[i];
      n *= right.trange().elements().size()[i];
    }

    // Compute the number of tiles in the inner dimension.
    std::size_t K = 1ul;
    for(std::size_t i = left_middle; i < left_end; ++i)
      K *= left.range().size()[i];

    // Construct the result range
    typename impl_type::trange_type trange(ranges.begin(), ranges.end());

    // Construct the process grid
    TiledArray::detail::ProcGrid proc_grid(world, M, N, m, n);

    return TiledArray::detail::DistEval<typename Op::result_type, Policy>(
        std::shared_ptr<impl_type>( new impl_type(left, right, world, trange,
        shape, pmap, perm, op, K, proc_grid)));
  }

  template <typename T, unsigned int DIM, typename Tile, typename Policy, typename Op>
  static TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<typename Array<T, DIM, Tile, Policy>::value_type, Op>, Policy>
  make_array_eval(
      const Array<T, DIM, Tile, Policy>& array,
      TiledArray::World& world,
      const typename TiledArray::detail::DistEval<Tile, Policy>::shape_type& shape,
      const std::shared_ptr<typename TiledArray::detail::DistEval<Tile, Policy>::pmap_interface>& pmap,
      const Permutation& perm,
      const Op& op)
  {
    typedef TiledArray::detail::ArrayEvalImpl<Array<T, DIM, Tile, Policy>, Op, Policy> impl_type;
    return TiledArray::detail::DistEval<TiledArray::detail::LazyArrayTile<typename TiledArray::Array<T, DIM, Tile, Policy>::value_type, Op>, Policy>(
        std::shared_ptr<impl_type>(new impl_type(array, world,
        (perm ? perm ^ array.trange() : array.trange()), shape, pmap, perm, op)));
  }

  ArrayN left;
  ArrayN right;
  detail::ProcGrid proc_grid;
  array_eval_type left_arg;
  array_eval_type right_arg;
  impl_type::trange_type result_tr;
  std::shared_ptr<impl_type::pmap_interface> pmap;
  op_type op;

}; // ContractionEvalFixture

BOOST_FIXTURE_TEST_SUITE( dist_eval_contraction_eval_suite, ContractionEvalFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;

  BOOST_REQUIRE_NO_THROW(make_contract_eval(left_arg, right_arg, left.get_world(),
      DenseShape(), pmap, Permutation(), op));


  dist_eval_type1 contract = make_contract_eval(left_arg, right_arg,
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

  dist_eval_type1 contract = make_contract_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), pmap, Permutation(), op);

  // Check evaluation
  BOOST_REQUIRE_NO_THROW(contract.eval());
  BOOST_REQUIRE_NO_THROW(contract.wait());


  // Compute the reference contraction
  const matrix_type l = copy_to_matrix(left, 1), r = copy_to_matrix(right, GlobalFixture::dim - 1);
  const matrix_type reference = l * r;

  dist_eval_type1::pmap_interface::const_iterator it = contract.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = contract.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {

    // Get the array evaluator tile.
    Future<dist_eval_type1::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = contract.get(*it));

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


BOOST_AUTO_TEST_CASE( perm_eval )
{
  typedef detail::DistEval<op_type::result_type, DensePolicy> dist_eval_type1;
  Permutation perm({1,0});
  op_type pop(madness::cblas::NoTrans, madness::cblas::NoTrans, 1, 2u, tr.tiles().dim(), tr.tiles().dim(), perm);

  dist_eval_type1 contract = make_contract_eval(left_arg, right_arg,
      left_arg.get_world(), DenseShape(), pmap, perm, pop);

  // Check evaluation
  BOOST_REQUIRE_NO_THROW(contract.eval());
  BOOST_REQUIRE_NO_THROW(contract.wait());


  // Compute the reference contraction
  const matrix_type l = copy_to_matrix(left, 1), r = copy_to_matrix(right, GlobalFixture::dim - 1);
  const matrix_type reference = (l * r).transpose();

  dist_eval_type1::pmap_interface::const_iterator it = contract.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = contract.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {

    // Get the array evaluator tile.
    Future<dist_eval_type1::value_type> tile;
    BOOST_REQUIRE_NO_THROW(tile = contract.get(*it));

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

#ifndef TILEDARRAY_ENABLE_OLD_SUMMA

BOOST_AUTO_TEST_CASE( sparse_eval )
{
  typedef detail::DistEval<op_type::result_type, SparsePolicy> dist_eval_type1;
  typedef Array<int, GlobalFixture::dim, Tensor<int>, SparsePolicy> array_type;
  typedef detail::DistEval<detail::LazyArrayTile<array_type::value_type, array_op_type>,
      SparsePolicy> array_eval_type;

  array_type left(*GlobalFixture::world, tr, make_shape(tr, 0.4, 23));

  array_type right(*GlobalFixture::world, tr, make_shape(tr, 0.4, 42));

  // Fill arrays with random data
  rand_fill_array(left);
  left.truncate();
  rand_fill_array(right);
  right.truncate();

  array_eval_type left_arg(make_array_eval(left, left.get_world(), left.get_shape(),
      proc_grid.make_row_phase_pmap(tr.tiles().volume() / tr.tiles().size().front()),
      Permutation(), array_op_type()));
  array_eval_type right_arg(make_array_eval(right, right.get_world(), right.get_shape(),
      proc_grid.make_col_phase_pmap(tr.tiles().volume() / tr.tiles().size().back()),
      Permutation(), array_op_type()));

  const SparseShape<float> result_shape = left_arg.shape().gemm(right_arg.shape(), 1, op.gemm_helper());

  dist_eval_type1 contract = make_contract_eval(left_arg, right_arg,
      left_arg.get_world(), result_shape, pmap, Permutation(), op);

  // Check evaluation
  BOOST_REQUIRE_NO_THROW(contract.eval());
  BOOST_REQUIRE_NO_THROW(contract.wait());

  // Compute the reference contraction
  const matrix_type l = copy_to_matrix(left, 1), r = copy_to_matrix(right, GlobalFixture::dim - 1);
  const matrix_type reference = l * r;

  dist_eval_type1::pmap_interface::const_iterator it = contract.pmap()->begin();
  const dist_eval_type1::pmap_interface::const_iterator end = contract.pmap()->end();

  // Check that each tile has been properly scaled.
  for(; it != end; ++it) {
    // Skip zero tiles
    if(contract.is_zero(*it)) {
      dist_eval_type1::range_type range = contract.trange().make_tile_range(*it);

      BOOST_CHECK((reference.block(range.start()[0], range.start()[1],
          range.size()[0], range.size()[1]).array() == 0).all());

    } else {
      // Get the array evaluator tile.
      Future<dist_eval_type1::value_type> tile;
      BOOST_REQUIRE_NO_THROW(tile = contract.get(*it));

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

}

#endif // TILEDARRAY_ENABLE_OLD_SUMMA

BOOST_AUTO_TEST_SUITE_END()
