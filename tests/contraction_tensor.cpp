/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/contraction_tensor.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct ContractionTensorFixture : public AnnotatedTensorFixture {
  typedef TensorExpression<array_annotation::value_type> tensor_expression;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;

  ContractionTensorFixture() :
      aa_left(a(left_var)), aa_right(a(right_var)),
      ctt(make_contraction_tensor(aa_left, aa_right))
  { }

  static const VariableList left_var;
  static const VariableList right_var;

  array_annotation aa_left;
  array_annotation aa_right;
  tensor_expression ctt;
};

const VariableList ContractionTensorFixture::left_var(
    AnnotatedTensorFixture::make_var_list(0, GlobalFixture::dim));
const VariableList ContractionTensorFixture::right_var(
    AnnotatedTensorFixture::make_var_list(1, GlobalFixture::dim + 1));


BOOST_FIXTURE_TEST_SUITE( contraction_tensor_suite, ContractionTensorFixture )

BOOST_AUTO_TEST_CASE( range )
{
  std::array<TiledRange1, 2> ranges = {{ a.trange().data().front(), a.trange().data().back() }};

  TiledRange dtr(ranges.begin(), ranges.end());
  BOOST_CHECK_EQUAL(ctt.range(), dtr.tiles());
  BOOST_CHECK_EQUAL(ctt.size(), dtr.tiles().volume());
  BOOST_CHECK_EQUAL(ctt.trange(), dtr);
}

BOOST_AUTO_TEST_CASE( vars )
{
  VariableList var2(left_var.data().front() + "," + right_var.data().back());
  BOOST_CHECK_EQUAL(ctt.vars(), var2);
}

BOOST_AUTO_TEST_CASE( shape )
{
  BOOST_CHECK_EQUAL(ctt.is_dense(), a.is_dense());
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(ctt.get_shape(), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( result )
{
  // Get tiling dimensions for contraction
  const std::size_t M = a.trange().tiles().size().front();
  const std::size_t N = a.trange().tiles().size().back();
  const std::size_t K = a.trange().tiles().volume() / M;
  const std::size_t size = M * N;

  // Evaluate and wait for it to finish.
  ctt.eval(ctt.vars(), std::shared_ptr<tensor_expression::pmap_interface>(
      new TiledArray::detail::BlockedPmap(* GlobalFixture::world, size))).get();


  // Check that the range is unchanged
  BOOST_CHECK_EQUAL(ctt.trange().tiles().size()[0], a.trange().tiles().size().front());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().size()[1], a.trange().tiles().size().back());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().start()[0], a.trange().tiles().start().front());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().start()[1], a.trange().tiles().start().back());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().finish()[0], a.trange().tiles().finish().front());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().finish()[1], a.trange().tiles().finish().back());
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[0], a.trange().elements().size().front());
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[1], a.trange().elements().size().back());
  BOOST_CHECK_EQUAL(ctt.trange().elements().start()[0], a.trange().elements().start().front());
  BOOST_CHECK_EQUAL(ctt.trange().elements().start()[1], a.trange().elements().start().back());
  BOOST_CHECK_EQUAL(ctt.trange().elements().finish()[0], a.trange().elements().finish().front());
  BOOST_CHECK_EQUAL(ctt.trange().elements().finish()[1], a.trange().elements().finish().back());

  // Check that all the tiles have been evaluated.
  world.gop.fence();
  std::size_t x = std::distance(ctt.begin(), ctt.end());
  world.gop.sum(x);
  BOOST_CHECK_EQUAL(ctt.size(), x);


  for(std::size_t m = 0ul; m < M; ++m) {
    for(std::size_t n = 0ul; n < N; ++n) {

      madness::Future<tensor_expression::value_type> tile = ctt[m * N + n];
//      std::stringstream ss;
//      ss << tile << "\n";

      // Get tile dimensions
      const std::size_t I = tile.get().range().size()[0];
      const std::size_t J = tile.get().range().size()[1];

//      ss << "I = " << I << " J = " << J << "\n";
//      std::cout << ss.str();

      // Create a matrix to hold the expected contraction result
      matrix_type result(I, J);
      result.fill(0);

      // Compute the expected value of the result tile
      for(std::size_t k = 0ul; k < K; ++k) {
        // Get the contraction arguments for contraction k
        madness::Future<ArrayN::value_type> left = a.find(m * K + k);
        madness::Future<ArrayN::value_type> right = a.find(k * N + n);

        const std::size_t L = left.get().range().volume() / I;

        // Construct an equivilant matrix for the left and right argument tiles
        Eigen::Map<const matrix_type> left_matrix(left.get().data(), I, L);
        Eigen::Map<const matrix_type> right_matrix(right.get().data(), L, J);

        // Add to the contraction result
        result.noalias() += left_matrix * right_matrix;

      }

      // Check that the result tile is correct.
      for(std::size_t i = 0ul; i < I; ++i) {
        for(std::size_t j = 0ul; j < J; ++j) {
          BOOST_CHECK_EQUAL(result(i,j), tile.get()[i * J + j]);
        }
      }
    }
  }

}

BOOST_AUTO_TEST_CASE( permute_result )
{
  Permutation p(1,0);

  const std::size_t M = a.trange().tiles().size().front();
  const std::size_t N = a.trange().tiles().size().back();
  const std::size_t K = a.trange().tiles().volume() / M;
  const std::size_t size = M * N;

  // Evaluate and wait for it to finish.
  ctt.eval(p ^ ctt.vars(), std::shared_ptr<tensor_expression::pmap_interface>(
      new TiledArray::detail::BlockedPmap(* GlobalFixture::world, size))).get();

  // Check that the range has been permuted correctly.
  BOOST_CHECK_EQUAL(ctt.trange().tiles().size()[0], a.trange().tiles().size().back());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().size()[1], a.trange().tiles().size().front());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().start()[0], a.trange().tiles().start().back());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().start()[1], a.trange().tiles().start().front());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().finish()[0], a.trange().tiles().finish().back());
  BOOST_CHECK_EQUAL(ctt.trange().tiles().finish()[1], a.trange().tiles().finish().front());

  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[0], a.trange().elements().size().back());
  BOOST_CHECK_EQUAL(ctt.trange().elements().size()[1], a.trange().elements().size().front());
  BOOST_CHECK_EQUAL(ctt.trange().elements().start()[0], a.trange().elements().start().back());
  BOOST_CHECK_EQUAL(ctt.trange().elements().start()[1], a.trange().elements().start().front());
  BOOST_CHECK_EQUAL(ctt.trange().elements().finish()[0], a.trange().elements().finish().back());
  BOOST_CHECK_EQUAL(ctt.trange().elements().finish()[1], a.trange().elements().finish().front());

  // Check that all the tiles have been evaluated.
  world.gop.fence();
  std::size_t n = std::distance(ctt.begin(), ctt.end());
  world.gop.sum(n);
  BOOST_CHECK_EQUAL(ctt.size(), n);

  for(std::size_t m = 0ul; m < M; ++m) {
    for(std::size_t n = 0ul; n < N; ++n) {

      madness::Future<tensor_expression::value_type> tile = ctt[n * M + m];

      // Get tile dimensions
      const std::size_t I = tile.get().range().size().back();
      const std::size_t J = tile.get().range().size().front();

      // Create a matrix to hold the expected contraction result
      matrix_type result(I, J);
      result.fill(0);

      // Compute the expected value of the result tile
      for(std::size_t k = 0ul; k < K; ++k) {
        // Get the contraction arguments for contraction k
        madness::Future<ArrayN::value_type> left = a.find(m * K + k);
        madness::Future<ArrayN::value_type> right = a.find(k * N + n);

        const std::size_t L = left.get().range().volume() / I;

        // Construct an equivilant matrix for the left and right argument tiles
        Eigen::Map<const matrix_type> left_matrix(left.get().data(), I, L);
        Eigen::Map<const matrix_type> right_matrix(right.get().data(), J, L);

        // Add to the contraction result
        result += left_matrix * right_matrix.transpose();

      }

      // Check that the result tile is correct.
      for(std::size_t i = 0ul; i < I; ++i) {
        for(std::size_t j = 0ul; j < J; ++j) {
          BOOST_CHECK_EQUAL(result(i,j), tile.get()[i * J + j]);
        }
      }
    }
  }

}


BOOST_AUTO_TEST_SUITE_END()
