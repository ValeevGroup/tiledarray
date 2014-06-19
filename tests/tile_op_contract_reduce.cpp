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
 *  tile_op_contract_reduce.cpp
 *  Dec 16, 2013
 *
 */

#include "TiledArray/tensor.h"
#include "TiledArray/tile_op/contract_reduce.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::math;

struct ContractReduceFixture {
  typedef TiledArray::Tensor<int> tensor_type;
  typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_type;

  ContractReduceFixture() { }

  ~ContractReduceFixture() { }

  static tensor_type make_tensor(const std::size_t i0, const std::size_t j0,
      const std::size_t i, const std::size_t j)
  {
    std::size_t start[2] = {i0, j0}, finish[2] = {i, j};
    tensor_type result(typename tensor_type::range_type(start, finish));
    rand_fill(result);
    return result;
  }

  static tensor_type make_tensor(const std::size_t i0, const std::size_t j0, const std::size_t k0,
      const std::size_t i, const std::size_t j, const std::size_t k)
  {
    std::size_t start[3] = {i0, j0, k0}, finish[3] = {i, j, k};
    tensor_type result(typename tensor_type::range_type(start, finish));
    rand_fill(result);
    return result;
  }

  static void rand_fill(tensor_type& tensor) {
    for(std::size_t i = 0ul; i < tensor.size(); ++i)
      tensor[i] = GlobalFixture::world->rand() % 27;
  }

}; // ContractReduceFixture

#define TA_CHECK_TENSOR_MATRIX_EQUAL( t , m ) \
{ \
  std::size_t i[2]; \
  for(i[0] = t.range().start()[0]; i[0] < t.range().finish()[0]; ++i[0]) { \
    for(i[1] = t.range().start()[1]; i[1] < t.range().finish()[1]; ++i[1]) { \
      BOOST_TEST_CHECKPOINT( "Checking result at i = {" << i[0] << "," << i[1] << "}" ); \
      BOOST_CHECK_EQUAL(t[i], m(i[0] - t.range().start()[0], i[1] - t.range().start()[1])); \
    } \
  } \
}

BOOST_FIXTURE_TEST_SUITE( tile_op_contract_reduce_suite, ContractReduceFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW((ContractReduce<tensor_type, tensor_type, tensor_type>( \
      madness::cblas::NoTrans, madness::cblas::NoTrans, 1, 2u, 2u, 2u)));
}

BOOST_AUTO_TEST_CASE( make_result )
{
  // Check the seed operation produces an empty tensor.
  ContractReduce<tensor_type, tensor_type, tensor_type>
  op(madness::cblas::NoTrans, madness::cblas::NoTrans, 1, 2u, 2u, 2u);
  tensor_type result;
  BOOST_REQUIRE_NO_THROW(result = op());
  BOOST_CHECK(result.empty());

}

BOOST_AUTO_TEST_CASE( permute_empty )
{
  // Check the seed operation produces an empty tensor.
  ContractReduce<tensor_type, tensor_type, tensor_type>
  op(madness::cblas::NoTrans, madness::cblas::NoTrans, 1, 2u, 2u, 2u);
  tensor_type t, result;
  BOOST_REQUIRE_NO_THROW(result = op(t));

}

// TODO: Test non-empty permutation

BOOST_AUTO_TEST_CASE( matrix_multiply )
{
  // Set dimension constants
  const std::size_t
      left_outer_start = 2, left_outer_finish = 20,
      inner_start = 3, inner_finish = 30,
      right_outer_start = 4, right_outer_finish = 40;

  // Construct tensors
  tensor_type left = make_tensor(left_outer_start, inner_start, left_outer_finish, inner_finish);
  tensor_type leftT = make_tensor(inner_start, left_outer_start, inner_finish, left_outer_finish);
  tensor_type right = make_tensor(inner_start, right_outer_start, inner_finish, right_outer_finish);
  tensor_type rightT = make_tensor(right_outer_start, inner_start, right_outer_finish, inner_finish);

  const std::size_t m = left_outer_finish - left_outer_start;
  const std::size_t n = right_outer_finish - right_outer_start;
  const std::size_t k = inner_finish - inner_start;

  // Construct matrix maps for argument tensors
  Eigen::Map<const matrix_type, Eigen::AutoAlign>
    A(left.data(), m, k), B(right.data(), k, n),
    AT(leftT.data(), k, m), BT(rightT.data(), n, k);

  tensor_type result;

  ////////////////////////
  // Test NoTrans, NoTrans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::NoTrans, madness::cblas::NoTrans, 3, 2u, 2u, 2u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, left, right));
  }

  // Compute reference values and compare to the result
  matrix_type  C = 3 * A * B;
  Eigen::Map<const matrix_type, Eigen::AutoAlign> result_map(result.data(), m, n);
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test Trans, NoTrans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::Trans, madness::cblas::NoTrans, 3, 2u, 2u, 2u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, leftT, right));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  C.noalias() += 3 * AT.transpose() * B;
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test NoTrans, Trans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::NoTrans, madness::cblas::Trans, 3, 2u, 2u, 2u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, left, rightT));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  C.noalias() += 3 * A * BT.transpose();
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test Trans, Trans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::Trans, madness::cblas::Trans, 3, 2u, 2u, 2u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, leftT, rightT));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  C.noalias() += 3 * AT.transpose() * BT.transpose();
  BOOST_CHECK_EQUAL(result_map, C);
}

BOOST_AUTO_TEST_CASE( tensor_contract1 )
{
  // Set dimension constants
  const std::size_t
      left_outer_start = 2, left_outer_finish = 20,
      inner1_start = 3, inner1_finish = 30,
      inner2_start = 4, inner2_finish = 40,
      right_outer_start = 5, right_outer_finish = 50;

  // Construct tensors
  tensor_type left = make_tensor(left_outer_start, inner1_start, inner2_start, left_outer_finish, inner1_finish, inner2_finish);
  tensor_type right = make_tensor(inner1_start, inner2_start, right_outer_start, inner1_finish, inner2_finish, right_outer_finish);
  tensor_type leftT = make_tensor(inner1_start, inner2_start, left_outer_start, inner1_finish, inner2_finish, left_outer_finish);
  tensor_type rightT = make_tensor(right_outer_start, inner1_start, inner2_start, right_outer_finish, inner1_finish, inner2_finish);

  const std::size_t m = left_outer_finish - left_outer_start;
  const std::size_t n = right_outer_finish - right_outer_start;
  const std::size_t k = (inner1_finish - inner1_start) * (inner2_finish - inner2_start);

  // Construct matrix maps for argument tensors
  Eigen::Map<const matrix_type, Eigen::AutoAlign>
    A(left.data(), m, k), B(right.data(), k, n),
    AT(leftT.data(), k, m), BT(rightT.data(), n, k);

  tensor_type result;

  ////////////////////////
  // Test NoTrans, NoTrans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::NoTrans, madness::cblas::NoTrans, 3, 2u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, left, right));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  matrix_type  C = 3 * A * B;
  Eigen::Map<const matrix_type, Eigen::AutoAlign> result_map(result.data(), m, n);
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test Trans, NoTrans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::Trans, madness::cblas::NoTrans, 3, 2u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, leftT, right));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  C.noalias() += 3 * AT.transpose() * B;
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test NoTrans, Trans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::NoTrans, madness::cblas::Trans, 3, 2u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, left, rightT));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  C.noalias() += 3 * A * BT.transpose();
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test Trans, Trans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::Trans, madness::cblas::Trans, 3, 2u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, leftT, rightT));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], right_outer_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], right_outer_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], m);
  BOOST_CHECK_EQUAL(result.range().size()[1], n);

  // Compute reference values and compare to the result
  C.noalias() += 3 * AT.transpose() * BT.transpose();
  BOOST_CHECK_EQUAL(result_map, C);
}

BOOST_AUTO_TEST_CASE( tensor_contract2 )
{
  // Set dimension constants
  const std::size_t
      left_outer1_start = 2, left_outer1_finish = 20,
      left_outer2_start = 3, left_outer2_finish = 30,
      inner_start = 3, inner_finish = 30,
      right_outer1_start = 5, right_outer1_finish = 50,
      right_outer2_start = 5, right_outer2_finish = 50;

  // Construct tensors
  tensor_type left = make_tensor(left_outer1_start, left_outer2_start, inner_start, left_outer1_finish, left_outer2_finish, inner_finish);
  tensor_type right = make_tensor(inner_start, right_outer1_start, right_outer2_start, inner_finish, right_outer1_finish, right_outer2_finish);
  tensor_type leftT = make_tensor(inner_start, left_outer1_start, left_outer2_start, inner_finish, left_outer1_finish, left_outer2_finish);
  tensor_type rightT = make_tensor(right_outer1_start, right_outer2_start, inner_start, right_outer1_finish, right_outer2_finish, inner_finish);

  const std::size_t m = (left_outer1_finish - left_outer1_start) * (left_outer2_finish - left_outer2_start);
  const std::size_t n = (right_outer1_finish - right_outer1_start) * (right_outer2_finish - right_outer2_start);
  const std::size_t k = inner_finish - inner_start;

  // Construct matrix maps for argument tensors
  Eigen::Map<const matrix_type, Eigen::AutoAlign>
    A(left.data(), m, k), B(right.data(), k, n),
    AT(leftT.data(), k, m), BT(rightT.data(), n, k);

  tensor_type result;

  ////////////////////////
  // Test NoTrans, NoTrans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::NoTrans, madness::cblas::NoTrans, 3, 4u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, left, right));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().start()[2], right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[3], right_outer2_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], left_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[2], right_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[3], right_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], left_outer1_finish - left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[1], left_outer2_finish - left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().size()[2], right_outer1_finish - right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[3], right_outer2_finish - right_outer2_start);

  // Compute reference values and compare to the result
  matrix_type  C = 3 * A * B;
  Eigen::Map<const matrix_type, Eigen::AutoAlign> result_map(result.data(), m, n);
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test Trans, NoTrans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::Trans, madness::cblas::NoTrans, 3, 4u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, leftT, right));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().start()[2], right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[3], right_outer2_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], left_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[2], right_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[3], right_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], left_outer1_finish - left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[1], left_outer2_finish - left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().size()[2], right_outer1_finish - right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[3], right_outer2_finish - right_outer2_start);

  // Compute reference values and compare to the result
  C.noalias() += 3 * AT.transpose() * B;
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test NoTrans, Trans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::NoTrans, madness::cblas::Trans, 3, 4u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, left, rightT));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().start()[2], right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[3], right_outer2_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], left_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[2], right_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[3], right_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], left_outer1_finish - left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[1], left_outer2_finish - left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().size()[2], right_outer1_finish - right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[3], right_outer2_finish - right_outer2_start);

  // Compute reference values and compare to the result
  C.noalias() += 3 * A * BT.transpose();
  BOOST_CHECK_EQUAL(result_map, C);

  ////////////////////////
  // Test Trans, Trans
  {
    ContractReduce<tensor_type, tensor_type, tensor_type>
    op(madness::cblas::Trans, madness::cblas::Trans, 3, 4u, 3u, 3u);

    // Do contraction operation
    BOOST_REQUIRE_NO_THROW(op(result, leftT, rightT));
  }

  // Check dimensions of the result
  BOOST_CHECK_EQUAL(result.range().start()[0], left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[1], left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().start()[2], right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().start()[3], right_outer2_start);
  BOOST_CHECK_EQUAL(result.range().finish()[0], left_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[1], left_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[2], right_outer1_finish);
  BOOST_CHECK_EQUAL(result.range().finish()[3], right_outer2_finish);
  BOOST_CHECK_EQUAL(result.range().size()[0], left_outer1_finish - left_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[1], left_outer2_finish - left_outer2_start);
  BOOST_CHECK_EQUAL(result.range().size()[2], right_outer1_finish - right_outer1_start);
  BOOST_CHECK_EQUAL(result.range().size()[3], right_outer2_finish - right_outer2_start);

  // Compute reference values and compare to the result
  C.noalias() += 3 * AT.transpose() * BT.transpose();
  BOOST_CHECK_EQUAL(result_map, C);
}


BOOST_AUTO_TEST_SUITE_END()
