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
 *  sparse_shape.cpp
 *  Jul 18, 2013
 *
 */

#include "TiledArray/sparse_shape.h"
#include "TiledArray/eigen.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct SparseShapeBaseFixture {
  typedef std::vector<std::size_t> vec_type;

  SparseShapeBaseFixture() : left_tensor(range), right_tensor(range) {
    GlobalFixture::world->srand(37ul);
    for(Tensor<float>::iterator it = left_tensor.begin(); it != left_tensor.end(); ++it)
      *it = GlobalFixture::world->rand() % 101;

    for(Tensor<float>::iterator it = right_tensor.begin(); it != right_tensor.end(); ++it)
      *it = GlobalFixture::world->rand() % 101;
  }

  ~SparseShapeBaseFixture() { }

  static const Range range;

  Tensor<float> left_tensor;
  Tensor<float> right_tensor;

}; // SparseShapeBaseFixture

struct SparseShapeFixture : public SparseShapeBaseFixture {
  typedef std::vector<std::size_t> vec_type;

  SparseShapeFixture() :
    left(left_tensor, 10.1),
    right(right_tensor, 10.1)
  { }

  ~SparseShapeFixture() { }

  static const float factor;
  static const DenseShape dense_shape;

  SparseShape<float> left;
  SparseShape<float> right;

}; // SparseShapeFixture

// Static consts
const Range SparseShapeBaseFixture::range(std::vector<std::size_t>(3, 0), std::vector<std::size_t>(3, 5));
const float SparseShapeFixture::factor = 3.1;
const DenseShape SparseShapeFixture::dense_shape = DenseShape();

BOOST_FIXTURE_TEST_SUITE( sparse_shape_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_CHECK_NO_THROW(SparseShape<float> x);
}

BOOST_AUTO_TEST_CASE( cont_sparse_sparse )
{
  // Create a matrix with the expected output
  const std::size_t m = left.data().range().size().front();
  const std::size_t n = right.data().range().size().back();
  const std::size_t k = left.data().size() / m;
  EigenMatrixXf test_result =
      math::eigen_map(left.data().data(), m, k) * math::eigen_map(right.data().data(), k, n);

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, left.data().range().dim(), right.data().range().dim());
  SparseShape<float> result = left.gemm(right, 1.0, gemm_helper);

  // Check that the result is correct
  std::array<std::size_t, 2> i = {{ 0, 0 }};
  for(i[0] = 0ul; i[0] < 5; ++i[0])
    for(i[1] = 0ul; i[1] < 5; ++i[1])
      BOOST_CHECK_EQUAL(result[i], test_result(i[0], i[1]));
}

BOOST_AUTO_TEST_CASE( cont_sparse_sparse_perm )
{
  // Create a matrix with the expected output
  const std::size_t m = left.data().range().size().front();
  const std::size_t n = right.data().range().size().back();
  const std::size_t k = left.data().size() / m;
  EigenMatrixXf test_result =
      (math::eigen_map(left.data().data(), m, k) *
          math::eigen_map(right.data().data(), k, n)).transpose();

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, left.data().range().dim(), right.data().range().dim());
  SparseShape<float> result = left.gemm(right, 1.0, gemm_helper).perm(Permutation(1,0));

  // Check that the result is correct
  std::array<std::size_t, 2> i = {{ 0, 0 }};
  for(i[0] = 0ul; i[0] < 5; ++i[0])
    for(i[1] = 0ul; i[1] < 5; ++i[1])
      BOOST_CHECK_EQUAL(result.data()[i], test_result(i[0], i[1]));
}

BOOST_AUTO_TEST_SUITE_END()
