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

#ifndef TILEDARRAY_TEST_CONTRACTION_HELPERS_H__INCLUDED
#define TILEDARRAY_TEST_CONTRACTION_HELPERS_H__INCLUDED
#include "TiledArray/expressions/contraction_helpers.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

BOOST_AUTO_TEST_SUITE(range_from_annotation_fxn)

BOOST_AUTO_TEST_CASE(vvv) {
  Range r14({1}, {4});
  Tensor<double> lhs(r14), rhs(r14);
  BipartiteIndexList i("i");
  auto r = range_from_annotation(i, i, i, lhs, rhs);
  BOOST_CHECK(r == r14);
}

BOOST_AUTO_TEST_CASE(vvm) {
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14), rhs(r14_23);
  BipartiteIndexList i("i"), ij("i,j");
  auto r = range_from_annotation(i, i, ij, lhs, rhs);
  BOOST_CHECK(r == r14);
}

BOOST_AUTO_TEST_CASE(vmv) {
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14_23), rhs(r14);
  BipartiteIndexList i("i"), ij("i,j");
  auto r = range_from_annotation(i, ij, i, lhs, rhs);
  BOOST_CHECK(r == r14);
}

BOOST_AUTO_TEST_CASE(mvm) {
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14), rhs(r14_23);
  BipartiteIndexList i("i"), ij("i,j");
  auto r = range_from_annotation(ij, i, ij, lhs, rhs);
  BOOST_CHECK(r == r14_23);
}

BOOST_AUTO_TEST_CASE(mmv) {
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14_23), rhs(r14);
  BipartiteIndexList i("i"), ij("i,j");
  auto r = range_from_annotation(ij, ij, i, lhs, rhs);
  BOOST_CHECK(r == r14_23);
}

BOOST_AUTO_TEST_CASE(mmm_hadamard) {
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14_23), rhs(r14_23);
  BipartiteIndexList ij("i,j");
  auto r = range_from_annotation(ij, ij, ij, lhs, rhs);
  BOOST_CHECK(r == r14_23);
}

BOOST_AUTO_TEST_CASE(mmm_contract) {
  Range r14_23({1, 2}, {4, 3});
  Range r23_35({2, 3}, {3, 5});
  Range r14_35({1, 3}, {4, 5});
  Range r23({2}, {3});

  Tensor<double> lhs(r14_23), rhs(r23_35);
  BipartiteIndexList ij("i,j"), jk("j, k"), ik("i, k"), j("j");
  {
    auto r = range_from_annotation(ik, ij, jk, lhs, rhs);
    BOOST_CHECK(r == r14_35);
  }

  {
    auto r = range_from_annotation(j, ij, jk, lhs, rhs);
    BOOST_CHECK(r == r23);
  }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(make_index_fxn)

using index_type = std::vector<std::size_t>;

BOOST_AUTO_TEST_CASE(one_free) {
  BipartiteIndexList i("i");
  auto idx =
      make_index(i, BipartiteIndexList{}, i, index_type{0}, index_type{});
  BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(one_bound) {
  BipartiteIndexList i("i");
  auto idx =
      make_index(BipartiteIndexList{}, i, i, index_type{}, index_type{0});
  BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(is_free) {
  BipartiteIndexList i("i"), j("j");
  auto idx = make_index(i, j, i, index_type{0}, index_type{1});
  BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(is_bound) {
  BipartiteIndexList i("i"), j("j");
  auto idx = make_index(j, i, i, index_type{1}, index_type{0});
  BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(both_bound) {
  BipartiteIndexList i("i"), j("j"), ij("i, j");
  auto idx = make_index(j, i, ij, index_type{1}, index_type{0});
  index_type corr{0, 1};
  BOOST_CHECK_EQUAL(idx, corr);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(einsum)

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_TEST_CONTRACTION_HELPERS_H__INCLUDED
