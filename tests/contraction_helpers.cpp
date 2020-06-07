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
#include "tiledarray.h"
#include "TiledArray/expressions/contraction_helpers.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

BOOST_AUTO_TEST_SUITE(range_from_annotation_fxn)

BOOST_AUTO_TEST_CASE(vvv){
  Range r14({1}, {4});
  Tensor<double> lhs(r14), rhs(r14);
  VariableList i("i");
  auto r = range_from_annotation(i, i, i, lhs, rhs);
  BOOST_CHECK(r == r14);
}

BOOST_AUTO_TEST_CASE(vvm){
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14), rhs(r14_23);
  VariableList i("i"), ij("i,j");
  auto r = range_from_annotation(i, i, ij, lhs, rhs);
  BOOST_CHECK(r == r14);
}

BOOST_AUTO_TEST_CASE(vmv){
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14_23), rhs(r14);
  VariableList i("i"), ij("i,j");
  auto r = range_from_annotation(i, ij, i, lhs, rhs);
  BOOST_CHECK(r == r14);
}

BOOST_AUTO_TEST_CASE(mvm){
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14), rhs(r14_23);
  VariableList i("i"), ij("i,j");
  auto r = range_from_annotation(ij, i, ij, lhs, rhs);
  BOOST_CHECK(r == r14_23);
}

BOOST_AUTO_TEST_CASE(mmv){
  Range r14({1}, {4});
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14_23), rhs(r14);
  VariableList i("i"), ij("i,j");
  auto r = range_from_annotation(ij, ij, i, lhs, rhs);
  BOOST_CHECK(r == r14_23);
}

BOOST_AUTO_TEST_CASE(mmm_hadamard){
  Range r14_23({1, 2}, {4, 3});
  Tensor<double> lhs(r14_23), rhs(r14_23);
  VariableList ij("i,j");
  auto r = range_from_annotation(ij, ij, ij, lhs, rhs);
  BOOST_CHECK(r == r14_23);
}

BOOST_AUTO_TEST_CASE(mmm_contract){
  Range r14_23({1, 2}, {4, 3});
  Range r23_35({2, 3}, {3, 5});
  Range r14_35({1, 3}, {4, 5});
  Range r23({2}, {3});

  Tensor<double> lhs(r14_23), rhs(r23_35);
  VariableList ij("i,j"), jk("j, k"), ik("i, k"), j("j");
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


BOOST_AUTO_TEST_CASE(one_free){
    VariableList i("i");
    auto idx = make_index(i, VariableList{}, i, index_type{0}, index_type{});
    BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(one_bound){
    VariableList i("i");
    auto idx = make_index(VariableList{}, i, i, index_type{}, index_type{0});
    BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(is_free){
    VariableList i("i"), j("j");
    auto idx = make_index(i, j, i, index_type{0}, index_type{1});
    BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(is_bound){
    VariableList i("i"), j("j");
    auto idx = make_index(j, i, i, index_type{1}, index_type{0});
    BOOST_CHECK_EQUAL(idx, index_type{0});
}

BOOST_AUTO_TEST_CASE(both_bound){
    VariableList i("i"), j("j"), ij("i, j");
    auto idx = make_index(j, i, ij, index_type{1}, index_type{0});
    index_type corr{0, 1};
    BOOST_CHECK_EQUAL(idx, corr);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(s_t_t_contract)

BOOST_AUTO_TEST_CASE(vv){
  Tensor<double> lhs(Range{3}, {60, 96, 86});
  Tensor<double> rhs(Range{3}, {97, 80, 68});
  double corr = 19348;
  VariableList empty, lidx("i"), ridx("i");
  auto rv = kernels::s_t_t_contract_(empty, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(mm){
  Tensor<double> lhs(Range{3, 10}, {97, 61, 8, 33, 29, 35, 69, 16, 96, 94, 39, 40, 49, 19, 89, 25, 35, 40, 93, 44, 74, 39, 31, 24, 17, 84, 80, 97, 39, 36});
  Tensor<double> rhs(Range{3, 10}, {67, 63, 98, 52, 40, 78, 52, 16, 95, 87, 84, 8, 20, 82, 24, 40, 79, 9, 50, 62, 46, 24, 63, 36, 93, 16, 52, 48, 4, 83});
  double corr = 79689;
  VariableList empty, lidx("i,j"), ridx("i,j");
  auto rv = kernels::s_t_t_contract_(empty, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(tt){
  Tensor<double> lhs(Range{3, 10, 2}, {72, 37, 15, 44, 35, 64, 40, 64, 38, 60, 94, 24, 93, 25, 68, 27, 17, 49, 53, 53, 58, 42, 14, 32, 39, 9, 58, 1, 18, 63, 80, 96, 64, 67, 66, 26, 18, 92, 33, 99, 10, 5, 54, 54, 32, 35, 20, 27, 33, 11, 93, 85, 37, 30, 28, 47, 12, 52, 21, 12});
  Tensor<double> rhs(Range{3, 10, 2}, {70, 1, 70, 63, 7, 96, 56, 2, 66, 99, 61, 21, 37, 89, 90, 65, 85, 37, 94, 53, 66, 6, 15, 68, 34, 54, 21, 63, 80, 9, 82, 56, 33, 49, 81, 76, 67, 94, 93, 10, 79, 34, 86, 37, 3, 9, 10, 25, 65, 68, 74, 36, 95, 73, 59, 54, 72, 75, 69, 3});
  double corr = 144931;
  VariableList empty, lidx("i,j,k"), ridx("i,j,k");
  auto rv = kernels::s_t_t_contract_(empty, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_SUITE_END()

#endif // TILEDARRAY_TEST_CONTRACTION_HELPERS_H__INCLUDED
