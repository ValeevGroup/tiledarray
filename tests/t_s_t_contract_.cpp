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
#include "TiledArray/expressions/contraction_helpers.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

BOOST_AUTO_TEST_SUITE(t_s_t_contract_fxn)

BOOST_AUTO_TEST_CASE(i_i){
double lhs = 3.0;
  Tensor<double> rhs(Range{4}, {19, 86, 78, 26});
  Tensor<double> corr(Range{4}, {57.0, 258.0, 234.0, 78.0});
  VariableList oidx("i"), lidx, ridx("i");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ij_ij){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2}, {71, 49, 20, 28, 9, 98, 100, 74});
  Tensor<double> corr(Range{4, 2}, {213.0, 147.0, 60.0, 84.0, 27.0, 294.0, 300.0, 222.0});
  VariableList oidx("i,j"), lidx, ridx("i,j");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ji_ij){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2}, {71, 49, 20, 28, 9, 98, 100, 74});
  Tensor<double> corr(Range{2, 4}, {213.0, 60.0, 27.0, 300.0, 147.0, 84.0, 294.0, 222.0});
  VariableList oidx("j,i"), lidx, ridx("i,j");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_ijk){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2, 6}, {98, 99, 25, 83, 14, 50, 89, 78, 71, 8, 69, 3, 66, 98, 25, 90, 32, 17, 35, 39, 27, 4, 32, 63, 38, 14, 88, 73, 93, 18, 31, 86, 41, 98, 21, 75, 68, 97, 78, 51, 78, 68, 11, 78, 97, 59, 86, 79});
  Tensor<double> corr(Range{4, 2, 6}, {294.0, 297.0, 75.0, 249.0, 42.0, 150.0, 267.0, 234.0, 213.0, 24.0, 207.0, 9.0, 198.0, 294.0, 75.0, 270.0, 96.0, 51.0, 105.0, 117.0, 81.0, 12.0, 96.0, 189.0, 114.0, 42.0, 264.0, 219.0, 279.0, 54.0, 93.0, 258.0, 123.0, 294.0, 63.0, 225.0, 204.0, 291.0, 234.0, 153.0, 234.0, 204.0, 33.0, 234.0, 291.0, 177.0, 258.0, 237.0});
  VariableList oidx("i,j,k"), lidx, ridx("i,j,k");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ikj_ijk){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2, 6}, {98, 99, 25, 83, 14, 50, 89, 78, 71, 8, 69, 3, 66, 98, 25, 90, 32, 17, 35, 39, 27, 4, 32, 63, 38, 14, 88, 73, 93, 18, 31, 86, 41, 98, 21, 75, 68, 97, 78, 51, 78, 68, 11, 78, 97, 59, 86, 79});
  Tensor<double> corr(Range{4, 6, 2}, {294.0, 267.0, 297.0, 234.0, 75.0, 213.0, 249.0, 24.0, 42.0, 207.0, 150.0, 9.0, 198.0, 105.0, 294.0, 117.0, 75.0, 81.0, 270.0, 12.0, 96.0, 96.0, 51.0, 189.0, 114.0, 93.0, 42.0, 258.0, 264.0, 123.0, 219.0, 294.0, 279.0, 63.0, 54.0, 225.0, 204.0, 33.0, 291.0, 234.0, 234.0, 291.0, 153.0, 177.0, 234.0, 258.0, 204.0, 237.0});
  VariableList oidx("i,k,j"), lidx, ridx("i,j,k");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(jik_ijk){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2, 6}, {98, 99, 25, 83, 14, 50, 89, 78, 71, 8, 69, 3, 66, 98, 25, 90, 32, 17, 35, 39, 27, 4, 32, 63, 38, 14, 88, 73, 93, 18, 31, 86, 41, 98, 21, 75, 68, 97, 78, 51, 78, 68, 11, 78, 97, 59, 86, 79});
  Tensor<double> corr(Range{2, 4, 6}, {294.0, 297.0, 75.0, 249.0, 42.0, 150.0, 198.0, 294.0, 75.0, 270.0, 96.0, 51.0, 114.0, 42.0, 264.0, 219.0, 279.0, 54.0, 204.0, 291.0, 234.0, 153.0, 234.0, 204.0, 267.0, 234.0, 213.0, 24.0, 207.0, 9.0, 105.0, 117.0, 81.0, 12.0, 96.0, 189.0, 93.0, 258.0, 123.0, 294.0, 63.0, 225.0, 33.0, 234.0, 291.0, 177.0, 258.0, 237.0});
  VariableList oidx("j,i,k"), lidx, ridx("i,j,k");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(jki_ijk){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2, 6}, {98, 99, 25, 83, 14, 50, 89, 78, 71, 8, 69, 3, 66, 98, 25, 90, 32, 17, 35, 39, 27, 4, 32, 63, 38, 14, 88, 73, 93, 18, 31, 86, 41, 98, 21, 75, 68, 97, 78, 51, 78, 68, 11, 78, 97, 59, 86, 79});
  Tensor<double> corr(Range{2, 6, 4}, {294.0, 198.0, 114.0, 204.0, 297.0, 294.0, 42.0, 291.0, 75.0, 75.0, 264.0, 234.0, 249.0, 270.0, 219.0, 153.0, 42.0, 96.0, 279.0, 234.0, 150.0, 51.0, 54.0, 204.0, 267.0, 105.0, 93.0, 33.0, 234.0, 117.0, 258.0, 234.0, 213.0, 81.0, 123.0, 291.0, 24.0, 12.0, 294.0, 177.0, 207.0, 96.0, 63.0, 258.0, 9.0, 189.0, 225.0, 237.0});
  VariableList oidx("j,k,i"), lidx, ridx("i,j,k");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(kij_ijk){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2, 6}, {98, 99, 25, 83, 14, 50, 89, 78, 71, 8, 69, 3, 66, 98, 25, 90, 32, 17, 35, 39, 27, 4, 32, 63, 38, 14, 88, 73, 93, 18, 31, 86, 41, 98, 21, 75, 68, 97, 78, 51, 78, 68, 11, 78, 97, 59, 86, 79});
  Tensor<double> corr(Range{6, 4, 2}, {294.0, 267.0, 198.0, 105.0, 114.0, 93.0, 204.0, 33.0, 297.0, 234.0, 294.0, 117.0, 42.0, 258.0, 291.0, 234.0, 75.0, 213.0, 75.0, 81.0, 264.0, 123.0, 234.0, 291.0, 249.0, 24.0, 270.0, 12.0, 219.0, 294.0, 153.0, 177.0, 42.0, 207.0, 96.0, 96.0, 279.0, 63.0, 234.0, 258.0, 150.0, 9.0, 51.0, 189.0, 54.0, 225.0, 204.0, 237.0});
  VariableList oidx("k,i,j"), lidx, ridx("i,j,k");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(kji_ijk){
double lhs = 3.0;
  Tensor<double> rhs(Range{4, 2, 6}, {98, 99, 25, 83, 14, 50, 89, 78, 71, 8, 69, 3, 66, 98, 25, 90, 32, 17, 35, 39, 27, 4, 32, 63, 38, 14, 88, 73, 93, 18, 31, 86, 41, 98, 21, 75, 68, 97, 78, 51, 78, 68, 11, 78, 97, 59, 86, 79});
  Tensor<double> corr(Range{6, 2, 4}, {294.0, 198.0, 114.0, 204.0, 267.0, 105.0, 93.0, 33.0, 297.0, 294.0, 42.0, 291.0, 234.0, 117.0, 258.0, 234.0, 75.0, 75.0, 264.0, 234.0, 213.0, 81.0, 123.0, 291.0, 249.0, 270.0, 219.0, 153.0, 24.0, 12.0, 294.0, 177.0, 42.0, 96.0, 279.0, 234.0, 207.0, 96.0, 63.0, 258.0, 150.0, 51.0, 54.0, 204.0, 9.0, 189.0, 225.0, 237.0});
  VariableList oidx("k,j,i"), lidx, ridx("i,j,k");
  auto rv = kernels::t_s_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}


BOOST_AUTO_TEST_SUITE_END()
