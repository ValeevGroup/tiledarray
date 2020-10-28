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

BOOST_AUTO_TEST_SUITE(s_t_t_contract_fxn, TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(i_i) {
  Tensor<double> lhs(Range{4}, {80, 36, 4, 36});
  Tensor<double> rhs(Range{4}, {48, 96, 51, 81});
  double corr = 10416;
  BipartiteIndexList oidx, lidx("i"), ridx("i");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ij_ij) {
  Tensor<double> lhs(Range{4, 2}, {7, 56, 88, 23, 11, 85, 39, 87});
  Tensor<double> rhs(Range{4, 2}, {75, 51, 29, 80, 82, 28, 76, 10});
  double corr = 14889;
  BipartiteIndexList oidx, lidx("i,j"), ridx("i,j");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ij_ji) {
  Tensor<double> lhs(Range{4, 2}, {7, 56, 88, 23, 11, 85, 39, 87});
  Tensor<double> rhs(Range{2, 4}, {12, 62, 44, 48, 37, 64, 76, 65});
  double corr = 23555;
  BipartiteIndexList oidx, lidx("i,j"), ridx("j,i");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_ijk) {
  Tensor<double> lhs(
      Range{4, 2, 6},
      {19, 12, 37, 28, 56, 34, 44, 17, 28, 85, 48, 99, 25, 13,  98, 83,
       47, 68, 26, 62, 20, 18, 71, 67, 23, 95, 46, 12, 67, 100, 74, 11,
       25, 42, 84, 37, 80, 76, 96, 84, 81, 77, 87, 44, 4,  12,  27, 63});
  Tensor<double> rhs(
      Range{4, 2, 6},
      {80, 100, 6,  47, 63, 45, 81, 27, 28, 44, 28, 15, 100, 27,  26, 8,
       43, 71,  4,  25, 79, 44, 83, 41, 63, 93, 54, 89, 24,  48,  58, 90,
       10, 89,  98, 60, 23, 85, 16, 44, 1,  35, 65, 89, 18,  100, 9,  68});
  double corr = 117914;
  BipartiteIndexList oidx, lidx("i,j,k"), ridx("i,j,k");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_ikj) {
  Tensor<double> lhs(
      Range{4, 2, 6},
      {19, 12, 37, 28, 56, 34, 44, 17, 28, 85, 48, 99, 25, 13,  98, 83,
       47, 68, 26, 62, 20, 18, 71, 67, 23, 95, 46, 12, 67, 100, 74, 11,
       25, 42, 84, 37, 80, 76, 96, 84, 81, 77, 87, 44, 4,  12,  27, 63});
  Tensor<double> rhs(
      Range{4, 6, 2},
      {63, 32, 82, 42, 54, 65, 100, 86, 27, 19, 47, 63, 19, 11, 91, 85,
       45, 78, 1,  30, 60, 50, 53,  80, 16, 16, 72, 22, 35, 95, 64, 82,
       12, 22, 34, 98, 65, 76, 17,  33, 42, 5,  87, 2,  94, 67, 75, 55});
  double corr = 128743;
  BipartiteIndexList oidx, lidx("i,j,k"), ridx("i,k,j");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_jik) {
  Tensor<double> lhs(
      Range{4, 2, 6},
      {19, 12, 37, 28, 56, 34, 44, 17, 28, 85, 48, 99, 25, 13,  98, 83,
       47, 68, 26, 62, 20, 18, 71, 67, 23, 95, 46, 12, 67, 100, 74, 11,
       25, 42, 84, 37, 80, 76, 96, 84, 81, 77, 87, 44, 4,  12,  27, 63});
  Tensor<double> rhs(
      Range{2, 4, 6},
      {88, 98, 61, 26, 98, 81, 21, 24, 13, 75, 87, 53, 17, 14, 53, 41,
       21, 94, 23, 81, 14, 85, 82, 59, 43, 63, 75, 51, 87, 56, 75, 44,
       38, 62, 10, 21, 54, 22, 85, 33, 13, 17, 98, 97, 58, 41, 33, 53});
  double corr = 128234;
  BipartiteIndexList oidx, lidx("i,j,k"), ridx("j,i,k");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_jki) {
  Tensor<double> lhs(
      Range{4, 2, 6},
      {19, 12, 37, 28, 56, 34, 44, 17, 28, 85, 48, 99, 25, 13,  98, 83,
       47, 68, 26, 62, 20, 18, 71, 67, 23, 95, 46, 12, 67, 100, 74, 11,
       25, 42, 84, 37, 80, 76, 96, 84, 81, 77, 87, 44, 4,  12,  27, 63});
  Tensor<double> rhs(
      Range{2, 6, 4},
      {75, 76, 21, 80,  29, 4,  44, 61, 61, 63, 50, 34, 25, 44, 3,  37,
       87, 37, 45, 25,  67, 99, 47, 59, 72, 74, 87, 59, 93, 67, 4,  18,
       84, 76, 57, 100, 26, 93, 12, 72, 17, 55, 58, 12, 25, 51, 58, 68});
  double corr = 127241;
  BipartiteIndexList oidx, lidx("i,j,k"), ridx("j,k,i");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_kij) {
  Tensor<double> lhs(
      Range{4, 2, 6},
      {19, 12, 37, 28, 56, 34, 44, 17, 28, 85, 48, 99, 25, 13,  98, 83,
       47, 68, 26, 62, 20, 18, 71, 67, 23, 95, 46, 12, 67, 100, 74, 11,
       25, 42, 84, 37, 80, 76, 96, 84, 81, 77, 87, 44, 4,  12,  27, 63});
  Tensor<double> rhs(
      Range{6, 4, 2},
      {71, 32, 85, 9,  21, 84, 94, 68, 98, 36, 75, 43, 71, 37, 55, 67,
       78, 89, 84, 77, 29, 28, 46, 25, 89, 40, 65, 28, 99, 65, 73, 50,
       23, 97, 94, 79, 86, 54, 35, 29, 18, 99, 62, 61, 54, 41, 45, 50});
  double corr = 151236;
  BipartiteIndexList oidx, lidx("i,j,k"), ridx("k,i,j");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_CASE(ijk_kji) {
  Tensor<double> lhs(
      Range{4, 2, 6},
      {19, 12, 37, 28, 56, 34, 44, 17, 28, 85, 48, 99, 25, 13,  98, 83,
       47, 68, 26, 62, 20, 18, 71, 67, 23, 95, 46, 12, 67, 100, 74, 11,
       25, 42, 84, 37, 80, 76, 96, 84, 81, 77, 87, 44, 4,  12,  27, 63});
  Tensor<double> rhs(
      Range{6, 2, 4},
      {99, 82, 42, 11, 30, 30, 73, 13, 4,  45, 83, 5,  45, 90, 15, 53,
       68, 3,  91, 33, 78, 51, 45, 5,  97, 53, 59, 43, 11, 58, 96, 97,
       71, 37, 62, 6,  83, 40, 86, 22, 7,  8,  32, 63, 15, 67, 92, 16});
  double corr = 109489;
  BipartiteIndexList oidx, lidx("i,j,k"), ridx("k,j,i");
  auto rv = kernels::s_t_t_contract_(oidx, lidx, ridx, lhs, rhs);
  BOOST_CHECK_EQUAL(rv, corr);
}

BOOST_AUTO_TEST_SUITE_END()
