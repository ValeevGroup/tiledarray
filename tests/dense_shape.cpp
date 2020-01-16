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
 *  dense_shape.cpp
 *  Jul 18, 2013
 *
 */

#include "TiledArray/dense_shape.h"
#include "unit_test_config.h"

struct DenseShapeFixture {
  DenseShapeFixture() {}

  ~DenseShapeFixture() {}

};  // DenseShapeFixture

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE(dense_shape_suite, DenseShapeFixture)

BOOST_AUTO_TEST_CASE(constructor) { BOOST_CHECK_NO_THROW(DenseShape()); }

BOOST_AUTO_TEST_SUITE_END()
