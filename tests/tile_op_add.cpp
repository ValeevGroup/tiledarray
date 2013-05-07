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
 *  justus
 *  Department of Chemistry
 *  Virginia Tech
 *  Blacksburg, VA 24061
 *
 *  tile_op_add.cpp
 *  May 7, 2013
 *
 */

#include "TiledArray/tile_op/add.h"
#include "unit_test_config.h"
#include "TiledArray/tensor.h"
#include "range_fixture.h"

struct AddFixture : public RangeFixture {

  AddFixture() :
    a(RangeFixture::r, 5),
    b(RangeFixture::r, 7),
    c(RangeFixture::r)
  { }

  ~AddFixture() { }

  Tensor<int> a;
  Tensor<int> b;
  Tensor<int> c;

}; // AddFixture

BOOST_FIXTURE_TEST_SUITE( tile_op_add_suite, AddFixture )

BOOST_AUTO_TEST_CASE( constructor )
{

}

BOOST_AUTO_TEST_SUITE_END()
