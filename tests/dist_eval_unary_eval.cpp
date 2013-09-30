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
 *  unary_eval.cpp
 *  August 8, 2013
 *
 */

#include "TiledArray/dist_eval/unary_eval.h"
#include "TiledArray/array.h"
#include "TiledArray/dense_shape.h"
#include "unit_test_config.h"
#include "array_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct TestPolicy {
  typedef std::size_t size_type;
  typedef TiledRange trange_type;
  typedef trange_type::range_type range_type;
  typedef DenseShape shape_type;
  typedef Tensor<int> tile_type;
  typedef madness::Future<tile_type> future;

};

struct UnaryEvalFixture {
  UnaryEvalFixture() { }

}; // struct UnaryEvalFixture

BOOST_FIXTURE_TEST_SUITE( unary_eval_suite, UnaryEvalFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
}



BOOST_AUTO_TEST_SUITE_END()
