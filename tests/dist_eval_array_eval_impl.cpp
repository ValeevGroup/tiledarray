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
 *  dist_eval_array_eval.cpp
 *  Sep 15, 2013
 *
 */

#include "TiledArray/dist_eval/array_eval.h"
#include "unit_test_config.h"
#include "array_fixture.h"
#include "TiledArray/tile_op/scal.h"
#include "TiledArray/shape.h"


using namespace TiledArray;



struct ArrayEvalImplFixture : public ArrayFixture {
  typedef math::Scal<ArrayN::value_type::eval_type,
      ArrayN::value_type::eval_type, false> op_type;
  typedef detail::ArrayEvalImpl<ArrayN, op_type, DensePolicy> impl_type;
  ArrayEvalImplFixture() : op(3) { }

  ~ArrayEvalImplFixture() { }

   op_type op;
}; // ArrayEvalFixture

BOOST_FIXTURE_TEST_SUITE( array_eval_suite, ArrayEvalImplFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(impl_type(a, Permutation(), DenseShape(), a.get_pmap(), op));
}

BOOST_AUTO_TEST_SUITE_END()
