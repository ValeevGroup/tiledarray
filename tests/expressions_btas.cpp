/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions_sparse.cpp
 *  May 4, 2018
 *
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_BTAS
#include "expressions_fixture.h"

typedef ExpressionsFixture<
    TA::Tile<btas::Tensor<int, TiledArray::Range, btas::varray<int>>>,
    TA::DensePolicy>
    EF_TABtasI;

typedef boost::mpl::vector<EF_TABtasI> Fixtures;

BOOST_AUTO_TEST_SUITE(expressions_btas_suite)

#include "expressions_impl.h"

#endif
