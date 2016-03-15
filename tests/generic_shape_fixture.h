/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  Ed Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  generic_shape_fixture.h
 *  Oct 26, 2015
 *
 */

#ifndef TILEDARRAY_TEST_GENERIC_SHAPE_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_GENERIC_SHAPE_FIXTURE_H__INCLUDED

#include "TiledArray/shape/generic_shape.h"
#include "sparse_shape_fixture.h"

namespace TiledArray {


  struct GenericShapeFixture : public SparseShapeFixture {

    GenericShapeFixture() : SparseShapeFixture()
    {
    }

    ~GenericShapeFixture() { }

  }; // GenericShapeFixture

} // namespace TiledArray

#endif // TILEDARRAY_TEST_GENERIC_SHAPE_FIXTURE_H__INCLUDED
