/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  sparse_shape_fixture.h
 *  Feb 23, 2014
 *
 */

#ifndef TILEDARRAY_TEST_SPARSE_SHAPE_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_SPARSE_SHAPE_FIXTURE_H__INCLUDED

#include "TiledArray/sparse_shape.h"
#include "range_fixture.h"

namespace TiledArray {


  struct SparseShapeFixture : public TiledRangeFixture{
    typedef std::vector<std::size_t> vec_type;

    SparseShapeFixture() :
      sparse_shape(make_shape(tr, 0.5, 23)),
      left(make_shape(tr, 0.1, 23)),
      right(make_shape(tr, 0.1, 23))
    {
      SparseShape<float>::threshold(0.5);
    }

    ~SparseShapeFixture() { }

    static SparseShape<float> make_shape(const TiledRange& trange, const float fill_percent, const int seed) {
      GlobalFixture::world->srand(seed);
      float max = 0.0f;
      Tensor<float> shape_data(trange.tiles());
      for(std::size_t i = 0ul; i < shape_data.size(); ++i) {
        shape_data[i] = GlobalFixture::world->rand();
        max = std::max(max, shape_data[i]);
      }

      shape_data *= 27.0f / max;

      const std::size_t n = shape_data.size() * ((1.0 - fill_percent) - (1.0 / (2.0 * 27.0)));
      for(std::size_t i = 0ul; i < n; ++i) {
        shape_data[GlobalFixture::world->rand() % shape_data.size()] = 0.1;
      }

      return SparseShape<float>(shape_data, trange);
    }

    SparseShape<float> sparse_shape;
    SparseShape<float> left;
    SparseShape<float> right;

  }; // SparseShapeFixture

} // namespace TiledArray

#endif // TILEDARRAY_TEST_SPARSE_SHAPE_FIXTURE_H__INCLUDED
