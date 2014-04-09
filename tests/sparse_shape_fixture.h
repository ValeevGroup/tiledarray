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
      SparseShape<float>::threshold(0.001);
    }

    ~SparseShapeFixture() { }


    static Tensor<float> make_norm_tensor(const TiledRange& trange, const int seed) {
      GlobalFixture::world->srand(seed);
      Tensor<float> norms(trange.tiles());
      for(Tensor<float>::size_type i = 0ul; i < norms.size(); ++i) {
        const Range range = trange.make_tile_range(i);
        norms[i] = (GlobalFixture::world->rand() % 101);
        norms[i] = std::sqrt(norms[i] * norms[i] * range.volume());
      }

      return norms;
    }

    static SparseShape<float> make_shape(const TiledRange& trange, const float fill_percent, const int seed) {
      Tensor<float> shape_data = make_norm_tensor(trange, seed);

      const std::size_t n = float(shape_data.size()) * (1.0 - fill_percent);
      for(std::size_t i = 0ul; i < n; ++i) {
        shape_data[GlobalFixture::world->rand() % shape_data.size()] = SparseShape<float>::threshold() * 0.1;
      }

      return SparseShape<float>(shape_data, trange);
    }

    SparseShape<float> sparse_shape;
    SparseShape<float> left;
    SparseShape<float> right;

  }; // SparseShapeFixture

} // namespace TiledArray

#endif // TILEDARRAY_TEST_SPARSE_SHAPE_FIXTURE_H__INCLUDED
