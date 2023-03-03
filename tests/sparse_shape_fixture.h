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

struct SparseShapeFixture : public TiledRangeFixture {
  typedef std::vector<std::size_t> vec_type;

  SparseShapeFixture()
      : sparse_shape(make_shape(tr, 0.5, 42)),
        left(make_shape(tr, 0.1, 23)),
        right(make_shape(tr, 0.1, 82)),
        perm(make_perm()),
        perm_index(tr.tiles_range(), perm),
        tolerance(0.0001)

  {
    SparseShape<float>::threshold(default_threshold);
  }

  ~SparseShapeFixture() {}

  static Tensor<float> make_norm_tensor(const TiledRange& trange,
                                        const float fill_percent,
                                        const int seed) {
    GlobalFixture::world->srand(seed);
    Tensor<float> norms(trange.tiles_range());
    for (Tensor<float>::size_type i = 0ul; i < norms.size(); ++i) {
      const Range range = trange.make_tile_range(i);
      norms[i] = (GlobalFixture::world->rand() % 101);
      norms[i] = std::sqrt(norms[i] * norms[i] * range.volume());
    }

    const std::size_t n = float(norms.size()) * (1.0 - fill_percent);
    for (std::size_t i = 0ul; i < n; ++i) {
      norms[GlobalFixture::world->rand() % norms.size()] =
          SparseShape<float>::threshold() * 0.1;
    }

    return norms;
  }

  static SparseShape<float> make_shape(const TiledRange& trange,
                                       const float fill_percent,
                                       const int seed) {
    Tensor<float> tile_norms = make_norm_tensor(trange, fill_percent, seed);
    return SparseShape<float>(tile_norms, trange);
  }

  static Permutation make_perm() {
    std::array<unsigned int, GlobalFixture::dim> temp;
    for (std::size_t i = 0; i < temp.size(); ++i) temp[i] = i + 1;

    temp.back() = 0;

    return Permutation(temp.begin(), temp.end());
  }

  static void reset_threshold() {
    SparseShape<float>::threshold(default_threshold);
  }

  static auto set_threshold_to_max() {
    SparseShape<float>::threshold(std::numeric_limits<float>::max());
    return std::shared_ptr<void>(nullptr, [](void*) { reset_threshold(); });
  }

  static auto tweak_threshold() {
    SparseShape<float>::threshold(10 * SparseShape<float>::threshold());
    return std::shared_ptr<void>(nullptr, [](void*) { reset_threshold(); });
  }

  SparseShape<float> sparse_shape;
  SparseShape<float> left;
  SparseShape<float> right;
  Permutation perm;
  TiledArray::detail::PermIndex perm_index;
  const float tolerance;
  static constexpr float default_threshold = 0.001;
};  // SparseShapeFixture

}  // namespace TiledArray

#endif  // TILEDARRAY_TEST_SPARSE_SHAPE_FIXTURE_H__INCLUDED
