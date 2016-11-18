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

#include "TiledArray/shape/sparse_shape.h"
#include "range_fixture.h"

namespace TiledArray {


  struct SparseShapeFixture : public TiledRangeFixture{
    typedef std::vector<std::size_t> vec_type;

    SparseShapeFixture() :
      sparse_shape(make_shape(tr, 0.5, 42)),
      left(make_shape(tr, 0.1, 23)),
      right(make_shape(tr, 0.1, 82)),
      perm(make_perm()),
      perm_index(tr.tiles_range(), perm),
      tolerance(0.0001)

    {
      SparseShape<float>::threshold(0.001);
    }

    ~SparseShapeFixture() { }

    template <typename Real = float>
    static Tensor<Real> make_norm_tensor(const TiledRange& trange, const float fill_fraction, const int seed) {
      assert(fill_fraction <= 1.0 && fill_fraction >= 0.0);
      GlobalFixture::world->srand(seed);
      Tensor<Real> norms(trange.tiles());
      for(typename Tensor<Real>::size_type i = 0ul; i < norms.size(); ++i) {
        const Range range = trange.make_tile_range(i);
        // make sure nonzero tile norms are MUCH greater than threshold since SparseShape scales norms by 1/volume
        norms[i] = SparseShape<Real>::threshold() * (100 + (GlobalFixture::world->rand() % 1000));
      }

      const std::size_t target_num_zeroes = std::min(norms.size(), static_cast<std::size_t>(double(norms.size()) * (1.0 - fill_fraction)));
      std::size_t num_zeroes = 0ul;
      while (num_zeroes != target_num_zeroes) {
        const Real zero_norm = SparseShape<Real>::threshold() / 2;
        const size_t rand_idx = GlobalFixture::world->rand() % norms.size();
        if (norms[rand_idx] != zero_norm) {
          norms[rand_idx] = zero_norm;
          ++num_zeroes;
        }
      }
      return norms;

    }

    template <typename Real = float>
    static SparseShape<Real> make_shape(const TiledRange& trange, const float fill_percent, const int seed) {
      Tensor<Real> tile_norms = make_norm_tensor<Real>(trange, fill_percent, seed);
      auto result = SparseShape<Real>(tile_norms, trange);
      return result;
    }

    static Permutation make_perm() {
      std::array<unsigned int, GlobalFixture::dim> temp;
      for(std::size_t i = 0; i < temp.size(); ++i)
        temp[i] = i + 1;

      temp.back() = 0;

      return Permutation(temp.begin(), temp.end());
    }

    SparseShape<float> sparse_shape;
    SparseShape<float> left;
    SparseShape<float> right;
    Permutation perm;
    TiledArray::detail::PermIndex perm_index;
    const float tolerance;
  }; // SparseShapeFixture

} // namespace TiledArray

#endif // TILEDARRAY_TEST_SPARSE_SHAPE_FIXTURE_H__INCLUDED
