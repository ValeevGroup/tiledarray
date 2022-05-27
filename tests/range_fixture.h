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
 */

#ifndef TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
#define TILEDARRAY_RANGE_FIXTURE_H__INCLUDED

#include <iostream>
#include "TiledArray/range.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"
#include "global_fixture.h"

using namespace TiledArray;

struct RangeFixture {
  typedef Range::index_view_type index_view_type;
  typedef Range::index index;
  typedef Range::size_type size_type;

  static const index start;
  static const index finish;
  static const std::vector<std::size_t> size;
  static const std::vector<std::size_t> weight;
  static const size_type volume;
  static const index p0;
  static const index p1;
  static const index p2;
  static const index p3;
  static const index p4;
  static const index p5;
  static const index p6;

  RangeFixture() : r(start, finish) {}

  ~RangeFixture() {}

  template <typename A>
  static std::vector<std::size_t> calc_weight(const A& size, unsigned int n) {
    std::vector<std::size_t> weight(n);
    std::size_t volume = 1ul;
    for (int i = int(n) - 1; i >= 0; --i) {
      weight[i] = volume;
      volume *= size[i];
    }
    return weight;
  }

  Range r;
};

struct Range1Fixture {
  static const size_t ntiles = 5;

  Range1Fixture()
      : a(init_tiling<ntiles + 1>()),
        tiles(0, a.size() - 1),
        elements(a.front(), a.back()),
        tr1(a.begin(), a.end()) {}
  ~Range1Fixture() {}

  template <std::size_t D>
  static std::array<std::size_t, D> init_tiling() {
    std::array<std::size_t, D> result;
    result[0] = 0u;
    for (std::size_t i = 1; i < D; ++i)
      result[i] = result[i - 1] + GlobalFixture::primes[i - 1];
    return result;
  }

  const std::array<std::size_t, ntiles + 1> a;
  const TiledRange1::range_type tiles;
  const TiledRange1::range_type elements;
  TiledRange1 tr1;
  std::array<TiledRange1::range_type, ntiles> tile;
};

struct TiledRangeFixtureBase : public Range1Fixture {
  TiledRangeFixtureBase() {
    std::fill(dims.begin(), dims.end(), tr1);
    std::fill(extents.begin(), extents.end(), tr1.extent());
  }
  std::array<TiledRange1, GlobalFixture::dim> dims;
  std::array<long, GlobalFixture::dim> extents;
};  // struct TiledRangeFixtureBase

struct TiledRangeFixture : public RangeFixture, public TiledRangeFixtureBase {
  typedef TiledRange TRangeN;
  typedef TRangeN::range_type::index tile_index;

  TiledRangeFixture()
      : tiles_range(TiledRangeFixture::index(GlobalFixture::dim, 0),
                    TiledRangeFixture::index(GlobalFixture::dim, 5)),
        elements_range(TiledRangeFixture::tile_index(GlobalFixture::dim, 0),
                       TiledRangeFixture::tile_index(GlobalFixture::dim, a[5])),
        tr(dims.begin(), dims.end()) {}

  ~TiledRangeFixture() {}

  static tile_index fill_tile_index(TRangeN::range_type::index::value_type);

  const TRangeN::range_type tiles_range;
  const TRangeN::range_type elements_range;
  TRangeN tr;
};

#endif  // TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
