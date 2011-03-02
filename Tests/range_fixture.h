#ifndef TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
#define TILEDARRAY_RANGE_FIXTURE_H__INCLUDED

#include "TiledArray/range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/array_util.h"
#include <iostream>

using namespace TiledArray;

struct RangeFixture {
  typedef Range<GlobalFixture::coordinate_system> RangeN;
  typedef RangeN::size_array size_array;
  typedef RangeN::index index;
  typedef RangeN::ordinal_index ordinal_index;
  typedef RangeN::volume_type volume_type;

  static const index start;
  static const index finish;
  static const size_array size;
  static const size_array weight;
  static const volume_type volume;
  static const index p0;
  static const index p1;
  static const index p2;
  static const index p3;
  static const index p4;
  static const index p5;
  static const index p6;

  RangeFixture();

  ~RangeFixture() { }

  RangeN r;
};

template <typename Index>
Index fill_index(typename Index::index value) {
  Index result;
  std::fill(result.begin(), result.end(), value);
  return result;
}

struct Range1Fixture {
  typedef TiledRange1<GlobalFixture::coordinate_system> range1_type;
  typedef range1_type::ordinal_index ordinal_index;
  typedef range1_type::tile_coordinate_system::index tile_index;

  static const std::array<std::size_t, 6> a;
  static const range1_type::range_type tiles;
  static const range1_type::tile_range_type elements;

  Range1Fixture() : tr1(a.begin(), a.end()) { }
  ~Range1Fixture() { }

  template <std::size_t D>
  static std::array<std::size_t, D> init_tiling() {
    std::array<std::size_t, D> result;
    result[0] = 0u;
    for(std::size_t i = 1; i < D; ++i)
      result[i] = result[i - 1] + GlobalFixture::primes[i - 1];
    return result;
  }

  range1_type tr1;
  std::array<range1_type::tile_range_type, 5> tile;
};

struct TiledRangeFixtureBase : public Range1Fixture {
  TiledRangeFixtureBase() : dims(range1_type::coordinate_system::dim, tr1) { }
  std::vector<range1_type> dims;
}; // struct TiledRangeFixtureBase

struct TiledRangeFixture : public RangeFixture, public TiledRangeFixtureBase {
  typedef TiledRange<GlobalFixture::coordinate_system> TRangeN;
  typedef TRangeN::tile_index tile_index;

  static const TRangeN::range_type tile_range;
  static const TRangeN::tile_range_type element_range;

  TiledRangeFixture() : tr(dims.begin(), dims.end()) {
  }

  ~TiledRangeFixture() { }

  static tile_index fill_tile_index(tile_index::index);

  TRangeN tr;
};

#endif // TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
