#ifndef TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
#define TILEDARRAY_RANGE_FIXTURE_H__INCLUDED

#include "global_fixture.h"
#include "TiledArray/range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/tiled_range.h"
#include <iostream>

using namespace TiledArray;

struct RangeFixture {
  typedef StaticRange<GlobalFixture::coordinate_system> RangeN;
  typedef RangeN::size_array size_array;
  typedef RangeN::index index;
  typedef RangeN::size_type size_type;

  static const index start;
  static const index finish;
  static const size_array size;
  static const size_array weight;
  static const size_type volume;
  static const index p0;
  static const index p1;
  static const index p2;
  static const index p3;
  static const index p4;
  static const index p5;
  static const index p6;

  RangeFixture() { }

  ~RangeFixture() { }

  static size_array calc_weight(const size_array& size) {
    size_array weight;
    TiledArray::detail::calc_weight(weight, size, GlobalFixture::coordinate_system::order);
    return weight;
  }

  static std::vector<std::size_t> calc_weight(const std::vector<std::size_t>& size) {
    std::vector<std::size_t> weight(size.size());
    TiledArray::detail::calc_weight(weight, size, GlobalFixture::coordinate_system::order);
    return weight;
  }

};

struct StaticRangeFixture : public RangeFixture {
  typedef StaticRange<GlobalFixture::coordinate_system> StaticRangeN;
  typedef StaticRangeN::size_array size_array;
  typedef StaticRangeN::index index;
  typedef StaticRangeN::size_type size_type;

  StaticRangeFixture();

  StaticRangeN r;
};


struct DynamicRangeFixture : public RangeFixture {
  typedef DynamicRange::size_array size_array;
  typedef DynamicRange::size_type size_type;

  DynamicRangeFixture();

  DynamicRange r;
};

struct Range1Fixture {

  static const std::array<std::size_t, 6> a;
  static const TiledRange1::range_type tiles;
  static const TiledRange1::range_type elements;

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

  TiledRange1 tr1;
  std::array<TiledRange1::range_type, 5> tile;
};

struct TiledRangeFixtureBase : public Range1Fixture {
  TiledRangeFixtureBase() : dims(GlobalFixture::coordinate_system::dim, tr1) { }
  std::vector<TiledRange1> dims;
}; // struct TiledRangeFixtureBase

struct TiledRangeFixture : public StaticRangeFixture, public TiledRangeFixtureBase {
  typedef StaticTiledRange<GlobalFixture::coordinate_system> TRangeN;
  typedef TRangeN::tile_range_type::index tile_index;

  static const TRangeN::range_type tile_range;
  static const TRangeN::tile_range_type element_range;

  TiledRangeFixture() : tr(dims.begin(), dims.end()) {
  }

  ~TiledRangeFixture() { }

  static tile_index fill_tile_index(TRangeN::tile_range_type::index::value_type);

  TRangeN tr;
};

#endif // TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
