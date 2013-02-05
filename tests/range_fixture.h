#ifndef TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
#define TILEDARRAY_RANGE_FIXTURE_H__INCLUDED

#include "global_fixture.h"
#include "TiledArray/range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/tiled_range.h"
#include <iostream>

using namespace TiledArray;

struct RangeFixture {
  typedef Range::size_array size_array;
  typedef Range::index index;
  typedef Range::size_type size_type;

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

  RangeFixture() : r(start, finish) { }

  ~RangeFixture() { }

  static size_array calc_weight(const size_array& size) {
    size_array weight(size.size());
    TiledArray::detail::calc_weight(weight, size);
    return weight;
  }

  Range r;
};


struct Range1Fixture {


  Range1Fixture() :
      a(init_tiling<6>()),
      tiles(0, a.size() - 1),
      elements(a.front(), a.back()),
      tr1(a.begin(), a.end())
  { }
  ~Range1Fixture() { }

  template <std::size_t D>
  static std::array<std::size_t, D> init_tiling() {
    std::array<std::size_t, D> result;
    result[0] = 0u;
    for(std::size_t i = 1; i < D; ++i)
      result[i] = result[i - 1] + GlobalFixture::primes[i - 1];
    return result;
  }

  const std::array<std::size_t, 6> a;
  const TiledRange1::range_type tiles;
  const TiledRange1::range_type elements;
  TiledRange1 tr1;
  std::array<TiledRange1::range_type, 5> tile;
};

struct TiledRangeFixtureBase : public Range1Fixture {
  TiledRangeFixtureBase() : dims(GlobalFixture::dim, tr1) { }
  std::vector<TiledRange1> dims;
}; // struct TiledRangeFixtureBase

struct TiledRangeFixture : public RangeFixture, public TiledRangeFixtureBase {
  typedef TiledRange TRangeN;
  typedef TRangeN::tile_range_type::index tile_index;


  TiledRangeFixture() :
    tile_range(TiledRangeFixture::index(GlobalFixture::dim, 0),
        TiledRangeFixture::index(GlobalFixture::dim, 5)),
    element_range(TiledRangeFixture::tile_index(GlobalFixture::dim, 0),
        TiledRangeFixture::tile_index(GlobalFixture::dim, a[5])),
    tr(dims.begin(), dims.end())
  { }

  ~TiledRangeFixture() { }

  static tile_index fill_tile_index(TRangeN::tile_range_type::index::value_type);

  const TRangeN::range_type tile_range;
  const TRangeN::tile_range_type element_range;
  TRangeN tr;
};

#endif // TILEDARRAY_RANGE_FIXTURE_H__INCLUDED
