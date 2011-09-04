#ifndef TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED

#include "TiledArray/array.h"
#include "range_fixture.h"
#include <vector>
#include "unit_test_config.h"

struct ArrayFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::coordinate_system> ArrayN;
  typedef ArrayN::index index;
  typedef ArrayN::ordinal_index ordinal_index;
  typedef ArrayN::value_type tile_type;

  ArrayFixture();


  std::vector<std::size_t> list;
  madness::World& world;
  ArrayN a;
}; // struct ArrayFixture

#endif // TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED
