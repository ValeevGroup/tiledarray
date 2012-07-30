#ifndef TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED

#include "TiledArray/coordinate_system.h"
#include <world/stdarray.h>

namespace madness {
  class World;
} // namespace madness

#ifndef TEST_DIM
#define TEST_DIM 3U
#endif
#if TEST_DIM > 20
#error "TEST_DIM cannot be greater than 20"
#endif

struct GlobalFixture {
  GlobalFixture();
  ~GlobalFixture();

  typedef TiledArray::CoordinateSystem<TEST_DIM, 1U> coordinate_system;
  typedef TiledArray::CoordinateSystem<TEST_DIM, 0U> element_coordinate_system;

  static madness::World* world;
  static unsigned int count;
  static const std::array<std::size_t, 20> primes;
};

#endif // TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
