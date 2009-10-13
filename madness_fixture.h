#ifndef TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED

#include <world/world.h>

struct MadnessFixture {
  MadnessFixture();
  ~MadnessFixture();

  static madness::World* world;
};

#endif // TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
