#ifndef TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED

#include "TiledArray/package.h"
#include <world/world.h>
#include "TiledArray/package.h"

struct MadnessFixture {
  MadnessFixture();
  ~MadnessFixture();

  static madness::World* world;
  static unsigned int count;
};

#endif // TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
