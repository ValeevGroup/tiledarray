#ifndef TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED

namespace madness {
  class World;
} // namespace madness

struct GlobalFixture {
  GlobalFixture();
  ~GlobalFixture();

  static madness::World* world;
  static unsigned int count;
};

#endif // TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
