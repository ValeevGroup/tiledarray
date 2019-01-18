#include <tiledarray.h> // imports most TiledArray features

int main(int argc, char* argv[]) {
  // Initialize MADWorld
  madness::World& world = madness::initialize(argc, argv);

  // Do some work here.

  // Finalize MADWorld
  madness::finalize();
  return 0;
}