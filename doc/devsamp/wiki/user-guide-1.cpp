#include <tiledarray.h>

int main(int argc, char* argv[]) {
  // Initialize runtime
  madness::World& world = madness::initialize(argc, argv);

  // Do some work here.

  madness::finalize();
  return 0;
}