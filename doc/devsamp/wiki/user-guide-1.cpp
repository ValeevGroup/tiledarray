#include <tiledarray.h> // imports most TiledArray features

int main(int argc, char* argv[]) {
  // Initialize TA
  auto& world = TA::initialize(argc, argv);

  // Do some work here.

  // Finalize TA
  TA::finalize();
  return 0;
}