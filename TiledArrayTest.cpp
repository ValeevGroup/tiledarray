//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#define TEST_COORDINATES
//#define TEST_PERMUTATION
//#define TEST_RANGE1
//#define TEST_RANGE
//#define TEST_SHAPE
//#define TEST_TILEMAP
//#define TEST_TILE
#define TEST_ARRAY

#include "coordinatestest.h"
#include "permutationtest.h"
#include "range1test.h"
#include "rangetest.h"
#include "shapetest.h"
#include "tilemaptest.h"
#include "tiletest.h"
#include "arraytest.h"
#include <madness_runtime.h>
#include <world/world.h>

namespace TiledArray { }
using namespace TiledArray;
using namespace madness;

int main(int argc, char* argv[]) {

  // start up MPI and MADNESS
  MPI::Init(argc, argv);
  ThreadPool::begin();
  RMI::begin();
  madness::World world(MPI::COMM_WORLD);
  redirectio(world);
  world.gop.fence();

  RUN_COORDINATES_TEST
  RUN_PERMUTATION_TEST
  RUN_RANGE1_TEST
  RUN_RANGE_TEST
  RUN_SHAPE_TEST
  RUN_TILEMAP_TEST
  RUN_TILE_TEST
  RUN_ARRAY_TEST

  world.gop.fence();
  RMI::end();
  MPI::Finalize();

  return 0;
}
