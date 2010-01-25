//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#define BOOST_TEST_MAIN TiledArray Tests
#include "unit_test_config.h"

GlobalFixture::GlobalFixture() {
  madness::initialize(boost::unit_test::framework::master_test_suite().argc,
      boost::unit_test::framework::master_test_suite().argv);

  if(count == 0) {
    world = new madness::World(MPI::COMM_WORLD);
    world->args(boost::unit_test::framework::master_test_suite().argc,
        boost::unit_test::framework::master_test_suite().argv);
  }

  ++count;
  world->gop.fence();
}

GlobalFixture::~GlobalFixture() {
  world->gop.fence();

  --count;
  if(count == 0) {
    delete world;
    world = NULL;
  }
  madness::finalize();
}

madness::World* GlobalFixture::world = NULL;
unsigned int GlobalFixture::count = 0;


// This line will initialize mpi and madness.
BOOST_GLOBAL_FIXTURE( GlobalFixture )
