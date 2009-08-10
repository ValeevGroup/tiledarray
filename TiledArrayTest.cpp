//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN Tiled Array Tests
#include <boost/test/included/unit_test.hpp>
#include <madness_runtime.h>
#include <world/world.h>

struct MpiFixture {
  MpiFixture() {
    madness::initialize(boost::unit_test::framework::master_test_suite().argc,
        boost::unit_test::framework::master_test_suite().argv);
  }

  ~MpiFixture() {
    madness::finalize();
  }
};

struct MadnessFixture : public MpiFixture {
  MadnessFixture() : world(MPI::COMM_WORLD) {
    world.args(boost::unit_test::framework::master_test_suite().argc,
        boost::unit_test::framework::master_test_suite().argv);

    world.gop.fence();
  }

  ~MadnessFixture() {
    world.gop.fence();
  }

  madness::World world;
};

BOOST_GLOBAL_FIXTURE( MadnessFixture );

/*
int main(int argc, char* argv[]) {

  // start up MPI and MADNESS
  MPI::Init(argc, argv);
  ThreadPool::begin();
  RMI::begin();
  madness::World world(MPI::COMM_WORLD);
  redirectio(world);
  world.gop.fence();



  world.gop.fence();
  RMI::end();
  MPI::Finalize();

  return 0;
}
*/
