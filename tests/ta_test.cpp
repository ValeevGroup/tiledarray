/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#define BOOST_TEST_MODULE TiledArray Tests
#include <TiledArray/initialize.h>
#include "TiledArray/external/madness.h"
#include "unit_test_config.h"

int wait_for_debugger = 0;

GlobalFixture::GlobalFixture() {
  world = &TiledArray::initialize(
      boost::unit_test::framework::master_test_suite().argc,
      boost::unit_test::framework::master_test_suite().argv);
  world->gop.fence();
  if (wait_for_debugger) {
    if (world->rank() == 0)
      std::cerr << "waiting for debugger to attach ... then set "
                   "wait_for_debugger to 0 to proceed"
                << std::endl;
    while (wait_for_debugger) {
    }
  }
}

GlobalFixture::~GlobalFixture() {
  world->gop.fence();
  TiledArray::finalize();
}

TiledArray::World* GlobalFixture::world = NULL;
const std::array<std::size_t, 20> GlobalFixture::primes = {
    {2,  3,  5,  7,  11, 13, 17, 19, 23, 29,
     31, 37, 41, 43, 47, 53, 59, 61, 67, 71}};

// This line will initialize mpi and madness.
BOOST_GLOBAL_FIXTURE(GlobalFixture);
