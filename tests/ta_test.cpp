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

#define BOOST_TEST_MAIN TiledArray Tests
#include "unit_test_config.h"
#include <TiledArray/runtime.h>
#include <world/world.h>

GlobalFixture::GlobalFixture() :
  ta_runtime(boost::unit_test::framework::master_test_suite().argc,
      boost::unit_test::framework::master_test_suite().argv)
{
  world = & ta_runtime.get_world();
  world->gop.fence();
}

GlobalFixture::~GlobalFixture() { }

madness::World* GlobalFixture::world = NULL;
const std::array<std::size_t, 20> GlobalFixture::primes =
    {{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71 }};


// This line will initialize mpi and madness.
BOOST_GLOBAL_FIXTURE( GlobalFixture )
