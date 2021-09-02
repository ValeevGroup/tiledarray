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
#include <cstdlib>
#include "TiledArray/external/madness.h"
#include "unit_test_config.h"

#include <TiledArray/error.h>
#if (TA_ASSERT_POLICY != TA_ASSERT_THROW)
#error "TiledArray unit tests require TA_ASSERT_POLICY=TA_ASSERT_THROW"
#endif

GlobalFixture::GlobalFixture() {
  if (world == nullptr) {
    world = &TiledArray::initialize(
        boost::unit_test::framework::master_test_suite().argc,
        boost::unit_test::framework::master_test_suite().argv);

    //  N.B. uncomment to create debugger:
    // using TiledArray::Debugger;
    // auto debugger = std::make_shared<Debugger>("ta_test");
    // Debugger::set_default_debugger(debugger);
    // debugger->set_prefix(world->rank());
    // choose lldb or gdb
    // debugger->set_cmd("lldb_xterm");
    // debugger->set_cmd("gdb_xterm");
    // to launch a debugger here or elsewhere:
    // Debugger::default_debugger()->debug("ready to run");
  }

  if (world->rank() != 0) {
    boost::unit_test::unit_test_log.set_threshold_level(
        boost::unit_test::log_all_errors);
  }
}

GlobalFixture::~GlobalFixture() {
  if (world) {
    world->gop.fence();
    TiledArray::finalize();
    world = nullptr;
  }
}

TiledArray::World* GlobalFixture::world = nullptr;
const std::array<std::size_t, 20> GlobalFixture::primes = {
    {2,  3,  5,  7,  11, 13, 17, 19, 23, 29,
     31, 37, 41, 43, 47, 53, 59, 61, 67, 71}};

bool GlobalFixture::is_distributed() {
  if (world)
    return world->size() > 1;
  else {
    auto envvar_cstr = std::getenv("TA_UT_DISTRIBUTED");
    return envvar_cstr;
  }
}

TiledArray::unit_test_enabler GlobalFixture::world_size_gt_1() {
  return TiledArray::unit_test_enabler(is_distributed());
}

TiledArray::unit_test_enabler GlobalFixture::world_size_eq_1() {
  return TiledArray::unit_test_enabler(!is_distributed());
}

// This line will initialize mpi and madness.
BOOST_GLOBAL_FIXTURE(GlobalFixture);
