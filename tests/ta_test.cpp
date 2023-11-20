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

#ifdef TA_TENSOR_MEM_TRACE
#include <TiledArray/tensor.h>
#endif

#include <TiledArray/error.h>

GlobalFixture::GlobalFixture() {
  if (world == nullptr) {
    world = &TiledArray::initialize(
        boost::unit_test::framework::master_test_suite().argc,
        boost::unit_test::framework::master_test_suite().argv);
  }

  if (world->rank() != 0) {
    boost::unit_test::unit_test_log.set_threshold_level(
        boost::unit_test::log_all_errors);
  }

#ifdef TA_TENSOR_MEM_TRACE
  {
    TiledArray::Tensor<float>::trace_if_larger_than(1);
    //  TiledArray::Tensor<float>::ptr_registry()->log(&std::cout);
    TiledArray::Tensor<std::complex<float>>::trace_if_larger_than(1);
    //  TiledArray::Tensor<std::complex<float>>::ptr_registry()->log(&std::cout);
    TiledArray::Tensor<double>::trace_if_larger_than(1);
    //  TiledArray::Tensor<double>::ptr_registry()->log(&std::cout);
    TiledArray::Tensor<std::complex<double>>::trace_if_larger_than(1);
    //  TiledArray::Tensor<std::complex<double>>::ptr_registry()->log(&std::cout);
    TiledArray::Tensor<int>::trace_if_larger_than(1);
    //  TiledArray::Tensor<int>::ptr_registry()->log(&std::cout);
    TiledArray::Tensor<long>::trace_if_larger_than(1);
    //  TiledArray::Tensor<long>::trace_if_larger_than(1);
  }
#endif

  // uncomment to create or create+launch debugger
  // TiledArray::create_debugger("gdb_xterm", "ta_test");
  // TiledArray::create_debugger("lldb_xterm", "ta_test");
  // TiledArray::launch_gdb_xterm("ta_test");
  // TiledArray::launch_lldb_xterm("ta_test");
}

GlobalFixture::~GlobalFixture() {
  if (world) {
    world->gop.fence();
    TiledArray::finalize();
    world = nullptr;
  }
#ifdef TA_TENSOR_MEM_TRACE
  {
    TA_ASSERT(TiledArray::Tensor<float>::ptr_registry()->size() == 0);
    TA_ASSERT(TiledArray::Tensor<std::complex<float>>::ptr_registry()->size() ==
              0);
    TA_ASSERT(TiledArray::Tensor<double>::ptr_registry()->size() == 0);
    TA_ASSERT(
        TiledArray::Tensor<std::complex<double>>::ptr_registry()->size() == 0);
    TA_ASSERT(TiledArray::Tensor<int>::ptr_registry()->size() == 0);
    TA_ASSERT(TiledArray::Tensor<long>::ptr_registry()->size() == 0);
  }
#endif
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
