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

#ifndef TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED

#include <array>
#include <functional>

#ifndef TILEDARRAY_UNIT_TEST_CONFIG_H__INCLUDED
#include <unit_test_config.h>
#endif

namespace madness {
class World;
}  // namespace madness

#ifndef TEST_DIM
#define TEST_DIM 3u
#endif
#if TEST_DIM > 20
#error "TEST_DIM cannot be greater than 20"
#endif

namespace TiledArray {
struct unit_test_enabler {
  using enabler_t = std::function<boost::test_tools::assertion_result(
      boost::unit_test::test_unit_id)>;
  enabler_t enabler;

  unit_test_enabler(enabler_t e) : enabler(e) {}
  unit_test_enabler(bool tf)
      : enabler([result = tf](boost::unit_test::test_unit_id) {
          return static_cast<boost::test_tools::assertion_result>(result);
        }) {}

  boost::test_tools::assertion_result operator()(
      boost::unit_test::test_unit_id id) {
    return enabler(id);
  }
};
}  // namespace TiledArray

struct GlobalFixture {
  GlobalFixture();
  ~GlobalFixture();

  static const unsigned int dim = TEST_DIM;

  static madness::World* world;
  static const std::array<std::size_t, 20> primes;

  // returns world.size > 1 if world is initialized, else return true if envvar
  // TA_UT_DISTRIBUTED is set
  static bool is_distributed();
  static TiledArray::unit_test_enabler world_size_gt_1();
  static TiledArray::unit_test_enabler world_size_eq_1();
};

#include <TiledArray/util/bug.h>

#endif  // TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
