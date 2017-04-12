/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2017  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  type_traits.cpp
 *  Apr 7, 2017
 *
 */

#include "unit_test_config.h"

#include "TiledArray/meta.h"

#include <cmath>

struct MetaFixture {
};  // MetaFixture

BOOST_FIXTURE_TEST_SUITE(meta_suite, MetaFixture)

using namespace TiledArray::meta;

double sin(double x) {
  return std::sin(x);
}
double cos(double x) {
  return std::cos(x);
}
madness::Future<double> async_cos(double x) {
  return TiledArray::get_default_world().taskq.add(cos, x);
}

BOOST_AUTO_TEST_CASE(sanity) {
  invoke(sin, invoke(cos, 2.0));
  invoke(sin, invoke(async_cos, 2.0));
}

BOOST_AUTO_TEST_SUITE_END()
