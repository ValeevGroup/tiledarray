/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  dist_op_dist_cache.cpp
 *  Oct 14, 2013
 *
 */

#include "TiledArray/madness.h"
#include "unit_test_config.h"

using namespace madness::detail;

struct DistCacheFixture {

  DistCacheFixture() { }

  ~DistCacheFixture() { }

}; // DistCacheFixture

BOOST_FIXTURE_TEST_SUITE( dist_op_dist_cache_suite, DistCacheFixture )

BOOST_AUTO_TEST_CASE( set_and_get_cache )
{
  const int key = 1;
  const int value = 42;

  madness::Future<int> data = madness::Future<int>::default_initializer();
  BOOST_CHECK_NO_THROW(data = DistCache<int>::template get_cache_value<int>(key));
  BOOST_CHECK(! data.probe());
  BOOST_CHECK_NO_THROW(DistCache<int>::template set_cache_value<int>(key, value));

  BOOST_CHECK(data.probe());
  BOOST_CHECK_EQUAL(data.get(), value);
}

BOOST_AUTO_TEST_CASE( set_and_get_cache_v2 )
{
  const int key = 1;
  const madness::Future<int> value(42);

  madness::Future<int> data;
  BOOST_CHECK_NO_THROW(DistCache<int>::get_cache_value(key, data));
  BOOST_CHECK(! data.probe());
  BOOST_CHECK_NO_THROW(DistCache<int>::template set_cache_value<int>(key, value));

  BOOST_CHECK(data.probe());
  BOOST_CHECK_EQUAL(data.get(), value.get());
}

BOOST_AUTO_TEST_SUITE_END()
