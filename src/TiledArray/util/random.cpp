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
 */

#include <TiledArray/util/random.h>

#include <TiledArray/util/thread_specific.h>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

namespace TiledArray {

namespace detail {

auto& random_engine_pool_accessor() {
  using engine_t = boost::random::mt19937;
  static auto tspool = make_tspool(engine_t(1));
  return tspool;
}
}  // namespace detail

boost::random::mt19937& random_engine() {
  return detail::random_engine_pool_accessor().local();
}

int rand() {
  using dist_t = boost::random::uniform_int_distribution<>;
  static dist_t dist(0, RAND_MAX);

  return dist(random_engine());
}

double drand() {
  using dist_t = boost::random::uniform_real_distribution<double>;
  static dist_t dist(0, 1);

  return dist(random_engine());
}

void srand(unsigned int seed) {
  using engine_t = boost::random::mt19937;
  detail::random_engine_pool_accessor() = detail::make_tspool(engine_t(seed));
}

}  // namespace TiledArray
