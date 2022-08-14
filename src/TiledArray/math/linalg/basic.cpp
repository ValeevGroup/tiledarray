/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2022  Virginia Tech
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
 *  util.h
 *  Aug 8, 2022
 *
 */

#include "TiledArray/external/madness.h"
#include "TiledArray/math/linalg/forward.h"

namespace TiledArray::math::linalg {

namespace detail {

LinearAlgebraBackend& linalg_backend_accessor() {
  static LinearAlgebraBackend backend = LinearAlgebraBackend::BestAvailable;
  return backend;
}

std::size_t& linalg_crossover_to_distributed_accessor() {
  static std::size_t crossover_to_distributed = 2048 * 2048;
  return crossover_to_distributed;
}

}  // namespace detail

/// @return the linear algebra backend to use for matrix algebra
/// @note the default is LinearAlgebraBackend::BestAvailable
LinearAlgebraBackend get_linalg_backend() {
  return detail::linalg_backend_accessor();
}

/// @param[in] b the linear algebra backend to use after this
/// @note this is a collective call over the default world
void set_linalg_backend(LinearAlgebraBackend b) {
  get_default_world().gop.fence();
  detail::linalg_backend_accessor() = b;
}

/// @return the crossover-to-distributed threshold specifies the matrix volume
/// for which to switch to the distributed-memory backend (if available) for
/// linear algebra solvers
std::size_t get_linalg_crossover_to_distributed() {
  return detail::linalg_crossover_to_distributed_accessor();
}

/// @param[in] c the crossover-to-distributed threshold to use following this
/// call
/// @note this is a collective call over the default world
void set_linalg_crossover_to_distributed(std::size_t c) {
  get_default_world().gop.fence();
  detail::linalg_crossover_to_distributed_accessor() = c;
}

}  // namespace TiledArray::math::linalg
