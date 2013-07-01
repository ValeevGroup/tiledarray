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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  tiled_array.cpp
 *  Jul 1, 2013
 *
 */

#include <world/world.h>
#include <TiledArray/runtime.h>

namespace TiledArray {

  // Instantiate static member variables for TiledArray objects
  madness::World* Runtime::world_ = NULL;

  Runtime::Runtime(int argc, char** argv) {
    TA_USER_ASSERT(! is_initialized(),
        "TiledArray::Runtime has already been initialized; only one instance of TiledArray::Runtime is allowed.");

    // Initialize madness runtime
    madness::initialize(argc, argv);
    world_ = new madness::World(SafeMPI::COMM_WORLD);
    world_->args(argc, argv);
  }

  Runtime::~Runtime() {
    delete world_;
    world_ = NULL;
    madness::finalize();
  }

} // namespace TiledArray
