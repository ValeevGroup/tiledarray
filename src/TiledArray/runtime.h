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
 *  runtime.h
 *  Jul 1, 2013
 *
 */

#ifndef TILEDARRAY_RUNTIME_H__INCLUDED
#define TILEDARRAY_RUNTIME_H__INCLUDED

#include <world/worldfwd.h>
#include <TiledArray/error.h>

namespace TiledArray {

  /// TiledArray runtime environment singleton

  /// Instantiate this object in the main() function of your program.
  class Runtime {
  private:
    static madness::World* world_; ///< The default world object

  public:
    /// Runtime singleton constructor
    Runtime(int argc, char** argv);

    /// Runtime singleton destructor
    ~Runtime();

    /// Check that the runtime environment has been initialized.
    static bool is_initialized() { return world_ != NULL; }

    /// Accessor for the default world object of TiledArray

    /// \return A reference to the default world object.
    static madness::World& get_world() {
      TA_USER_ASSERT(is_initialized(),
          "TiledArray runtime has not yet been initialized.");

      return *world_;
    }

  }; // class Runtime

} // namespace TiledArray


#endif // TILEDARRAY_RUNTIME_H__INCLUDED
