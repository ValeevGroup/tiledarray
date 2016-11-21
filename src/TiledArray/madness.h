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

#ifndef TILEDARRAY_MADNESS_H__INCLUDED
#define TILEDARRAY_MADNESS_H__INCLUDED

// This needs to be defined before world/worldreduce.h and world/worlddc.h
#ifndef WORLD_INSTANTIATE_STATIC_TEMPLATES
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#endif // WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <memory>
#pragma GCC diagnostic push
#pragma GCC system_header
#include <madness/world/MADworld.h>
#include <madness/tensor/cblas.h>
#pragma GCC diagnostic pop
#include <TiledArray/error.h>

namespace TiledArray {
// Import some MADNESS classes into TiledArray for convenience.
  using madness::World;
  using madness::Future;
  using madness::initialize;
  using madness::finalize;

  // it is useful to specify the implicit execution context for the TiledArray
  // DSL on a per-scope basis ... this assumes that only 1 thread (usually, main)
  // parses TiledArray DSL
  namespace detail {
    struct default_world {
      static World& get() {
        if (!world()) {
          TA_USER_ASSERT(madness::initialized(),
                         "TiledArray::detail::default_world::get() called "
                         "before madness::initialize()");
          world() = &madness::World::get_default();
        }
        return *world();
      }
      static void set(World* w) {
        world() = w;
      }
     private:
      static World*& world() {
        static World* world_ = nullptr;
        return world_;
      }
    };
  }  // namespace detail

  /// \brief Sets the default World to \c world .

  /// Expressions that follow this call will use
  /// \c world as the default execution context (the use of default World can be
  /// overridden by explicit mechanisms for specifying the execution context,
  /// like Expr::set_world() or assigning an expression to a DistArray that has
  /// already been initialized with a World).
  ///
  /// \note set_default_world() and get_default_world() are only useful if 1
  /// thread (usually, the main thread ) creates TiledArray expressions.
  ///
  /// \param world a World object which will become the new default
  static void set_default_world(World& world) {
    return detail::default_world::set(&world);
  }
  /// Accesses the default World.
  /// \return the current default World
  static World& get_default_world() {
    return detail::default_world::get();
  }
  /// Resets the default World to the world returned by
  /// madness::World::get_default(), i.e. the World that spans all processes
  static void reset_default_world() {
    return detail::default_world::set(nullptr);
  }

  namespace {
  auto world_resetter = [](World* w) { set_default_world(*w); };
  }  // namespace detail

  /// Sets the default World to \c world and returns a smart pointer to
  /// the current default World. Releasing this pointer
  /// will automatically set the default World to the World that it points.
  /// Thus use this as follows:
  /// \code
  /// {
  ///   auto popper = push_default_world(new_world);
  ///   assert(&new_world == &get_default_world()); // new_world is now default
  ///   ... // TiledArray expressions will use new_world now
  /// }  // popper destructor resets the default World back to the old value
  /// \endcode
  ///
  /// \param world a World object which will become the new default
  /// \return a smart pointer to the current default World (i.e. not \c world)
  ///         whose deleter will reset the default World back to the stored
  ///         value
  static std::unique_ptr<World, decltype(world_resetter)>
  push_default_world(World& world) {
    World* current_world = &get_default_world();
    set_default_world(world);
    return std::unique_ptr<World, decltype(world_resetter)>(
        current_world, world_resetter);
  }

}  // namespace TiledArray

#endif // TILEDARRAY_MADNESS_H__INCLUDED
