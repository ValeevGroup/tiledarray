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

#ifndef TILEDARRAY_EXTERNAL_MADNESS_H__INCLUDED
#define TILEDARRAY_EXTERNAL_MADNESS_H__INCLUDED

#include <memory>

#include <TiledArray/config.h>

TILEDARRAY_PRAGMA_GCC(diagnostic push)
TILEDARRAY_PRAGMA_GCC(system_header)

#include <madness/world/MADworld.h>
#include <madness/world/worldmem.h>

TILEDARRAY_PRAGMA_GCC(diagnostic pop)

#include <TiledArray/error.h>

namespace TiledArray {
// Import some MADNESS classes into TiledArray for convenience.
using madness::Future;
using madness::World;

// it is useful to specify the implicit execution context for the TiledArray
// DSL on a per-scope basis ... this assumes that only 1 thread (usually, main)
// parses TiledArray DSL
namespace detail {
struct default_world {
  static World& get() {
    if (!world()) {
      TA_ASSERT(madness::initialized() &&
                "TiledArray::detail::default_world::get() called "
                "before madness::initialize() OR after madness::finalize()");
#if MADNESS_DISABLE_WORLD_GET_DEFAULT
      TA_EXCEPTION(
          "TiledArray::set_default_world() must be called before calling "
          "TiledArray::get_default_world() if MADWorld configured with "
          "DISABLE_WORLD_GET_DEFAULT=ON");
#endif
      world() = &madness::World::get_default();
    }
    return *world();
  }
  static void set(World* w) { world() = w; }
  /// @return pointer to the default world, if set, or nullptr otherwise
  static World* query() { return world(); }

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
inline void set_default_world(World& world) {
  return detail::default_world::set(&world);
}
/// Accesses the default World.
/// \return the current default World
inline World& get_default_world() { return detail::default_world::get(); }
/// Resets the default World to the world returned by
/// madness::World::get_default(), i.e. the World that spans all processes
inline void reset_default_world() {
  return detail::default_world::set(nullptr);
}

namespace {
auto world_resetter = [](World* w) { set_default_world(*w); };
}  // namespace

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
inline std::unique_ptr<World, decltype(world_resetter)> push_default_world(
    World& world) {
  World* current_world = detail::default_world::query();
  set_default_world(world);
  return std::unique_ptr<World, decltype(world_resetter)>(current_world,
                                                          world_resetter);
}

inline World split(const World& w, int color, int key = 0) {
  auto comm = w.mpi.comm().Split(color, key);
  return std::move(comm);
}

namespace detail {
inline std::pair<int, int> mpi_local_rank_size(World& world) {
  auto host_comm =
      world.mpi.comm().Split_type(SafeMPI::Intracomm::SHARED_SPLIT_TYPE, 0);
  return std::make_pair(host_comm.Get_rank(), host_comm.Get_size());
}
}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_EXTERNAL_MADNESS_H__INCLUDED
