#ifndef _tiledarray_madnessruntime_h_
#define _tiledarray_madnessruntime_h_

#include <boost/array.hpp>
#include <TiledArray/package.h>
#include <world/worldtypes.h>

#ifdef SEEK_SET
#  undef SEEK_SET
#endif
#ifdef SEEK_CUR
#  undef SEEK_CUR
#endif
#ifdef SEEK_END
#  undef SEEK_END
#endif

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <world/world.h>
#include <TiledArray/package.h>

// Forward declaration of madness classes so we don't have to pull in all of the
// madness header files.

namespace madness {
  template <typename T>
  class Future;
  class World;
  template <class Derived>
  class WorldObject;
  template <typename keyT, typename valueT, typename hashfunT>
  class WorldContainer;
}

// in absence of template typedefs, use macros
// Removed because it can cause errors in madness if its header appears before
// this one.
#if 0
namespace TiledArray {
#define Future madness::Future
#define DistributedWorld madness::World
#define DistributedObject madness::WorldObject
#define DistributedContainer madness::WorldContainer
  typedef ProcessID DistributedProcessID;
};
#endif

namespace madness {
  namespace archive {

    template <class Archive, class T>
    struct ArchiveStoreImpl;
    template <class Archive, class T>
    struct ArchiveLoadImpl;

    template <class Archive, typename T, std::size_t D>
    struct ArchiveStoreImpl<Archive, boost::array<T,D> > {
      static void store(const Archive& ar, const boost::array<T,D>& a) {
        ar & wrap(&a[0], D);
      }
    };

    template <class Archive, typename T, std::size_t D>
    struct ArchiveLoadImpl<Archive, boost::array<T,D> > {
      static void load(const Archive& ar, boost::array<T,D>& a) {
        ar & wrap((T*) &a[0], D);
      }
    };

  }
}

#endif
