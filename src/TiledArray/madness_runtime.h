#ifndef _tiledarray_madnessruntime_h_
#define _tiledarray_madnessruntime_h_

#include <boost/array.hpp>
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
        ar & wrap(static_cast<T*>(&a[0]), D);
      }
    };

  }
}

#endif
