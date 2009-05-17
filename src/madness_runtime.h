#ifndef _tiledarray_madnessruntime_h_
#define _tiledarray_madnessruntime_h_

#include <boost/array.hpp>

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

/// in absence of template typedefs, use macros
namespace TiledArray {
#define Future madness::Future
#define DistributedWorld madness::World
#define DistributedObject madness::WorldObject
#define DistributedContainer madness::WorldContainer
  typedef ProcessID DistributedProcessID;
};

namespace madness {
     namespace archive {

         template <class Archive, typename T, std::size_t D>
         struct ArchiveStoreImpl<Archive, boost::array<T,D> > {
             static void store(const Archive& ar, const boost::array<T,D>& a) {
               ar & wrap(&a[0], D);
             };
         };

         template <class Archive, typename T, std::size_t D>
         struct ArchiveLoadImpl<Archive, boost::array<T,D> > {
             static void load(const Archive& ar, boost::array<T,D>& a) {
               ar & wrap((T*) &a[0], D);
             };
         };

     };
 };

#if 0
namespace madness {
     namespace archive {

         template <class Archive, typename T>
         struct ArchiveStoreImpl<Archive, boost::shared_ptr<T> > {
             static void store(const Archive& ar, const boost::shared_ptr<T>& c) {
               if (c != 0)
                 ar & *c;
             };
         };

         template <class Archive, typename T>
         struct ArchiveLoadImpl<Archive, boost::shared_ptr<T> > {
             static void load(const Archive& ar, boost::shared_ptr<T>& c) {
                 ar & *c;
             };
         };

     };
 };
#endif

#endif
