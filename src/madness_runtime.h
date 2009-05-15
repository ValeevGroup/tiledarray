#ifndef _tiledarray_madnessruntime_h_
#define _tiledarray_madnessruntime_h_

#ifdef SEEK_SET
#  undef SEEK_SET
#endif
#ifdef SEEK_CUR
#  undef SEEK_CUR
#endif
#ifdef SEEK_END
#  undef SEEK_END
#endif

#include <world/world.h>

namespace TiledArray {
#define Future madness::Future
#define DistributedRuntime madness::World
#define DistributedObject madness::WorldObject
#define DistributedContainer madness::WorldContainer
  typedef ProcessID DistributedProcessID;
};

#endif
