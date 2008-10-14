#ifndef _tiledarray_madnessruntime_h_
#define _tiledarray_madnessruntime_h_

#include <world/worldobj.h>
#include <world/worlddc.h>
#include <world/worldtypes.h>

namespace TiledArray {
#define DistributedRuntime madness::World
#define DistributedObject madness::WorldObject
#define DistributedContainer madness::WorldContainer
  typedef ProcessID DistributedProcessID;
};

#endif
