#ifndef _tiledarray_madnessruntime_h_
#define _tiledarray_madnessruntime_h_

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
#include <world/worldhash.h>

#endif
