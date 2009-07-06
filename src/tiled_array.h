#ifndef TILED_ARRAY_H__INCLUDED
#define TILED_ARRAY_H__INCLUDED

#include <permutation.h>
#include <coordinates.h>
#include <predicate.h>
#include <iterator.h>
#include <range.h>
#include <tiled_range.h>
#include <shape.h>
#include <tile.h>
#include <local_array.h>
#include <distributed_array.h>
#include <replicated_array.h>

// Include madness
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <world/world.h>
#endif // TILED_ARRAY_H__INCLUDED
