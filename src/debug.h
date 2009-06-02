#ifndef DEBUG_H__INCLUDED
#define DEBUG_H__INCLUDED

#ifndef NDEBUG
#ifndef TILED_ARRAY_DEBUG
#define TILED_ARRAY_DEBUG 1
#endif
#endif

#ifdef DEBUG
#ifndef TILED_ARRAY_DEBUG
#define TILED_ARRAY_DEBUG 1
#endif
#endif

// If we are debugging TiledArray and NDEBUG is define, we need to undefine it
// so asserts will work.
#ifdef TILED_ARRAY_DEBUG
#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG 1
#else
#include <cassert>
#endif
#endif
#define TA_ASSERT( x ) assert( x )

#endif // DEBUG_H__INCLUDED
