#ifndef TILEDARRAY_MATH_H__INCLUDED
#define TILEDARRAY_MATH_H__INCLUDED

namespace TiledArray {
  namespace math {

    // Forward declarations
    template <typename, typename, typename, template <typename> class>
    class BinaryOp;

    template <typename, typename, template <typename> class>
    class UnaryOp;

  } // namespace math
} // namespace TiledArray

#include <TiledArray/variable_list_math.h>
#include <TiledArray/range_math.h>
#include <TiledArray/tiled_range_math.h>
#include <TiledArray/shape_math.h>
#include <TiledArray/tile_math.h>
#include <TiledArray/array_math.h>

#endif // TILEDARRAY_MATH_H__INCLUDED
