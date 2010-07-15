#ifndef TILEDARRAY_TYPES_H__INCLUDED
#define TILEDARRAY_TYPES_H__INCLUDED

namespace TiledArray {

  namespace detail {

    /// ArrayCoordinate Tag strut: It is used to ensure type safety between different tiling domains.
    template<unsigned int Level>
    struct LevelTag { };

    typedef enum {
      decreasing_dimension_order, // c-style
      increasing_dimension_order  // fortran
    } DimensionOrderType;

  } // namespace detail

  template <unsigned int, unsigned int, detail::DimensionOrderType, typename>
  class CoordinateSystem;
  template <typename I, unsigned int DIM, typename Tag>
  class ArrayCoordinate;
  template <typename, typename, typename>
  class Tile;
  template <typename>
  class Range;

} // namespace TiledArray

#endif // TILEDARRAY_TYPES_H__INCLUDED
