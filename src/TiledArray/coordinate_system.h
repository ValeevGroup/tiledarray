#ifndef TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
#define TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/coordinates.h>
#include <cstddef>

namespace TiledArray {

  namespace detail {

    /// Coordinate system level tag.

    /// This class is used to differentiate coordinate system types between
    /// tile coordinates (Level = 1) and element coordinate systems (Level = 0).
    template<unsigned int Level>
    struct LevelTag { };

  } // namespace detail

  /// CoordinateSystem is a policy class that specifies

  /// Specifies the details of a D-dimensional coordinate system.
  /// The default is for the last dimension to be least significant.
  template <unsigned int DIM, unsigned int Level = 1u>
  class CoordinateSystem {
  public:

    typedef ArrayCoordinate<DIM,
        detail::LevelTag<Level> > index;        ///< Coordinate index type for ranges and arrays
    typedef std::array<std::size_t, DIM> size_array;      ///< Array type for size and weight of ranges and arrays

    static const unsigned int dim = DIM;                ///< The number of dimensions in the coordinate system
    static const unsigned int level = Level;            ///< The coordinate system level (used to differentiate types of similar coordinate systems)

    static unsigned int get_dim() { return dim; }
    static unsigned int get_level() { return level; }
  }; // class CoordinateSystem

  namespace detail {

    template <typename CS>
    struct ChildCoordinateSystem {
      typedef CoordinateSystem<CS::dim, CS::level - 1> coordinate_system;
    };

  }  // namespace detail

} // namespace TiledArray

#endif // TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
