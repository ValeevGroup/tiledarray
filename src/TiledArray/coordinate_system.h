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

    /// Dimension order types
    typedef enum {
      decreasing_dimension_order = 1, ///< c-style dimension ordering
      increasing_dimension_order = 2  ///< fortran dimension ordering
    } DimensionOrderType;

  } // namespace detail

  /// CoordinateSystem is a policy class that specifies e.g. the order of significance of dimension.
  /// This allows to, for example, to define order of iteration to be compatible with C or Fortran arrays.
  /// Specifies the details of a D-dimensional coordinate system.
  /// The default is for the last dimension to be least significant.
  template <unsigned int DIM, unsigned int Level = 1u, detail::DimensionOrderType O = detail::decreasing_dimension_order, typename I = std::size_t>
  class CoordinateSystem {
    // Static asserts
    TA_STATIC_ASSERT(std::is_integral<I>::value);

  public:

    typedef I volume_type;                      ///< Type used to output range and array volume
    typedef I ordinal_index;                    ///< Linear ordinal index type for ranges and arrays
    typedef ArrayCoordinate<I, DIM,
        detail::LevelTag<Level> > index;        ///< Coordinate index type for ranges and arrays
    typedef std::array<I, DIM> size_array;      ///< Array type for size and weight of ranges and arrays

    static const unsigned int dim = DIM;                ///< The number of dimensions in the coordinate system
    static const unsigned int level = Level;            ///< The coordinate system level (used to differentiate types of similar coordinate systems)
    static const detail::DimensionOrderType order = O;  ///< The dimension ordering. This may be decreasing (c-style) or increasing (fortran) dimension ordering.

    static unsigned int get_dim() { return dim; }
    static unsigned int get_level() { return level; }
    static detail::DimensionOrderType get_order() { return order; }
  }; // class CoordinateSystem

  namespace detail {

    template <typename CS>
    struct ChildCoordinateSystem {
      typedef CoordinateSystem<CS::dim, CS::level - 1, CS::order, typename CS::ordinal_index> coordinate_system;
    };

  }  // namespace detail

} // namespace TiledArray


namespace madness {
  namespace archive {

    template <typename Archive, typename T>
    struct ArchiveStoreImpl;
    template <typename Archive, typename T>
    struct ArchiveLoadImpl;

    template <typename Archive>
    struct ArchiveStoreImpl<Archive, TiledArray::detail::DimensionOrderType > {
      static void store(const Archive& ar, const TiledArray::detail::DimensionOrderType& order) {
        int o = order;
        ar & o;
      }
    };

    template <typename Archive>
    struct ArchiveLoadImpl<Archive, TiledArray::detail::DimensionOrderType > {

      static void load(const Archive& ar, TiledArray::detail::DimensionOrderType& order) {
        int o = 0;
        ar & o;
        order = (o == 1 ? TiledArray::detail::decreasing_dimension_order :
            TiledArray::detail::increasing_dimension_order);
      }
    };
  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_COORDINATE_SYSTEM_H__INCLUDED
