#ifndef TILEDARRAY_PACKED_TILE_H__INCLUDED
#define TILEDARRAY_PACKED_TILE_H__INCLUDED

#include <boost/type_traits.hpp>


namespace TiledArray {

  /// Cartesian packed tile

  /// PackedTile is a Cartesian packed tile reference to another, n-dimensional,
  /// tile.
  template<typename T, unsigned int DIM>
  class PackedTile {
  public:
    typedef PackedTile<T, DIM> PackedTile_;
    typedef typename boost::remove_const<T>::type tile_type;
    typedef typename tile_type::value_type value_type;
    typedef CoordinateSystem<DIM, tile_type::coordinate_system::dimension_order> coordinate_system;
    typedef typename tile_type::ordinal_type ordinal_type;
    typedef Range<ordinal_type, DIM, LevelTag<0>, coordinate_system > range_type;
    typedef typename range_type::index_type index_type;
    typedef typename range_type::size_array size_array;
    typedef typename range_type::volume_type volume_type;
    typedef typename range_type::const_iterator index_iterator;
  private:
    typedef Eigen::Matrix< value_type , Eigen::Dynamic , Eigen::Dynamic,
        (CS::dimension_order == decreasing_dimension_order ? Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign > matrix_type;
    typedef Eigen::Map<matrix_type> map_type;
    typedef typename tile_type::const_iterator const_tile_iterator;
    typedef typename tile_type::iterator tile_iterator;
  public:
    typedef value_type & reference_type;
    typedef const value_type & const_reference_type;

  private:

    /// Default construction not allowed.
    PackedTile();

    tile_type& t_;   ///< reference to tile
    range_type r_;   ///< Packed tile range information
  };

} // namespace TiledArray

#endif // TILEDARRAY_PACKED_TILE_H__INCLUDED
