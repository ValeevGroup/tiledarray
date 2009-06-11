#ifndef ARRAY_H__INCLUDED
#define ARRAY_H__INCLUDED

#include <array_storage.h>
#include <range.h>
#include <tile.h>

#include <cassert>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM, typename CS>
  class Range;
  template <unsigned int DIM, typename CS>
  class Shape;
  template<typename T, unsigned int DIM, typename CS>
  class Tile;

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Array : public madness::WorldObject< Array<T,DIM,CS> > {
  public:
    typedef Array<T, DIM, CS> Array_;
    typedef T value_type;
    typedef CS coordinate_system;
    typedef Tile<value_type, DIM, coordinate_system> tile;
    typedef Range<DIM, CS> range_type;
    typedef Shape<DIM, CS> shape_type;
    typedef typename range_type::ordinal_index ordinal_index;
    typedef typename range_type::index_type index_type;
    typedef typename range_type::tile_index_type tile_index_type;

  private:
    typedef DistributedArrayStorage<tile, DIM, LevelTag<1>, coordinate_system > tile_container;

  public:
    typedef typename range_type::const_iterator range_iterator;
    typedef typename shape_type::const_iterator shape_iterator;
    typedef typename tile_container::iterator iterator;
    typedef typename tile_container::const_iterator const_iterator;


    /// creates an array living in world and described by shape. Optional
    /// val specifies the default value of every element
    template <typename S>
    Array(madness::World& world, const boost::shared_ptr<S>& shp, value_type val = value_type()) :
        madness::WorldObject<Array>(world), shape_(),
        range_(), tiles_(world, shp->range()->tiles().size())
    {
      this->process_pending();

      shape_ = boost::dynamic_pointer_cast<shape_type>(shp);
      range_ = boost::const_pointer_cast<range_type>(shape_->range());
      // Create local tiles.
      for(shape_iterator it = shape_->begin(); it != shape_->end(); ++it) {
        if(tiles_.is_local( *it )) {
          tiles_[ *it ] = tile(range_->tile( *it ), val);
        }
      }

      this->world.gop.fence(); // make sure everyone is done creating tiles.
    }

    iterator begin() { return tiles_.begin(); }
    const_iterator begin() const { return tiles_.begin(); }
    iterator end() { return tiles_.end(); }
    const_iterator end() const { return tiles_.end(); }

    /// assign val to each element
    Array_& assign(const value_type& val) {
      for(iterator it = begin(); it != end(); ++it)
        std::fill(it->second.begin(), it->second.end(), val);

      this->world.gop.fence(); // make sure everyone is done writing data.
      return *this;
    }

    /// Returns true if the tile specified by index is stored locally.
    bool is_local(const index_type& index) const {
      assert(shape_->includes(index));
      return tiles_.is_local(index);
    }

    tile& at(const index_type& i) {
      assert(shape_->includes(i));
      return tiles_.at(i);
    }

    const tile& at(const index_type& i) const {
      assert(shape_->includes(i));
      return tiles_.at(i);
    }

    tile& operator [](const index_type& i) {
      assert(shape_->includes(i));
      return tiles_[i];
    }

    const tile& operator [](const index_type& i) const {
      assert(shape_->includes(i));
      return tiles_[i];
    }

  private:

	Array();
    /// Returns the tile index that contains the element index e_idx.
    index_type get_tile_index(const tile_index_type& e_idx) const {
      assert(includes(e_idx));
      return * range_->find(e_idx);
    }

    boost::shared_ptr<shape_type> shape_;
    boost::shared_ptr<range_type> range_;
    tile_container tiles_;
  }; // class Array


};

#endif // TILEDARRAY_H__INCLUDED
