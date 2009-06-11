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
  template <typename I, unsigned int DIM, typename CS>
  class Range;
  template <typename I, unsigned int DIM, typename CS>
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

  private:
    typedef DistributedArrayStorage<tile, DIM, LevelTag<1>, coordinate_system > tile_container;

  public:
	typedef typename tile_container::ordinal_type ordinal_type;
    typedef Range<ordinal_type, DIM, CS> range_type;
    typedef Shape<ordinal_type, DIM, CS> shape_type;
    typedef typename range_type::index_type index_type;
    typedef typename tile::index_type tile_index_type;
    typedef typename range_type::size_array size_array;

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
      for(typename shape_type::const_iterator it = shape_->begin(); it != shape_->end(); ++it) {
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
    Array& assign(const value_type& val) {
      for(iterator it = begin(); it != end(); ++it)
        std::fill(it->second.begin(), it->second.end(), val);

      this->world.gop.fence(); // make sure everyone is done writing data.
      return *this;
    }

    /// Assign data to tiles with the function object gen over all elements.

    /// This function will assign data to each local element in the indices
    /// definded by [first,last). The input iterator (type InIter) must
    /// dereference to tile_index_type type. gen must be a function object or
    /// function that accepts a single tile_index_type as its parameter and
    /// returns a value_type (i.e. it must have the following signature
    /// value_type gen(const tile_index_type&).
    template <typename G>
    Array& assign(G gen) {
      for(iterator it = begin(); it != end(); ++it)
        it->second.assign(gen);

      this->world.gop.fence(); // make sure everyone is done writing data.
      return *this;
    }

    /// Assign data to tiles with the function object gen over [first,last) tiles.

    /// This function will assign data to each local tile in the indices
    /// definded by [first,last). The input iterator (type InIter) must
    /// dereference to index_type type. gen must be a function object or
    /// function that accepts a single tile_index_type as its parameter and
    /// returns a value_type (i.e. it must have the following signature
    /// value_type gen(const tile_index_type&).
    template <typename InIter, typename G>
    Array& assign(InIter first, InIter last, G gen) {
      for(; first != last; ++first)
        if(tiles_.is_local(*first))
          tiles_[*first].assign(gen);

      this->world.gop.fence(); // make sure everyone is done writing data.
      return *this;
    }

    Array& operator ^=(const Permutation<DIM>& p) {
      tiles_ ^= p; // move the tiles to the correct location
      shape_ ^= p; // shape will permute range_
      for(iterator it = begin(); it != end(); ++it) {

      }
    }

    /// Returns true if the tile specified by index is stored locally.
    bool is_local(const index_type& i) const {
      assert(shape_->includes(i));
      return tiles_.is_local(i);
    }

    bool includes(const index_type& i) const {
      return shape_->includes(i);
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
