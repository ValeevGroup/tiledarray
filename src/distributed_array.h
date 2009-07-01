#ifndef DISTRIBUTED_ARRAY_H__INCLUDED
#define DISTRIBUTED_ARRAY_H__INCLUDED
#if 0
#include <array.h>
#include <array_storage.h>
#include <range.h>
#include <world/world.h>

namespace TiledArray {

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Array : public madness::WorldObject< DistributedArray<T,DIM,CS> > {
  public:
    typedef Array<T, DIM, CS> Array_;
    typedef T value_type;
    typedef CS coordinate_system;
    typedef Tile<value_type, DIM, coordinate_system> tile;
    typedef TiledRange<DIM, CS> range_type;
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
    Array(madness::World& world, const boost::shared_ptr<shape_type>& shp, value_type val = value_type()) :
        madness::WorldObject<DistributedArray_>(world), shape_(shp),
        range_(shp->range()) tiles_(world, shp->range()->tiles().size())
    {
      this->process_pending();

      // Create local tiles.
      for(shape_iterator it = this->shape()->begin(); it != this->shape()->end(); ++it) {
        if(tiles_.is_local( *it )) {
          tiles_[ *it ] = tile(this->range()->tile( *it ), val);
        }
      }

      this->world.gop.fence(); // make sure everyone is done creating tiles.
    }

    iterator begin() { return tiles_.begin(); }
    const_iterator begin() const { return tiles_.begin(); }
    iterator end() { return tiles_.end(); }
    const_iterator end() const { return tiles_.end(); }

    /// assign val to each element
    DistributedArray_& assign(const value_type& val) {
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

    /// Returns the tile index that contains the element index e_idx.
    index_type get_tile_index(const tile_index_type& e_idx) const {
      assert(includes(e_idx));
      return * range_->find(e_idx);
    }

    boost::shared_ptr<shape_type> shape_;
    boost::shared_ptr<range_type> range_;
    tile_container tiles_;

    boost::shared_ptr<Array_> clone() const {
      boost::shared_ptr<Array_> array_clone(new DistributedArray(this->get_world(),
                                                                 this->shape_));
      return array_clone;
    }

  }; // class Array

} // namespace TiledArray
#endif
#endif // DISTRIBUTED_ARRAY_H__INCLUDED
