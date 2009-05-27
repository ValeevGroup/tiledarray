#ifndef DISTRIBUTED_ARRAY_H__INCLUDED
#define DISTRIBUTED_ARRAY_H__INCLUDED

#include <array.h>
#include <world/world.h>

namespace TiledArray {

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class DistributedArray : public Array<T, DIM, CS>,
                           public madness::WorldObject< DistributedArray<T,DIM,CS> > {

  public:
    typedef Array<T, DIM, CS> Array_;
    typedef T value_type;
    typedef CS coordinate_system;
    typedef DistributedArray<T, DIM, CS> DistributedArray_;
    typedef typename Array_::range_type range_type;
    typedef typename Array_::range_iterator range_iterator;
    typedef typename Array_::shape_type shape_type;
    typedef typename Array_::shape_iterator shape_iterator;
    typedef typename Array_::tile_index tile_index;
    typedef typename Array_::element_index element_index;
    typedef typename Array_::iterator iterator;
    typedef typename Array_::const_iterator const_iterator;
    typedef typename Array_::tile tile;
    typedef boost::shared_ptr<tile> tile_ptr;
    typedef typename range_type::ordinal_index ordinal_index;

    /// creates an array living in world and described by shape. Optional
    /// val specifies the default value of every element
    DistributedArray(madness::World& world,
                     const boost::shared_ptr<shape_type>& shp,
                     const value_type& val = value_type()) :
      Array_(shp), madness::WorldObject<DistributedArray_>(world), tiles_(world) {

      this->process_pending();

      // Create local tiles.
      for(shape_iterator it = this->shape()->begin(); it != this->shape()->end(); ++it) {

        const tile_index& t = *it;
        const ordinal_index ot = this->range()->ordinal(t);
        if (!tiles_.is_local( ot ))
          continue;

        // make TilePtr
        tile* tileptr = new tile(this->range()->size(t),
                                 this->range()->start_element(t),
                                 val);

        // insert into tile map
        tiles_.replace(std::make_pair(ot, tileptr));
      }
    }

    /// assign val to each element
    DistributedArray_& assign(const value_type& val) {
      for(typename tile_container::iterator it = tiles_.begin(); it != tiles_.end(); ++it) {
        tile* tileptr = it->second;
        std::fill(tileptr->begin(),tileptr->end(),val);
      }
      return *this;
    }

    /// where is tile k
    unsigned int proc(const tile_index& index) const {
      return static_cast<unsigned int>(tiles_.owner( this->range()->ordinal(index) ));
    }

    /// Returns true if the tile specified by index is stored locally.
    bool local(const tile_index& index) const {
      assert(includes(index));
      return tiles_.is_local(this->range()->ordinal(index));
    }

    tile& at(const tile_index& index) {
      assert(includes(index));
      return * (tiles_.find( this->range()->ordinal(index) ).get()->second);
    }

    const tile& at(const tile_index& index) const {
      assert(includes(index));
      return * (tiles_.find( this->range()->ordinal(index) ).get()->second);
    }

    tile& operator [](const tile_index& index) {
      assert(includes(index));
      return * (tiles_.find( this->range()->ordinal(index) ).get()->second);
    }

    const tile& operator [](const tile_index& index) const {
      assert(includes(index));
      return * (tiles_.find( this->range()->ordinal(index) ).get()->second);
    }


  private:
    typedef madness::WorldContainer<ordinal_index,tile*> tile_container;
    tile_container tiles_;

    boost::shared_ptr<Array_> clone() const {
      boost::shared_ptr<Array_> array_clone(new DistributedArray(this->get_world(),
                                                                 this->shape()));
      return array_clone;
    }

  }; // class DistributedArray

} // namespace TiledArray

#endif // DISTRIBUTED_ARRAY_H__INCLUDED
