#ifndef LOCAL_ARRAY_H__INCLUDED
#define LOCAL_ARRAY_H__INCLUDED

#include <map>
#include <boost/shared_ptr.hpp>
#include <shape.h>
#include <array.h>

namespace TiledArray {

  /// Tiled Array with all data stored locally.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class LocalArray : public Array<T, DIM, CS> {
  public:
    typedef Array<T, DIM, CS> Array_;
    typedef LocalArray<T, DIM, CS> LocalArray_;
    typedef typename Array_::range_type range_type;
    typedef typename Array_::range_iterator range_iterator;
    typedef typename Array_::shape_type shape_type;
    typedef typename Array_::shape_iterator shape_iterator;
    typedef typename Array_::tile_index tile_index;
    typedef typename Array_::element_index element_index;
    typedef typename Array_::tile tile;
    typedef typename Array_::tile_ptr tile_ptr;
    typedef T value_type;
    typedef CS coordinate_system;

  protected:
    typedef typename Array_::array_map array_map;

  public:
    LocalArray(const boost::shared_ptr<shape_type>& shp) : Array_(shp) {
      // Fill data_ with tiles.
      for(shape_iterator it = this->shape()->begin();
          it != this->shape()->end();
          ++it) {
        // make TilePtr
        tile_ptr tileptr(new tile(this->tile_extent(*it)));

        // insert into tile map
        this->data_.insert(typename array_map::value_type(*it, tileptr));
      }
    }

    LocalArray(const LocalArray& other) :
      Array_(other)
    { }

    tile& at(const tile_index& index) {
      assert(includes(index));
      return this->data_[index];
    }

    const tile& at(const tile_index& index) const {
      assert(includes(index));
      return this->data_[index];
    }

    /// assign val to each element
    void assign(const value_type& val) {
      // TODO can figure out when can memset?
      for(typename array_map::iterator tile_it = this->local_data().begin();
          tile_it != this->local_data().end();
          ++tile_it) {
        tile_ptr& tileptr = (*tile_it).second;
        const size_t size = tileptr->size();
        value_type* data = tileptr->data();
        // TODO why can't I seem to be able to use multi_array::begin() here???
        std::fill(data,data+size,val);
      }
    }

    /// where is tile k
    unsigned int proc(const tile_index& index) const {
      // TODO: this should return the current process always.
      return 0;
    }

    bool local(const tile_index& index) const {
      return includes(index);
    }

  private:
	LocalArray();
//	LocalArray(const LocalArray_&);

    boost::shared_ptr<Array_> clone() const {
      return boost::shared_ptr<Array_>(new LocalArray(*this));
    }


  }; // class LocalArray

} // namespace TiledArray

#endif // LOCAL_ARRAY_H__INCLUDED
