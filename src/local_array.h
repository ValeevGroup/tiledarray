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
    typedef Array<T, DIM, CS> base_type;
    typedef LocalArray<T, DIM, CS> this_type;
    typedef typename base_type::range_type range_type;
    typedef typename base_type::range_iterator range_iterator;
    typedef typename base_type::shape_type shape_type;
    typedef typename base_type::shape_iterator shape_iterator;
    typedef typename base_type::tile_index tile_index;
    typedef typename base_type::element_index element_index;
    typedef typename base_type::Tile Tile;
    typedef boost::shared_ptr<Tile> TilePtr;
    typedef T value_type;
    typedef CS coordinate_system;

    LocalArray(const boost::shared_ptr<shape_type>& shp) : base_type(shp) {
      // Fill data_ with tiles.
      for(shape_iterator it = this->shape()->begin();
          it != this->shape()->end();
          ++it) {
        // make TilePtr
        TilePtr tile_ptr(new Tile(this->tile_extent(*it)));

        // insert into tile map
        this->data_.insert(typename array_map::value_type(*it, tile_ptr));
      }
    }

    LocalArray(const LocalArray& other) :
      base_type(other), data_(other.data_)
    {
    }

    Tile& at(const tile_index& index) {
      assert(includes(index));
      return this->data_[index];
    }

    const Tile& at(const tile_index& index) const {
      assert(includes(index));
      return this->data_[index];
    }

    /// assign val to each element
    void assign(const value_type& val) {
      // TODO can figure out when can memset?
      for(typename array_map::iterator tile_it = data().begin();
          tile_it != data().end();
          ++tile_it) {
        TilePtr& tileptr = (*tile_it).second;
        const size_t size = tileptr->size();
        value_type* data = tileptr->data();
        // TODO why can't I seem to be able to use multi_array::begin() here???
        std::fill(data,data+size,val);
      }
    }

#if 0
    /// assign val to index
    value_type& assign(const element_index& e_idx, const value_type& val) {
      tile_index t_idx = get_tile_index(e_idx);
      assert(includes(t_idx));
      typename array_map::iterator t_it = this->data_.find(t_idx);
      element_index base_idx( e_idx - this->shape_->range()->start_element(t_idx) );
      (*t_it)(base_idx.data()) = val;

      return *t_it;
    }
#endif

    /// where is tile k
    unsigned int proc(const tile_index& index) const {
      // TODO: this should return the current process always.
      return 0;
    }

    bool local(const tile_index& index) const {
      return includes(index);
    }

  private:

    boost::shared_ptr<base_type> clone() const {
      return boost::shared_ptr<base_type>(new LocalArray(*this));
    }

    /// Map that stores all local tiles.
    typedef std::map<tile_index, TilePtr> array_map;
    array_map data_;
    array_map& data() { return data_; }
    const array_map& data() const { return data_; }

  }; // class LocalArray

} // namespace TiledArray

#endif // LOCAL_ARRAY_H__INCLUDED
