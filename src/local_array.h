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
  protected:
	typedef Array<T, DIM, CS> Array_;
    typedef LocalArray<T, DIM, CS> LocalArray_;
    typedef typename Array_::range range;
    typedef typename Array_::range_iterator range_iterator;
    typedef typename Array_::shape shape;
    typedef typename Array_::shape_iterator shape_iterator;

  public:
    typedef T value_type;
    typedef CS coordinate_system;
	typedef typename Array_::tile_index tile_index;
	typedef typename Array_::element_index element_index;
	typedef typename Array_::Tile tile;

  protected:
    typedef typename Array_::array_map array_map;

  public:
    LocalArray(const boost::shared_ptr<shape>& shp) : Array_(shp) {
      // Fill data_ with tiles.
      for(shape_iterator it = this->shape_->begin(); it != this->shape_->end(); ++it)
        this->data_.insert(typename array_map::value_type(*it, tile()));
    }

    LocalArray_& clone(const LocalArray_& other) {
      this->range_ = other.range_;
      this->data_ = other.data_;

      return *this;
    }

    tile& at(const tile_index& index) {
      assert(includes(index));
      return this->data_[index];
    }

    const tile& at(const tile_index& index) const {
      assert(includes(index));
      return this->data_[index];
    }

    /// assign val to each element
    virtual void assign(const value_type& val) {
      for(typename array_map::iterator tile_it = this->shape_->begin(); tile_it != this->shape->end(); ++tile_it)
        for(typename tile::iterator element_it = tile_it->begin(); element_it != tile_it->end(); ++element_it)
          *element_it = val;
    }

    /// assign val to index
    virtual value_type& assign(const element_index& e_idx, const value_type& val) {
      tile_index t_idx = get_tile_index(e_idx);
      assert(includes(t_idx));
      typename array_map::iterator t_it = this->data_.find(t_idx);
      element_index base_idx( e_idx - this->shape_->range()->start_element(t_idx) );
      (*t_it)(base_idx.data()) = val;

      return *t_it;
    }

    /// where is tile k
    virtual unsigned int proc(const tile_index& index) const {
      // TODO: this should return the current process always.
      return 0;
    }

    virtual bool local(const tile_index& index) const {
      return includes(index);
    }

  private:
	LocalArray();
	LocalArray(const LocalArray_&);

  }; // class LocalArray

} // namespace TiledArray

#endif // LOCAL_ARRAY_H__INCLUDED
