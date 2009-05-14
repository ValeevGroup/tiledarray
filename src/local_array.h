#ifndef LOCAL_ARRAY_H__INCLUDED
#define LOCAL_ARRAY_H__INCLUDED

#include <map>
#include <utility>
#include <boost/shared_ptr.hpp>
#include <shape.h>
#include <tile.h>
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
    typedef typename Array_::iterator iterator;
    typedef typename Array_::const_iterator const_iterator;
    typedef typename Array_::tile tile;
    typedef boost::shared_ptr<tile> tile_ptr;
    typedef std::map<tile_index, tile_ptr> array_map;
    typedef T value_type;
    typedef CS coordinate_system;

  public:
    LocalArray(const boost::shared_ptr<shape_type>& shp, const value_type& val = value_type()) : Array_(shp) {
      // Fill data_ with tiles.
      for(shape_iterator it = this->shape()->begin(); it != this->shape()->end(); ++it) {
        // make TilePtr
        tile_ptr tileptr = boost::make_shared<tile>(this->range()->size(*it),
            this->range()->start_element(*it), val);

        // insert into tile map
        data_.insert(std::make_pair(*it, tileptr));
      }
    }

    tile& at(const tile_index& index) {
      assert(includes(index));
      return this->data_[index];
    }

    const tile& at(const tile_index& index) const {
      assert(includes(index));
      return this->data_[index];
    }

    tile& operator [](const tile_index& index) {
      assert(includes(index));
      return this->data_[index];
    }

    const tile& operator [](const tile_index& index) const {
      assert(includes(index));
      return this->data_[index];
    }

    /// assign val to each element
    LocalArray_& assign(const value_type& val) {
      for(typename array_map::iterator it = data().begin(); it != data().end(); ++it) {
        tile_ptr& tileptr = it->second;
        std::fill(tileptr->begin(),tileptr->end(),val);
      }

      return *this;
    }

    template <typename Generator>
    LocalArray_& assign(Generator gen) {
      for(typename array_map::iterator it = data().begin(); it != data().end(); ++it) {
        tile_ptr& tileptr = it->second;
        tileptr->assign(gen);
      }

      return *this;
    }

    LocalArray_& operator ^=(const Permutation<DIM>& perm) {

      array_map temp;
      for(shape_iterator it = this->shape()->begin(); it != this->shape()->end(); ++it) {
        // make TilePtr
        tile_ptr tileptr = boost::make_shared<tile>(this->range()->size(*it) ^ perm,
            this->range()->start_element(*it) ^ perm, data_[*it]);

        // insert into tile map
        temp.insert(std::make_pair(*it ^ perm, tileptr));
      }

      data_ = temp;
      permute(perm);

      return *this;
    }

    /// where is tile k
    unsigned int proc(const tile_index& index) const {
      // TODO: this should return the current process always.
      return 0;
    }

    bool local(const tile_index& index) const {
      assert(includes(index));
      return true;
    }

  private:
	LocalArray();

    /// Map that stores all tiles that are stored locally by the array.
    array_map data_;
    array_map& data() { return data_; }
    const array_map& data() const { return data_; }

    boost::shared_ptr<Array_> clone() const {
      boost::shared_ptr<Array_> array_clone(new LocalArray(this->shape()));
      return array_clone;
    }


  }; // class LocalArray

} // namespace TiledArray

#endif // LOCAL_ARRAY_H__INCLUDED
