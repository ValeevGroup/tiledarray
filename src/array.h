#ifndef ARRAY_H__INCLUDED
#define ARRAY_H__INCLUDED

#include <array_storage.h>
#include <tiled_range.h>
#include <tile.h>

#include <cassert>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <typename I, unsigned int DIM, typename CS>
  class TiledRange;
  template<typename T, unsigned int DIM, typename CS>
  class Tile;

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Array : public madness::WorldObject< Array<T,DIM,CS> > {
  public:
    typedef Array<T, DIM, CS> Array_;
    typedef CS coordinate_system;
    typedef Tile<T, DIM, coordinate_system> tile_type;

  private:
    typedef DistributedArrayStorage<tile_type, DIM, LevelTag<1>, coordinate_system> data_container;

  public:
    typedef typename data_container::index_type index_type;
    typedef typename tile_type::index_type tile_index_type;
    typedef typename data_container::ordinal_type ordinal_type;
    typedef typename data_container::volume_type volume_type;
    typedef typename data_container::size_array size_array;
    typedef typename data_container::value_type value_type;
    typedef TiledRange<ordinal_type, DIM, CS> tiled_range_type;
    typedef tile_type & reference_type;
    typedef const tile_type & const_reference_type;
    typedef typename data_container::accessor accessor;
    typedef typename data_container::const_accessor const_accessor;
    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;

  private:
    // Prohibited operations
    Array();
    Array(const Array_&);
    Array_ operator=(const Array_&);

  public:
    /// creates an array living in world and described by shape. Optional
    /// val specifies the default value of every element
    Array(madness::World& world, const tiled_range_type& rng, value_type val = value_type()) :
        madness::WorldObject<Array_>(world), range_(rng), tiles_(world, rng.tiles().size())
    {
      this->process_pending();
    }

    /// Inserts a tile into the array.

    /// Inserts a tile with all elements initialized to a constant value.
    /// Non-local insertions will initiate non-blocking communication.
    void insert(const index_type& i, T value = T()) {
      tile_type t(range_.tile(i).start(), range_.tile(i).finish(), value);
      tiles_.insert(i, t);
    }

    /// Inserts a tile into the array.

    /// Inserts a tile with all elements initialized to the values given by the
    /// iterator range [first, last). Non-local insertions will initiate
    /// non-blocking communication.
    template<typename InIter>
    void insert(const index_type& i, InIter first, InIter last) {
      tile_type t(range_.tile(i).start(), range_.tile(i).finish(), first, last);
      tiles_.insert(i, t);
    }

    /// Inserts a tile into the array.

    /// Copies the given tile into the array. Non-local insertions will initiate
    /// non-blocking communication.
    void insert(const index_type& i, const tile_type& t) {
      TA_ASSERT(t.start() == range_.tile(i).start() && t.finish() == range_.tile(i).finish(),
          std::runtime_error("Array<...>::insert(...): Tile boundaries do not match array tile boundaries."));
      tiles_.insert(i, t);
    }

    /// Erases a tile from the array.

    /// This will remove the tile at the given index. It will initiate
    /// non-blocking for non-local tiles.
    void erase(const index_type& i) {
      tiles_.earse(i);
    }

    /// Erase a range of tiles from the array.

    /// This will remove the range of tiles from the array. The iterator must
    /// dereference to value_type (std::pair<index_type, tile_type>). It will
    /// initiate non-blocking communication for non-local tiles.
    template<typename InIter>
    void erase(InIter first, InIter last) {
      tiles_.earse(first, last);
    }

    /// Returns an iterator to the first local tile.
    iterator begin() { return tiles_.begin(); }
    /// returns a const_iterator to the first local tile.
    const_iterator begin() const { return tiles_.begin(); }
    /// Returns an iterator to the end of the local tile list.
    iterator end() { return tiles_.end(); }
    /// Returns a const_iterator to the end of the local tile list.
    const_iterator end() const { return tiles_.end(); }

    /// Resizes the array to the given tiled range.
    void resize(const tiled_range_type) {
      TA_ASSERT(false, std::runtime_error("Array<...>::resize(...): Function not implemented yet."));
    }

    /// Permutes the array. This will initiate blocking communication.
    Array& operator ^=(const Permutation<DIM>& p) {
      for(iterator it = begin(); it != end(); ++it)
        *it ^= p; // permute the individual tile
      range_ ^= p;
      tiles_ ^= p; // move the tiles to the correct location. Blocking communication here.
    }

    /// Returns true if the tile specified by index is stored locally.
    bool is_local(const index_type& i) const {
      return tiles_.is_local(i);
    }

    /// Returns true if the element specified by tile index i is stored locally.
    bool is_local(const tile_index_type& i) const {
      return range_.elements().includes(i) && is_local(get_tile_index(i));
    }

    /// Returns the index of the lower tile boundary.
    const index_type& start() const {
      return range_.tiles().start();
    }

    /// Returns the index  of the upper tile boundary.
    const index_type& finish() const {
      return range_.tiles().finish();
    }

    /// Returns a reference to the array's size array.
    const size_array& size() const {
      return tiles_.size();
    }

    /// Returns a reference to the dimension weight array.
    const size_array& weight() const {
      return tiles_.weight();
    }

    /// Returns the number of elements present in the array.

    /// If local == false, then the total number of tiles in the array will be
    /// returned. Otherwise, if local == true, it will return the number of
    /// tiles that are stored locally. The number of local tiles may or may not
    /// reflect the maximum possible number of tiles that can be stored locally.
    volume_type volume(bool local = false) const {
      return tiles_.volume(local);
    }

    /// Returns true if the tile is included in the array range.
    bool includes(const index_type& i) const {
      return tiles_.includes(i);
    }

    /// Returns a reference to the tile range object.
    const typename tiled_range_type::range_type& tiles() const {
      return range_.tiles();
    }

    /// Returns a reference to the element range object.
    const typename tiled_range_type::element_range_type& elements() const {
      return range_.elements();
    }

    /// Returns a reference to the specified tile range object.
    const typename tiled_range_type::tile_range_type& tile(const index_type& i) const {
      return range_.tile(i);
    }

  private:

    /// Returns the tile index that contains the element index e_idx.
    index_type get_tile_index(const tile_index_type& i) const {
      return * range_.find(i);
    }

    tiled_range_type range_;
    data_container tiles_;
  }; // class Array


};

#endif // TILEDARRAY_H__INCLUDED
