#ifndef ARRAY_H__INCLUDED
#define ARRAY_H__INCLUDED

#include <distributed_array.h>
#include <tiled_range.h>
#include <tile.h>
#include <annotated_array.h>
#include <array_util.h>

#include <cassert>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template<unsigned int DIM>
  class Permutation;
  template<typename I, unsigned int DIM, typename CS>
  class TiledRange;
  template<typename T, unsigned int DIM, typename CS>
  class Tile;
  template<typename T>
  class BaseArray;
  template <typename T, unsigned int DIM, typename CS>
  class Array;
  template<typename T, unsigned int DIM>
  BaseArray<T>* operator^=(BaseArray<T>*, const Permutation<DIM>&);
  template<typename T, unsigned int DIM, typename CS>
  void swap(Array<T, DIM, CS>&, Array<T, DIM, CS>&);

  namespace expressions {
    template<typename T>
    class AnnotatedArray;
  } // namespace expressions

  /// Array interface class.

  /// Provides a common interface for math operations on array objects.
  template<typename T>
  class BaseArray : public madness::WorldObject< BaseArray<T> > {
  private:
    BaseArray();
    BaseArray(const BaseArray<T>&);
    BaseArray<T>& operator=(const BaseArray<T>&);
  public:
    virtual void clear() = 0;
  protected:
    BaseArray(madness::World& world) : madness::WorldObject<BaseArray<T> >(world) { }
    virtual ~BaseArray() { }
    virtual void insert(const std::size_t, const T* first, const T* last) = 0;
    virtual void erase(const std::size_t) = 0;
    virtual bool is_local(const std::size_t) const = 0;
    virtual bool includes(const std::size_t) const = 0;
    virtual std::pair<const T*, const T*> data(const std::size_t) const = 0;
    virtual std::pair<const std::size_t*, const std::size_t*> size(const std::size_t) const = 0;
    virtual void permute(const std::size_t*) = 0;

    friend class expressions::AnnotatedArray<T>;
  }; // class BaseArray

  template<typename T, unsigned int DIM>
  BaseArray<T>* operator^=(BaseArray<T>* a, const Permutation<DIM>& p) {
    a->permute(p.begin(), p.end());
    return a;
  }

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Array : public BaseArray<T> {
  public:
    typedef Array<T, DIM, CS> Array_;
    typedef CS coordinate_system;
    typedef Tile<T, DIM, coordinate_system> tile_type;

  private:
    typedef DistributedArray<tile_type, DIM, LevelTag<1>, coordinate_system> data_container;

  public:
    typedef typename data_container::key_type key_type;
    typedef typename data_container::index_type index_type;
    typedef typename tile_type::index_type tile_index_type;
    typedef typename data_container::ordinal_type ordinal_type;
    typedef typename data_container::volume_type volume_type;
    typedef typename data_container::size_array size_array;
    typedef typename data_container::value_type value_type;
    typedef TiledRange<ordinal_type, DIM, CS> tiled_range_type;
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
    Array(madness::World& world, const tiled_range_type& rng) :
        BaseArray<T>(world), range_(rng), tiles_(world, rng.tiles().size())
    {
      this->process_pending();
    }

    /// AnnotatedArray copy constructor
    template<typename U>
    Array(const expressions::AnnotatedArray<U>& aarray) {
      // TODO: Implement this function
      TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
      TA_ASSERT((aarray.dim() == DIM), std::runtime_error,
          "The dimensions of the annotated tile do not match the dimensions of the tile.");
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// AnnotatedArray copy constructor
    template<typename U>
    Array(expressions::AnnotatedArray<U>&& aarray) {
      // TODO: Implement this function.
      TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
      TA_ASSERT((aarray.dim() == DIM), std::runtime_error,
          "The dimensions of the annotated array do not match the dimensions of the tile.");

    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Destructor function
    virtual ~Array() {}

    /// Copy the content of the other array into this array.

    /// Performs a deep copy of this array into the other array. The content of
    /// the other array will be deleted. This function is blocking and may cause
    /// some communication.
    void clone(const Array_& other) {
      range_ = other.range_;
      tiles_.clone(other.tiles_);
    }

    /// Inserts a tile into the array.

    /// Inserts a tile with all elements initialized to a constant value.
    /// Non-local insertions will initiate non-blocking communication.
    template<typename Key>
    void insert(const Key& k, T value = T()) {
      tile_type t(range_.tile(key_(k)), value);
      tiles_.insert(key_(k), t);
    }

    /// Inserts a tile into the array.

    /// Inserts a tile with all elements initialized to the values given by the
    /// iterator range [first, last). Non-local insertions will initiate
    /// non-blocking communication.
    template<typename Key, typename InIter>
    void insert(const Key& k, InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      tile_type t(range_.tile(key_(k)), first, last);
      tiles_.insert(key_(k), t);
    }

    /// Inserts a tile into the array.

    /// Copies the given tile into the array. Non-local insertions will initiate
    /// non-blocking communication.
    template<typename Key>
    void insert(const Key& k, const tile_type& t) {
      TA_ASSERT(t.range() == range_.tile(key_(k)), std::runtime_error,
          "Tile boundaries do not match array tile boundaries.");
      tiles_.insert(key_(k), t);
    }

    /// Inserts a tile into the array.

    /// Copies the given value_type into the array. Non-local insertions will
    /// initiate non-blocking communication.
    template<typename Key>
    void insert(const std::pair<Key, tile_type>& v) {
      insert(v.first, v.second);
    }

    /// Erases a tile from the array.

    /// This will remove the tile at the given index. It will initiate
    /// non-blocking for non-local tiles.
    template<typename Key>
    void erase(const Key& k) {
      tiles_.erase(key_(k));
    }

    /// Erase a range of tiles from the array.

    /// This will remove the range of tiles from the array. The iterator must
    /// dereference to value_type (std::pair<index_type, tile_type>). It will
    /// initiate non-blocking communication for non-local tiles.
    template<typename InIter>
    void erase(InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      for(; first != last; ++first)
        tiles_.erase(key_(first->first));
    }

    /// Removes all tiles from the array.
    virtual void clear() {
      tiles_.clear();
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

    /// The array will be resized to the given dimensions and tile boundaries.
    /// This will erase all data contained by the array.
    void resize(const tiled_range_type& r) {
      range_ = r;
      tiles_.resize(range_.tiles().size(), false);
    }

    /// Permutes the array. This will initiate blocking communication.
    Array_& operator ^=(const Permutation<DIM>& p) {
      for(iterator it = begin(); it != end(); ++it)
        it->second ^= p; // permute the individual tile
      range_ ^= p;
      tiles_ ^= p; // move the tiles to the correct location. Blocking communication here.

      return *this;
    }

    /// Returns true if the tile specified by index is stored locally.
    template<typename Key>
    bool is_local(const Key& k) const {
      return tiles_.is_local(key_(k));
    }

    /// Returns true if the element specified by tile index i is stored locally.
    bool is_local(const tile_index_type& i) const {
      return is_local(get_tile_index_(i));
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
    template<typename Key>
    bool includes(const Key& k) const {
      return tiles_.includes(key_(k));
    }

    /// Returns a Future iterator to an element at index i.

    /// This function will return an iterator to the element specified by index
    /// i. If the element is not local the it will use non-blocking communication
    /// to retrieve the data. The future will be immediately available if the data
    /// is local.
    template<typename Key>
    madness::Future<iterator> find(const Key& k) {
      return tiles_.find(key_(k));
    }

    /// Returns a Future const_iterator to an element at index i.

    /// This function will return a const_iterator to the element specified by
    /// index i. If the element is not local the it will use non-blocking
    /// communication to retrieve the data. The future will be immediately
    /// available if the data is local.
    template<typename Key>
    madness::Future<const_iterator> find(const Key& k) const {
      return tiles_.find(key_(k));
    }

    /// Sets an accessor to point to a local data element.

    /// This function will set an accessor to point to a local data element only.
    /// It will return false if the data element is remote or not found.
    template<typename Key>
    bool find(accessor& acc, const Key& k) {
      return tiles_.find(acc, key_(k));
    }

    /// Sets a const_accessor to point to a local data element.

    /// This function will set a const_accessor to point to a local data element
    /// only. It will return false if the data element is remote or not found.
    template<typename Key>
    bool find(const_accessor& acc, const Key& k) const {
      return tiles_.find(acc, key_(k));
    }

    expressions::AnnotatedArray<T> operator ()(const std::string& v) {
      expressions::AnnotatedArray<T> result(*this, expressions::VariableList(v));
      return result;
    }

    expressions::AnnotatedArray<const T> operator ()(const std::string& v) const {
      return expressions::AnnotatedArray<const T>(*this, expressions::VariableList(v));
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

  protected:
    /// Inserts a tile at the give ordinal index with the given data values.
    virtual void insert(const std::size_t i, const T* first, const T* last) {
      insert(i, first, last);
    }

    /// Erases a tile at the given ordinal index.
    virtual void erase(const std::size_t i) {
      erase(i);
    }

    /// Returns true if the ordinal index is stored locally.
    virtual bool is_local(const std::size_t i) const {
      return tiles_.is_local(get_index_(i));
    }

    /// Returns true if the ordinal index is included in the array.
    virtual bool includes(const std::size_t i) const {
      return i < tiles_.volume();
    }

    /// Returns a pair of pointers that point to the indicated tile's data.
    virtual std::pair<const T*, const T*> data(const std::size_t i) const {
      index_type index = get_index_(i);
      const_accessor acc;
      find(acc, index);
      const T* p = acc->second.data();
      acc.release();
      return std::make_pair<const T*, const T*>(p, p + range_.tile(index).volume());
    }

    /// Return the a pair of pointers to the size of the tile.
    virtual std::pair<const std::size_t*, const std::size_t*> size(const std::size_t i) const {
      index_type index = get_index_(i);
      return std::make_pair<const std::size_t*, const std::size_t*>
          (tile(index).size().begin(), tile(index).size().end());
    }

    virtual void permute(const std::size_t* first) {
      Permutation<DIM> p(first);
      for(iterator it = begin(); it != end(); ++it)
        it->second ^= p; // permute the individual tile
      range_ ^= p;
      tiles_ ^= p; // move the tiles to the correct location. Blocking communication here.
    }

  private:

    /// Returns the tile index that contains the element index e_idx.
    index_type get_tile_index_(const tile_index_type& i) const {
      return * range_.find(i);
    }

    /// Converts an ordinal into an index
    index_type get_index_(const ordinal_type i) const {
      index_type result;
      detail::calc_index(i, coordinate_system::rbegin(tiles_.weight()),
          coordinate_system::rend(tiles_.weight()),
          coordinate_system::rbegin(result));
      return result;
    }

    const ordinal_type& key_(const ordinal_type& o) const {
      return o;
    }

    const index_type key_(const index_type& i) const {
      return i - start();
    }

    const key_type& key_(const key_type& k) const {
      return k;
    }

    friend void swap<>(Array_&, Array_&);

    tiled_range_type range_;
    data_container tiles_;
  }; // class Array

  template<typename T, unsigned int DIM, typename CS>
  void swap(Array<T, DIM, CS>& a0, Array<T, DIM, CS>& a1) {
    TiledArray::swap(a0.range_, a1.range_);
    TiledArray::swap(a0.tiles_, a1.tiles_);
  }

} // namespace TiledArray

#endif // TILEDARRAY_H__INCLUDED
