#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/distributed_array.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/tile.h>
#include <TiledArray/annotated_array.h>
#include <TiledArray/array_util.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/array_ref.h>
#include <boost/shared_ptr.hpp>
#include <boost/functional.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <functional>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template<unsigned int DIM>
  class Permutation;
  template<typename I, unsigned int DIM, typename CS>
  class TiledRange;
  template<typename T, unsigned int DIM, typename CS>
  class Tile;
  template<typename T, typename I>
  class BaseArray;
  template <typename T, unsigned int DIM, typename CS, typename C>
  class Array;
  template<typename T, typename I, unsigned int DIM>
  BaseArray<T, I>* operator^=(BaseArray<T, I>*, const Permutation<DIM>&);
  template<typename T, unsigned int DIM, typename CS, typename C>
  void swap(Array<T, DIM, CS, C>&, Array<T, DIM, CS, C>&);

  namespace expressions {
    class VariableList;
//    namespace tile {
//      template<typename T>
//      class AnnotatedTile;
//
//    } // namespace tile
//
//    namespace array {
//
//      template<typename T>
//      class AnnotatedTile;
//    } // namespace array
  } // namespace expressions

  namespace detail {

    /// This class defines the operations for a specific tile needed by Array.
    template<typename T>
    class array_tile;

    /// This class defines the operations for a specific tile needed by Array.
    template<typename T>
    class array_tile<madness::Future<T> > : public array_tile<T> {
    public:
      typedef madness::Future<T> tile_type;
      typedef typename array_tile<T>::index_type index_type;
      typedef typename array_tile<T>::value_type value_type;

      /// Create a future to a tile that will be inserted into the array.
      template<typename I, unsigned int DIM, typename CS, typename Value>
      static madness::Future<T> create(const Range<I, DIM, LevelTag<0>, CS>& r, Value v) {
        madness::Future<T> f(array_tile<T>::create(r, v));
        return f;
      }

      /// Create a future to a tile that will be inserted into the array.
      template<typename I, unsigned int DIM, typename CS, typename InIter>
      static tile_type create(const Range<I, DIM, LevelTag<0>, CS>& r, InIter first, InIter last) {
        madness::Future<T> f(array_tile<T>::create(r, first, last));
        return f;
      }

      /// Created an annotated tile.
      static expressions::tile::AnnotatedTile<typename T::value_type> annotation(madness::Future<T>& f, const expressions::VariableList& v) {
        return array_tile<T>::annotation(f.get(), v);
      }

      static expressions::tile::AnnotatedTile<typename T::value_type> annotation(const madness::Future<T>& f, const std::string& v) {
        return array_tile<T>::annotation(f.get(), v);
      }

    };

    template<typename T, unsigned int DIM, typename CS>
    class array_tile<Tile<T, DIM, CS> > {
    public:
      typedef Tile<T, DIM, CS> tile_type;
      typedef typename Tile<T, DIM, CS>::index_type index_type;
      typedef typename Tile<T, DIM, CS>::value_type value_type;

      template<typename I>
      static tile_type create(const Range<I, DIM, LevelTag<0>, CS>& r, T val) {
        return tile_type(r, val);
      }

      template<typename I, typename InIter>
      static tile_type create(const Range<I, DIM, LevelTag<0>, CS>& r, InIter first, InIter last) {
        return tile_type(r, first, last);
      }

      static expressions::tile::AnnotatedTile<T> annotation(tile_type& t, const expressions::VariableList& v) {
        return t(v);
      }

      static expressions::tile::AnnotatedTile<T> annotation(tile_type& t, const std::string& v) {
        return t(v);
      }
    };

    template<typename T>
    class array_tile<expressions::tile::AnnotatedTile<T> > {
    public:
      typedef expressions::tile::AnnotatedTile<T> tile_type;
      typedef typename expressions::tile::AnnotatedTile<T>::index_type index_type;
      typedef typename expressions::tile::AnnotatedTile<T>::value_type value_type;

      template<typename I, unsigned int DIM, typename CS>
      static tile_type create(const Range<I, DIM, LevelTag<0>, CS>& range, T val) {
        return tile_type(range.size(), val);
      }

      template<typename I, unsigned int DIM, typename CS, typename InIter>
      static tile_type create(const Range<I, DIM, LevelTag<0>, CS>& range, InIter first, InIter last) {
        return tile_type(range.size(), first, last);
      }

      static expressions::tile::AnnotatedTile<T> annotation(tile_type& t, const expressions::VariableList& v) {
        TA_ASSERT(v == t.var(), std::runtime_error, "Variable list cannot be modified.");
        return t;
      }

      static expressions::tile::AnnotatedTile<T> annotation(tile_type& t, const std::string& v) {
        TA_ASSERT(v == t.var(), std::runtime_error, "Variable list cannot be modified.");
        return t;
      }
    };
  } // namespace detail



  /// Tiled Array with data distributed across many nodes.

  /// \arg \c T is the element type used by the tile.
  /// \arg \c DIM is the number of dimensions in the array.
  /// \arg \c CS is used to define the coordinate system.
  /// \arg \c C is the tile container type.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM>, typename C = Tile<T, DIM, CS> >
  class Array : public madness::WorldObject<Array<T, DIM, CS, C> > {
    BOOST_STATIC_ASSERT(DIM < TA_MAX_DIM);

  public:
    typedef Array<T, DIM, CS, C> Array_;
    typedef madness::WorldObject<Array<T, DIM, CS, C> > WorldObject_;
    typedef CS coordinate_system;
    typedef typename detail::array_tile<C>::tile_type tile_type;

    static const unsigned int dim;
    static const detail::DimensionOrderType order;

  private:
    typedef DistributedArray<tile_type, DIM, LevelTag<1>, coordinate_system> data_container;

  public:
    typedef typename data_container::key_type key_type;
    typedef typename data_container::index_type index_type;
    typedef typename detail::array_tile<C>::index_type tile_index_type;
    typedef typename data_container::ordinal_type ordinal_type;
    typedef typename data_container::volume_type volume_type;
    typedef typename data_container::size_array size_array;
    typedef typename data_container::value_type value_type;
    typedef TiledRange<ordinal_type, DIM, CS> tiled_range_type;
    typedef typename data_container::accessor accessor;
    typedef typename data_container::const_accessor const_accessor;
    typedef typename data_container::iterator iterator;
    typedef typename data_container::const_iterator const_iterator;
    typedef Range<ordinal_type, DIM, LevelTag<1>, coordinate_system > range_type;

  private:

    // Prohibited operations
    Array();
    Array(const Array_&);
    Array_ operator=(const Array_&);

  public:
    /// creates an array living in world and described by shape. Optional
    /// val specifies the default value of every element
    Array(madness::World& world, const tiled_range_type& rng) :
        WorldObject_(world), range_(rng), tiles_(world, rng.tiles().size())
    {
      this->process_pending();
    }

    /// creates an array living in world and described by shape. Optional
    /// val specifies the default value of every element
    Array(const madness::World& world, const tiled_range_type& rng) :
        WorldObject_(const_cast<madness::World&>(world)), range_(rng),
        tiles_(const_cast<madness::World&>(world), rng.tiles().size())
    {
      this->process_pending();
    }

    /// AnnotatedArray copy constructor
    template<typename U>
    Array(const expressions::array::AnnotatedArray<U>& aarray) :
        WorldObject_(aarray.get_world())
    {
      // TODO: Implement this function
      TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
      TA_ASSERT((aarray.dim() == DIM), std::runtime_error,
          "The dimensions of the annotated tile do not match the dimensions of the tile.");
      this->process_pending();
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// AnnotatedArray copy constructor
    template<typename U>
    Array(expressions::array::AnnotatedArray<U>&& aarray) {
      // TODO: Implement this function.
      TA_ASSERT(false, std::runtime_error, "Not yet implemented.");
      TA_ASSERT((aarray.dim() == DIM), std::runtime_error,
          "The dimensions of the annotated array do not match the dimensions of the tile.");
      this->process_pending();
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Destructor function
    ~Array() {}

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
    void clear() {
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

    madness::World& get_world() const { return WorldObject_::get_world(); }

    expressions::array::AnnotatedArray<T> operator ()(const std::string& v) {
      return expressions::array::AnnotatedArray<T>(*this, expressions::VariableList(v));
    }

    expressions::array::AnnotatedArray<T> operator ()(const std::string& v) const {
      return expressions::array::AnnotatedArray<T>(* const_cast<const Array_*>(this), expressions::VariableList(v));
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

    const tiled_range_type& range() const { return range_; }

    template<typename Key>
    ProcessID owner(const Key& k) const { return tiles_.owner(key_(k)); }

    template<typename Key>
    madness::Future<bool> probe(const Key& k) const {
      if(is_local(k)) {
        madness::Future<const_iterator> it = find(k);
        return madness::Future<bool>(it.get() != end());
      } else {
        madness::Future<bool> result;
        send(owner(k), &Array_::probe_handler, ord_(k), result.remote_ref(get_world()));
        return result;
      }
    }

  private:

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

    const ordinal_type key_(const key_type& k) const {
      return k.key1();
    }

    ordinal_type ord_(const index_type& i) const {
      return std::inner_product(i.begin(), i.end(), tiles_.weight().begin(),
          ordinal_type(0));
    }

    ordinal_type ord_(const ordinal_type& i) const {
      return i;
    }

    ordinal_type ord_(const key_type& k) const {
      if((k.keys() & 1) != 0)
        return k.key1();
      else
        return ord_(k.key2());
    }

    /// Handles probe requests
    madness::Void probe_handler(const ordinal_type& k, const madness::RemoteReference< madness::FutureImpl<bool> >& ref) const {
      madness::Future<bool> f(ref);
      madness::Future<const_iterator> t = find(k);
      f.set(t.get() != end());

      return madness::None;
    }

    friend void swap<>(Array_&, Array_&);

    tiled_range_type range_;
    data_container tiles_;
  }; // class Array

  // const static data member initialization
  template<typename T, unsigned int DIM, typename CS, typename C>
  const unsigned int Array<T, DIM, CS, C>::dim = DIM;
  template<typename T, unsigned int DIM, typename CS, typename C>
  const detail::DimensionOrderType  Array<T, DIM, CS, C>::order = CS::dimension_order;

  template<typename T, unsigned int DIM, typename CS, typename C>
  void swap(Array<T, DIM, CS, C>& a0, Array<T, DIM, CS, C>& a1) {
    TiledArray::swap(a0.range_, a1.range_);
    TiledArray::swap(a0.tiles_, a1.tiles_);
  }

} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
