#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/array_impl.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/annotated_array.h>
#include <world/shared_ptr.h>
#include <world/worlddc.h>

namespace TiledArray {

  namespace expressions {

    template <typename> class ReadableTiledTensor;

  }  // namespace expressions

  /// An n-dimensional, tiled array

  /// Array is considered a global object
  /// \tparam T The element type of for array tiles
  /// \tparam Coordinate system type
  /// \tparam Policy class for the array
  template <typename T, typename CS>
  class Array {
  private:
    typedef detail::ArrayImpl<T, CS> impl_type;
    typedef detail::DenseArrayImpl<T,CS> dense_impl_type;
    typedef detail::SparseArrayImpl<T,CS> sparse_impl_type;

  public:
    typedef Array<T, CS> Array_; ///< This object's type
    typedef CS coordinate_system; ///< The array coordinate system

    typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
    typedef typename coordinate_system::index index; ///< Array coordinate index type
    typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
    typedef typename coordinate_system::volume_type size_type; ///< Size type
    typedef typename coordinate_system::size_array size_array; ///< Size array type

    typedef typename impl_type::value_type value_type; ///< Tile type
    typedef typename impl_type::future future; ///< Future of \c value_type
    typedef future reference; ///< \c future type
    typedef future const_reference; ///< \c future type

    typedef StaticTiledRange<CS> trange_type; ///< Tile range type
    typedef typename trange_type::range_type range_type; ///< Range type for tiles
    typedef typename trange_type::tile_range_type tile_range_type; ///< Range type for elements

    typedef typename impl_type::iterator iterator; ///< Local tile iterator
    typedef typename impl_type::const_iterator const_iterator; ///< Local tile const iterator

    typedef typename impl_type::pmap_interface pmap_interface;

  private:

    /// Task functor used to initialize Array tiles.
    class InsertTiles {
    public:
      /// Constructor

      /// \param a A pointer to the array to initialize
      InsertTiles(Array_* a) : array_(a) { }

      /// Initialize the tile given by the iterator \c it

      /// \tparam It The iterator type
      /// \param it The iterator that points to the tile to initialize
      /// \return true
      template <typename It>
      typename madness::disable_if<std::is_integral<It>, bool>::type
      operator()(const It& it) const { return insert(*it); }

      /// Initialize the tile given by the ordinal index \c i

      /// \param i The ordinal index of the tile to initialize
      /// \return true
      bool operator()(ordinal_index i) const { return insert(i); }

      /// Initialize the tile given by the index \c i

      /// \param i The index of the tile to initialize
      /// \return true
      bool operator()(const index& i) const { return insert(i); }

    private:

      /// Insert tile \c i into the array

      /// \tparam Index The index type
      /// \param
      template<typename Index>
      bool insert(const Index& i) const {
        TA_ASSERT(array_->range().includes(i));

        if(array_->is_local(i))
          return array_->pimpl_->insert(i);

        return true;
      }

      Array_* array_;
    };


    // Initialize the array tiles.
    template <typename It>
    void init(const madness::Range<It>& range) {
      // Spawn tasks to initialize the tiles
      madness::Future<bool> done = get_world().taskq.for_each(range, InsertTiles(this));

      // Wait for everyone to finish initializing tiles (work while we wait).
      done.get();
      TA_ASSERT(done.get());

      // Process pending messages.
      pimpl_->process_pending();
    }

  public:

    /// Dense array constructor

    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param v The array version number.
    template <typename R>
    Array(madness::World& w, const TiledRange<R>& tr) :
        pimpl_(new dense_impl_type(w, tr, make_pmap(w)), madness::make_deferred_deleter<impl_type>(w))
    {
      init(madness::Range<ProcessID>(0, range().volume()));
    }

    template <typename R>
    Array(madness::World& w, const TiledRange<R>& tr, const std::shared_ptr<pmap_interface>& pmap) :
        pimpl_(new dense_impl_type(w, tr, pmap), madness::make_deferred_deleter<impl_type>(w))
    {
      init(madness::Range<ProcessID>(0, range().volume()));
    }

    /// Sparse array constructor

    /// \tparam InIter Input iterator type
    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param first An input iterator that points to the a list of tiles to be
    /// added to the sparse array.
    /// \param last An input iterator that points to the last position in a list
    /// of tiles to be added to the sparse array.
    /// \param v The array version number.
    template <typename R, typename InIter>
    Array(madness::World& w, const TiledRange<R>& tr, InIter first, InIter last) :
        pimpl_(new sparse_impl_type(w, tr, make_pmap(w), first, last), madness::make_deferred_deleter<impl_type>(w))
    {
      init(madness::Range<InIter>(first, last));
    }

    template <typename R, typename InIter>
    Array(madness::World& w, const TiledRange<R>& tr, InIter first, InIter last, const std::shared_ptr<pmap_interface>& pmap) :
        pimpl_(new sparse_impl_type(w, tr, first, last, pmap), madness::make_deferred_deleter<impl_type>(w))
    {
      init(madness::Range<InIter>(first, last));
    }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array(const Array_& other) :
      pimpl_(other.pimpl_)
    { }

    template <typename Derived>
    Array(const expressions::ReadableTiledTensor<Derived>& other) :
      pimpl_((other.is_dense() ?
          static_cast<impl_type*>(new dense_impl_type(other.get_world(), other.trange(), other.get_pmap()))
        :
          static_cast<impl_type*>(new sparse_impl_type(other.get_world(), other.trange(), other.get_pmap(), other.get_shape()))),
        madness::make_deferred_deleter<impl_type>(other.get_world()))
    {
      other.derived().eval_to(*this);
      pimpl_->process_pending();
    }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array_& operator=(const Array_& other) {
      pimpl_ = other.pimpl_;
      return *this;
    }

    template <typename Derived>
    Array_& operator=(const expressions::TiledTensor<Derived>& other) {
      Array_(other).swap(*this);

      return *this;
    }

    /// Begin iterator factory function

    /// \return An iterator to the first local tile.
    iterator begin() { return pimpl_->begin(); }

    /// Begin const iterator factory function

    /// \return A const iterator to the first local tile.
    const_iterator begin() const { return pimpl_->begin(); }

    /// End iterator factory function

    /// \return An iterator to one past the last local tile.
    iterator end() { return pimpl_->end(); }

    /// End const iterator factory function

    /// \return A const iterator to one past the last local tile.
    const_iterator end() const { return pimpl_->end(); }

    /// Find local or remote tile

    /// \tparam Index The index type
    template <typename Index>
    madness::Future<value_type> find(const Index& i) const {
      return pimpl_->find(i);
    }
    /// Set the data of tile \c i

    /// \tparam Index \c index or an integral type
    /// \tparam InIter An input iterator
    template <typename Index, typename InIter>
    typename madness::enable_if<detail::is_input_iterator<InIter> >::type
    set(const Index& i, InIter first) { pimpl_->set(i, first); }

    template <typename Index>
    void set(const Index& i, const T& v = T()) { pimpl_->set(i, v); }

    template <typename Index>
    void set(const Index& i, const madness::Future<value_type>& f) { pimpl_->set(i, f); }

  private:
    template <typename U>
    static value_type value_convert(const U& u) {
      return value_type(u);
    }

  public:
    template <typename Index, typename U>
    void set(const Index& i, const madness::Future<U>& f) {
      // Todo: There is no way to set a future with with an object that is not
      // the futures type. So we use a task to make a new future of the correct
      // type.
      madness::Future<value_type> result =
          get_world().taskq.add(&value_convert<U>, f, madness::TaskAttributes::hipri());
      pimpl_->set(i, result);
    }

    template <typename Index>
    void set(const Index& i, const value_type& v) { pimpl_->set(i, v); }

    template <typename Index, typename Value, typename InIter, typename Op>
    void reduce(const Index& i, const Value& value, InIter first, InIter last, Op op) {
      pimpl_->reduce(i, value, op, first, last);
    }

    /// Tiled range accessor

    /// \return A const reference to the tiled range object for the array
    /// \throw nothing
    const trange_type& trange() const { return pimpl_->tiling(); }

    /// Tile range accessor

    /// \return A const reference to the range object for the array tiles
    /// \throw nothing
    const range_type& tiles() const { return pimpl_->tiles(); }

    /// Tile range accessor

    /// \return A const reference to the range object for the array tiles
    /// \throw nothing
    const range_type& range() const { return tiles(); }

    /// Element range accessor

    /// \return A const reference to the range object for the array elements
    /// \throw nothing
    const tile_range_type& elements() const { return pimpl_->elements(); }

    size_type size() const { return pimpl_->tiles().volume(); }

    /// Create an annotated array

    /// \param v A string with a comma-separated list of variables
    /// \return An annotated array object that references this array
    expressions::AnnotatedArray<Array_> operator ()(const std::string& v) {
      return expressions::AnnotatedArray<Array_>(*this,
          expressions::VariableList(v));
    }

    /// Create an annotated array

    /// \param v A string with a comma-separated list of variables
    /// \return An annotated array object that references this array
    const expressions::AnnotatedArray<Array_> operator ()(const std::string& v) const {
      return expressions::AnnotatedArray<Array_>(* const_cast<Array_*>(this),
          expressions::VariableList(v));
    }

    /// Create an annotated array

    /// \param v A variable list object
    /// \return An annotated array object that references this array
    expressions::AnnotatedArray<Array_> operator ()(const expressions::VariableList& v) {
      return expressions::AnnotatedArray<Array_>(*this, v);
    }

    /// Create an annotated array

    /// \param v A variable list object
    /// \return An annotated array object that references this array
    const expressions::AnnotatedArray<Array_> operator ()(const expressions::VariableList& v) const {
      return expressions::AnnotatedArray<Array_>(* const_cast<Array_*>(this), v);
    }

    /// World accessor

    /// \return A reference to the world that owns this array.
    madness::World& get_world() const { return pimpl_->get_world(); }

    /// Process map accessor

    /// \return A reference to the world that owns this array.
    const std::shared_ptr<pmap_interface>& get_pmap() const { return pimpl_->get_pmap(); }

    /// Check dense/sparse quary

    /// \return \c true when \c Array is dense, \c false otherwise.
    bool is_dense() const { return pimpl_->is_dense(); }


    /// Shape map accessor

    /// Bits are \c true when the tile exists, ether locally or remotely. No
    /// no communication required.
    /// \return A bitset that maps the existence of tiles.
    /// \throw TiledArray::Exception When the Array is dense.
    const detail::Bitset<>& get_shape() const {
      TA_ASSERT(! is_dense());
      return pimpl_->get_shape();
    }

    /// Tile ownership

    /// \tparam Index An index type
    /// \param i The index of a tile
    /// \return The process ID of the owner of a tile.
    /// \note This does not indicate whether a tile exists or not. Only, who
    /// would own it if it does exist.
    template <typename Index>
    ProcessID owner(const Index& i) const { return pimpl_->owner(i); }

    template <typename Index>
    bool is_local(const Index& i) const {
      return pimpl_->is_local(i);
    }

    /// Check for zero tiles

    /// \return \c true if tile at index \c i is zero, false if the tile is
    /// non-zero or remote existence data is not available.
    template <typename Index>
    bool is_zero(const Index& i) const {
      return pimpl_->is_zero(i);
    }

    unsigned int version() const { return pimpl_->version(); }

    /// Serialize array

    /// Interprocess serialization communication only!
    /// \param ar Output archive
    void serialize(const madness::archive::BufferOutputArchive& ar) {
        ar & static_cast<madness::WorldObject<impl_type>*>(pimpl_.get());
    }

    /// Deserialize array

    /// Interprocess serialization communication only!
    /// \param ar Input archive
    void serialize(const madness::archive::BufferInputArchive& ar) {
        madness::WorldObject<impl_type>* ptr = NULL;
        ar & ptr;
        TA_ASSERT(ptr);
        pimpl_ = std::static_pointer_cast<impl_type>(ptr->shared_from_this());
    }

    /// Swap this array with \c other

    /// \param other The array to be swapped with this array.
    void swap(Array_& other) { std::swap(pimpl_, other.pimpl_); }

  private:

    static std::shared_ptr<pmap_interface> make_pmap(madness::World& w) {
      return std::shared_ptr<madness::WorldDCDefaultPmap<size_type> >(
          new madness::WorldDCDefaultPmap<size_type>(w));
    }

    ProcessID rank() const { return pimpl_->get_world().rank(); }

    std::shared_ptr<impl_type> pimpl_;
  }; // class Array

  namespace detail {

    template <typename T, typename CS>
    struct DefaultArrayPolicy {
      typedef Eigen::aligned_allocator<T> allocator;
      typedef typename ChildCoordinateSystem<CS>::coordinate_system tile_coordinate_system;
      typedef expressions::Tensor<T, StaticRange<tile_coordinate_system>, allocator> value_type;
      typedef StaticRange<tile_coordinate_system> range_type;

      static value_type construct_value() {
        return value_type();
      }

      template <typename InIter>
      static value_type construct_value(const range_type& r, InIter first) {
        return value_type(r, first);
      }

      static value_type construct_value(const range_type& r, T value) {
        return value_type(r, value);
      }

    }; // class TilePolicy

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
