#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/tile.h>
#include <TiledArray/array_impl.h>
#include <TiledArray/type_traits.h>
#include <world/sharedptr.h>

namespace TiledArray {

  namespace math {
    template <typename>
    class Contraction;

    template <typename, typename, typename, typename>
    struct BinaryOp;

    template <typename, typename, typename>
    struct UnaryOp;

  }  // namespace math

  namespace detail {
    template <typename, typename>
    struct DefaultArrayPolicy;
  }

  /// An n-dimensional, tiled array

  /// Array is considered a global object
  /// \tparam T The element type of for array tiles
  /// \tparam Coordinate system type
  /// \tparam Policy class for the array
  template <typename T, typename CS, typename P = detail::DefaultArrayPolicy<T, CS> >
  class Array {
  private:
    typedef P array_policy;
    typedef detail::ArrayImpl<T, CS, P> impl_type;

  public:
    typedef Array<T, CS, P> Array_; ///< This object's type
    typedef CS coordinate_system; ///< The array coordinate system

    typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
    typedef typename coordinate_system::index index; ///< Array coordinate index type
    typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
    typedef typename coordinate_system::size_array size_array; ///< Size array type

    typedef typename array_policy::value_type value_type; ///< Tile type
//    typedef value_type& reference; ///< Reference to tile type
//    typedef const value_type& const_reference; ///< Const reference to tile type

    typedef TiledRange<CS> tiled_range_type; ///< Tile range type
    typedef typename tiled_range_type::range_type range_type; ///< Range type for tiles
    typedef typename tiled_range_type::tile_range_type tile_range_type; ///< Range type for elements

    typedef typename impl_type::iterator iterator; ///< Local tile iterator
    typedef typename impl_type::const_iterator const_iterator; ///< Local tile const iterator
    typedef typename impl_type::accessor accessor; ///< Local tile accessor
    typedef typename impl_type::const_accessor const_accessor; ///< Local tile const accessor

  private:

    template <typename, typename, typename, typename>
    friend class math::BinaryOp;

    template <typename, typename, typename>
    friend class math::UnaryOp;

    template <typename, typename, typename> friend class Array;

    /// Sparse array constructor

    /// \tparam InIter Input iterator type
    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param first An input iterator that points to the a list of tiles to be
    /// added to the sparse array.
    /// \param last An input iterator that points to the last position in a list
    /// of tiles to be added to the sparse array.
    /// \param v The array version number.
    template <typename LeftArray, typename RightArray>
    Array(madness::World& w, const tiled_range_type& tr, const LeftArray& left, const RightArray& right, unsigned int v) :
        pimpl_(new impl_type(w, tr, left.pimpl_, right.pimpl_, v),
            madness::make_deferred_deleter<impl_type>(w))
    { }

    template <typename LeftArray, typename RightArray>
    Array(madness::World& w, const tiled_range_type& tr,
      const std::shared_ptr<math::Contraction<ordinal_index> >& cont,
      const LeftArray& left, const RightArray& right, unsigned int v) :
        pimpl_(new impl_type(w, tr, cont, left.pimpl_, right.pimpl_, v),
            madness::make_deferred_deleter<impl_type>(w))
    { }

    Array(madness::World& w, const tiled_range_type& tr, const Array_& arg, unsigned int v) :
        pimpl_(new impl_type(w, tr, arg.pimpl_, v),
            madness::make_deferred_deleter<impl_type>(w))
    { }

    Array(madness::World& w, const tiled_range_type& tr,
      const Permutation<coordinate_system::dim>& perm, const Array_& arg,
      unsigned int v) :
        pimpl_(new impl_type(w, tr, perm, arg.pimpl_, v),
            madness::make_deferred_deleter<impl_type>(w))
    { }

  public:

    /// Dense array constructor

    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param v The array version number.
    Array(madness::World& w, const tiled_range_type& tr) :
        pimpl_(new impl_type(w, tr, 0u), madness::make_deferred_deleter<impl_type>(w))
    { }

    /// Sparse array constructor

    /// \tparam InIter Input iterator type
    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param first An input iterator that points to the a list of tiles to be
    /// added to the sparse array.
    /// \param last An input iterator that points to the last position in a list
    /// of tiles to be added to the sparse array.
    /// \param v The array version number.
    template <typename InIter>
    Array(madness::World& w, const tiled_range_type& tr, InIter first, InIter last) :
        pimpl_(new impl_type(w, tr, first, last, 0u), madness::make_deferred_deleter<impl_type>(w))
    { }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array(const Array_& other) :
      pimpl_(other.pimpl_)
    { }

    Array(const expressions::AnnotatedArray<Array_>& other) :
      pimpl_(other.array().pimpl_)
    { }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array_& operator=(const Array_& other) {
      pimpl_ = other.pimpl_;
      return *this;
    }

    Array_& operator=(const expressions::AnnotatedArray<Array_>& other) {
      return operator=(other.array());
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
    void set(const Index& i, InIter first) {
      pimpl_->set(i, first);
    }

    template <typename Index>
    void set(const Index& i, const T& v = T()) { pimpl_->set(i, v); }

    template <typename Index>
    void set(const Index& i, const madness::Future<value_type>& f) { pimpl_->set(i, f); }

    template <typename Index, typename Value, typename InIter, typename Op>
    void reduce(const Index& i, const Value& value, InIter first, InIter last, Op op) {
      TA_ASSERT(! (pimpl_->is_zero(i)), std::runtime_error, "Cannot assign a zero tile.");
      ordinal_index o = tiles().ord(i);

      madness::Future<value_type> result = pimpl_->reduce(o, value, op, first, last, owner(o));

      // Result returned on all nodes but only the root node has the final value.
      if(is_local(o))
        set(o, result);
    }

    /// Tiled range accessor

    /// \return A const reference to the tiled range object for the array
    /// \throw nothing
    const tiled_range_type& tiling() const { return pimpl_->tiling(); }

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
        TA_ASSERT(ptr, std::runtime_error, "WorldObject pointer not found.");
        pimpl_.reset(static_cast<impl_type*>(ptr), & madness::detail::no_delete<impl_type>);
    }

    /// Swap this array with \c other

    /// \param other The array to be swapped with this array.
    void swap(Array_& other) { std::swap(pimpl_, other.pimpl_); }

  private:

    ProcessID rank() const { return pimpl_->get_world().rank(); }

    std::shared_ptr<impl_type> pimpl_;
  }; // class Array

  namespace detail {

    template <typename T, typename CS>
    struct DefaultArrayPolicy {
      typedef Eigen::aligned_allocator<T> allocator;
      typedef typename ChildCoordinateSystem<CS>::coordinate_system tile_coordinate_system;
      typedef Tile<T, tile_coordinate_system, allocator> value_type;
      typedef Range<tile_coordinate_system> range_type;

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
