/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/replicator.h>
#include <TiledArray/pmap/replicated_pmap.h>
//#include <TiledArray/tensor.h>
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/array_impl.h>
#include <TiledArray/conversions/truncate.h>

namespace TiledArray {

  // Forward declarations
  template <typename, typename> class Tensor;
  namespace expressions {
    template <typename> class TsrExpr;
  } // namespace expressions


  /// An n-dimensional, tiled array

  /// Array is the local representation of a global object. This means that the
  /// local array object will only contain a portion of the data. It may be used
  /// to construct distributed tensor algebraic operations.
  /// \tparam T The element type of for array tiles
  /// \tparam DIM The number of dimensions for this array object
  /// \tparam Tile The tile type [ Default = \c Tensor<T> ]
  template <typename T, unsigned int DIM,
      typename Tile = Tensor<T, Eigen::aligned_allocator<T> >,
      typename Policy = DensePolicy >
  class Array {
  public:
    typedef Array<T, DIM, Tile, Policy> Array_; ///< This object's type
    typedef TiledArray::detail::ArrayImpl<Tile, Policy> impl_type;
    typedef T element_type; ///< The tile element type
    typedef typename impl_type::trange_type trange_type; ///< Tile range type
    typedef typename impl_type::range_type range_type; ///< Range type for array tiling
    typedef typename impl_type::shape_type shape_type; ///< Shape type for array tiling
    typedef typename impl_type::range_type::index index; ///< Array coordinate index type
    typedef typename impl_type::size_type size_type; ///< Size type
    typedef typename impl_type::value_type value_type; ///< Tile type
    typedef typename impl_type::eval_type eval_type; ///< The tile evaluation type
    typedef typename impl_type::reference future; ///< Future of \c value_type
    typedef typename impl_type::reference reference; ///< \c future type
    typedef typename impl_type::const_reference const_reference; ///< \c future type
    typedef typename impl_type::iterator iterator; ///< Local tile iterator
    typedef typename impl_type::const_iterator const_iterator; ///< Local tile const iterator
    typedef typename impl_type::pmap_interface pmap_interface; ///< Process map interface type

  private:

    std::shared_ptr<impl_type> pimpl_; ///< Array implementation pointer

    static madness::AtomicInt cleanup_counter_;

    /// Array deleter function

    /// This function schedules a task for lazy cleanup. Array objects are
    /// deleted only after the object has been deleted in all processes.
    /// \param pimpl The implementation pointer to be deleted.
    static void lazy_deleter(const impl_type* const pimpl) {
      try {
        if(pimpl) {
          if(madness::initialized()) {
            madness::World& world = pimpl->get_world();
            const madness::uniqueidT id = pimpl->id();
            cleanup_counter_++;
            world.gop.lazy_sync(id, [pimpl]() {
              delete pimpl;
              Array_::cleanup_counter_--;
            });
          } else {
            delete pimpl;
          }
        }
      } catch(...) {
        fprintf(stderr, "!! TiledArray ERROR: An error occurred in Array::lazy_deleter()\n");
        // Abort since we cannot throw from a destructor.
        SafeMPI::COMM_WORLD.Abort(1);
      }
    }

    /// Sparse array initialization

    /// \param world The world where the array will live.
    /// \param trange The tiled range object that will be used to set the array tiling.
    /// \param shape The array shape that defines zero and non-zero tiles
    /// \param pmap The tile index -> process map
    static std::shared_ptr<impl_type>
    init(madness::World& world, const trange_type& trange, const shape_type& shape,
        std::shared_ptr<pmap_interface> pmap)
    {
      // User level validation of input

      // Validate the tiled range
      TA_USER_ASSERT(trange.tiles().dim() == DIM,
          "Array::Array() -- The dimension of tiled range is not equal to the array.");

      if(! pmap) {
        // Construct a default process map
        pmap = Policy::default_pmap(world, trange.tiles().volume());
      } else {
        // Validate the process map
        TA_USER_ASSERT(pmap->size() == trange.tiles().volume(),
            "Array::Array() -- The size of the process map is not equal to the number of tiles in the TiledRange object.");
        TA_USER_ASSERT(pmap->rank() == typename pmap_interface::size_type(world.rank()),
            "Array::Array() -- The rank of the process map is not equal to that of the world object.");
        TA_USER_ASSERT(pmap->procs() == typename pmap_interface::size_type(world.size()),
            "Array::Array() -- The number of processes in the process map is not equal to that of the world object.");
      }

      // Validate the shape
      TA_USER_ASSERT(! shape.empty(),
          "Array::Array() -- The shape is not initialized.");
      TA_USER_ASSERT(shape.validate(trange.tiles()),
          "Array::Array() -- The range of the shape is not equal to the tiles range.");

      return std::shared_ptr<impl_type>(new impl_type(world, trange, shape, pmap), lazy_deleter);
    }

  public:
    /// Default constructor
    Array() : pimpl_() { }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array(const Array_& other) : pimpl_(other.pimpl_) { }

    /// Dense array constructor

    /// \param world The world where the array will live.
    /// \param trange The tiled range object that will be used to set the array tiling.
    /// \param pmap The tile index -> process map
    Array(madness::World& world, const trange_type& trange,
        const std::shared_ptr<pmap_interface>& pmap = std::shared_ptr<pmap_interface>()) :
      pimpl_(init(world, trange, shape_type(), pmap))
    { }

    /// Sparse array constructor

    /// \param world The world where the array will live.
    /// \param trange The tiled range object that will be used to set the array tiling.
    /// \param shape The array shape that defines zero and non-zero tiles
    /// \param pmap The tile index -> process map
    Array(madness::World& world, const trange_type& trange, const shape_type& shape,
        const std::shared_ptr<pmap_interface>& pmap = std::shared_ptr<pmap_interface>()) :
      pimpl_(init(world, trange, shape, pmap))
    { }


    /// Unary mutating constructor

    /// Created a mutated copy of other, where each tile is mutated by \c op.
    ///
    /// \param other The array to be copied
    template <typename Op>
    Array(const Array_& other, Op&& op) :
      pimpl_()
    {
      *this = foreach(other, op);
    }

    /// Destructor

    /// This is a distributed lazy destructor. The object will only be deleted
    /// after the last reference to the world object on all nodes has been
    /// destroyed.
    ~Array() { }

    static void wait_for_lazy_cleanup(madness::World& world, const double timeout = 60.0) {
      try {
        madness::ThreadPool::await([&]() { return (cleanup_counter_ == 0); }, true);
      } catch(std::runtime_error& e) {
        printf("%i: Array lazy cleanup timeout with %i pending cleanup(s)\n",
            world.rank(), int(cleanup_counter_));
        throw;
      }
    }

    /// Copy constructor

    /// This is a shallow copy, that is no data is copied.
    /// \param other The array to be copied
    Array_& operator=(const Array_& other) {
      pimpl_ = other.pimpl_;

      return *this;
    }

    /// Global object id

    /// \return A globally unique identifier.
    /// \note This function is primarily used for debugging purposes. Users
    /// should not rely on this function.
    madness::uniqueidT id() const { return pimpl_->id(); }

    /// Begin iterator factory function

    /// \return An iterator to the first local tile.
    iterator begin() {
      check_pimpl();
      return pimpl_->begin();
    }

    /// Begin const iterator factory function

    /// \return A const iterator to the first local tile.
    const_iterator begin() const {
      check_pimpl();
      return pimpl_->begin();
    }

    /// End iterator factory function

    /// \return An iterator to one past the last local tile.
    iterator end() {
      check_pimpl();
      return pimpl_->end();
    }

    /// End const iterator factory function

    /// \return A const iterator to one past the last local tile.
    const_iterator end() const {
      check_pimpl();
      return pimpl_->end();
    }

    /// Find local or remote tile

    /// \tparam Index The index type
    template <typename Index>
    madness::Future<value_type> find(const Index& i) const {
      check_index(i);
      return pimpl_->get(i);
    }

    /// Set the data of tile \c i

    /// \tparam Index \c index or an integral type
    /// \tparam InIter An input iterator
    /// \param i The index of the tile to be set
    /// \param first The iterator that points to the new tile data
    template <typename Index, typename InIter>
    typename std::enable_if<detail::is_input_iterator<InIter>::value>::type
    set(const Index& i, InIter first) {
      check_index(i);
      pimpl_->set(i, value_type(pimpl_->trange().make_tile_range(i), first));
    }

    template <typename Index>
    void set(const Index& index, const T& value = T()) {
      check_index(index);
      pimpl_->set(index, value_type(pimpl_->trange().make_tile_range(index), value));
    }

    /// Set tile \c i with future \c f

    /// \tparam Index The index type (i.e. index or size_type)
    /// \param i The tile index to be set
    /// \param f A future to the tile
    template <typename Index>
    void set(const Index& i, const madness::Future<value_type>& f) {
      check_index(i);
      pimpl_->set(i, f);
    }

    /// Set tile \c i to value \c v

    /// \tparam Index The index type (i.e. index or size_type)
    /// \param i The tile index to be set
    /// \param v The tile value
    template <typename Index>
    void set(const Index& i, const value_type& v) {
      check_index(i);
      pimpl_->set(i, v);
    }

    /// Fill all local tiles

    /// \param value The fill value
    void fill_local(const T& value = T()) {
      init_local_tiles([=] (const range_type& range)
          { return value_type(range, value); });
    }

    /// Fill all local tiles

    /// \param value The fill value
    void set_all_local(const T& value = T()) {
      fill_local(value);
    }


    /// Initialize tiles with a user provided functor

    /// This function is used to initialize tiles of the array via a function
    /// (or functor). The work is done in parallel, therefore \c op must be a
    /// thread safe function/functor. The signature of the functor should be:
    /// \code
    /// value_type op(const range_type&)
    /// \endcode
    /// \tparam Op Tile operation type
    /// \param op The operation used to generate tiles
    /// \param wait Wait for all tiles to be set before proceeding
    /// \note It is typically not necessary to wait for tile initialization
    /// before using arrays in tensor arithmetic expressions.
    template <typename Op>
    void init_local_tiles(Op&& op, const bool wait = false) {
      check_pimpl();
      madness::Range<typename pmap_interface::const_iterator>
          range(pimpl_->pmap()->begin(), pimpl_->pmap()->end(), 8);

      Array_ array(*this);

      madness::Future<bool> result = pimpl_->get_world().taskq.for_each(range,
          [=] (const typename pmap_interface::const_iterator& it) mutable {
            const size_type index = *it;
            if(! array.is_zero(index))
              array.set(index, op(array.trange().make_tile_range(*it)));
            return true;
          });

      // Wait for all tiles to be set
      if(wait)
        result.get();
    }

    /// Tiled range accessor

    /// \return A const reference to the tiled range object for the array
    /// \throw nothing
    const trange_type& trange() const {
      check_pimpl();
      return pimpl_->trange();
    }

    /// Tile range accessor

    /// \return A const reference to the range object for the array tiles
    /// \throw nothing
    const range_type& range() const {
      check_pimpl();
      return pimpl_->range();
    }

    /// Element range accessor

    /// \return A const reference to the range object for the array elements
    /// \throw nothing
    const typename trange_type::tile_range_type& elements() const {
      check_pimpl();
      return pimpl_->trange().elements();
    }

    size_type size() const {
      check_pimpl();
      return pimpl_->size();
    }

    /// Create a tensor expression

    /// \param vars A string with a comma-separated list of variables
    /// \return A const tensor expression object
    TiledArray::expressions::TsrExpr<const Array_>
    operator ()(const std::string& vars) const {
#ifndef NDEBUG
      const unsigned int n = 1u + std::count_if(vars.begin(), vars.end(),
          [](const char c) { return c == ','; });
      if(n != DIM) {
        if(madness::World::get_default().rank() == 0) {
          TA_USER_ERROR_MESSAGE( \
              "The number of array annotation variables is not equal to the array dimension:" \
              << "\n    number of variables  = " << n \
              << "\n    array dimension      = " << DIM );
        }

        TA_EXCEPTION("The number of array annotation variables is not equal to the array dimension.");
      }
#endif // NDEBUG
      return TiledArray::expressions::TsrExpr<const Array_>(*this, vars);
    }

    /// Create a tensor expression

    /// \param vars A string with a comma-separated list of variables
    /// \return A non-const tensor expression object
    TiledArray::expressions::TsrExpr<Array_>
    operator ()(const std::string& vars) {
#ifndef NDEBUG
      const unsigned int n = 1u + std::count_if(vars.begin(), vars.end(),
          [](const char c) { return c == ','; });
      if(n != DIM) {
        if(madness::World::get_default().rank() == 0) {
          TA_USER_ERROR_MESSAGE( \
              "The number of array annotation variables is not equal to the array dimension:" \
              << "\n    number of variables  = " << n \
              << "\n    array dimension      = " << DIM );
        }

        TA_EXCEPTION("The number of array annotation variables is not equal to the array dimension.");
      }
#endif // NDEBUG
      return TiledArray::expressions::TsrExpr<Array_>(*this, vars);
    }

    /// World accessor

    /// \return A reference to the world that owns this array.
    madness::World& get_world() const {
      check_pimpl();
      return pimpl_->get_world();
    }

    /// Process map accessor

    /// \return A reference to the world that owns this array.
    const std::shared_ptr<pmap_interface>& get_pmap() const {
      check_pimpl();
      return pimpl_->pmap();
    }

    /// Check dense/sparse

    /// \return \c true when \c Array is dense, \c false otherwise.
    bool is_dense() const {
      check_pimpl();
      return pimpl_->is_dense();
    }


    /// Shape map accessor

    /// Bits are \c true when the tile exists, either locally or remotely. No
    /// no communication required.
    /// \return A bitset that maps the existence of tiles.
    /// \throw TiledArray::Exception When the Array is dense.
    const shape_type& get_shape() const {  return pimpl_->shape(); }

    /// Tile ownership

    /// \tparam Index An index type
    /// \param i The index of a tile
    /// \return The process ID of the owner of a tile.
    /// \note This does not indicate whether a tile exists or not. Only, who
    /// would own it if it does exist.
    template <typename Index>
    ProcessID owner(const Index& i) const {
      check_index(i);
      return pimpl_->owner(i);
    }

    template <typename Index>
    bool is_local(const Index& i) const {
      check_index(i);
      return pimpl_->is_local(i);
    }

    /// Check for zero tiles

    /// \return \c true if tile at index \c i is zero, false if the tile is
    /// non-zero or remote existence data is not available.
    template <typename Index>
    bool is_zero(const Index& i) const {
      check_index(i);
      return pimpl_->is_zero(i);
    }

    /// Swap this array with \c other

    /// \param other The array to be swapped with this array.
    void swap(Array_& other) { std::swap(pimpl_, other.pimpl_); }

    /// Convert a distributed \c Array into a replicated array
    void make_replicated() {
      check_pimpl();
      if((! pimpl_->pmap()->is_replicated()) && (get_world().size() > 1)) {
        // Construct a replicated array
        std::shared_ptr<pmap_interface> pmap(new detail::ReplicatedPmap(get_world(), size()));
        Array_ result = Array_(get_world(), trange(), get_shape(), pmap);

        // Create the replicator object that will do an all-to-all broadcast of
        // the local tile data.
        std::shared_ptr<detail::Replicator<Array_> > replicator(
            new detail::Replicator<Array_>(*this, result));

        // Put the replicator pointer in the deferred cleanup object so it will
        // be deleted at the end of the next fence.
        TA_ASSERT(replicator.unique()); // Required for deferred_cleanup
        madness::detail::deferred_cleanup(get_world(), replicator);

        Array_::operator=(result);
      }
    }

    /// Update shape data and remove tiles that are below the zero threshold

    /// \note This function is a no-op for dense arrays.
    void truncate() { TiledArray::truncate(*this); }

    bool is_initialized() const { return static_cast<bool>(pimpl_); }

  private:

    template <typename Index>
    typename std::enable_if<std::is_integral<Index>::value>::type
    check_index(const Index i) const {
      check_pimpl();
      TA_USER_ASSERT(pimpl_->range().includes(i),
          "The ordinal index used to access an array tile is out of range.");
    }

    template <typename Index>
    typename std::enable_if<! std::is_integral<Index>::value>::type
    check_index(const Index& i) const {
      check_pimpl();
      TA_USER_ASSERT(pimpl_->range().includes(i),
          "The coordinate index used to access an array tile is out of range.");
      TA_USER_ASSERT(i.size() == DIM,
          "The number of elements in the coordinate index does not match the dimension of the array.");
    }

    /// Makes sure pimpl has been initialized
    void check_pimpl() const {
      TA_USER_ASSERT(pimpl_,
          "The Array has not been initialized, likely reason: it was default constructed and used.");
    }

  }; // class Array


  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  madness::AtomicInt Array<T, DIM, Tile, Policy>::cleanup_counter_;

  /// Add the tensor to an output stream

  /// This function will iterate through all tiles on node 0 and print non-zero
  /// tiles. It will wait for each tile to be evaluated (i.e. it is a blocking
  /// function). Tasks will continue to be processed.
  /// \tparam T The element type of Array
  /// \tparam DIM The number of dimensions
  /// \tparam Tile The Tile type
  /// \param os The output stream
  /// \param a The array to be put in the output stream
  /// \return A reference to the output stream
  template <typename T, unsigned int DIM, typename Tile, typename Policy>
  inline std::ostream& operator<<(std::ostream& os, const Array<T, DIM, Tile, Policy>& a) {
    if(a.get_world().rank() == 0) {
      for(std::size_t i = 0; i < a.size(); ++i)
        if(! a.is_zero(i)) {
          const typename Array<T, DIM, Tile>::value_type tile = a.find(i).get();
          os << i << ": " << tile  << "\n";
        }
    }
    a.get_world().gop.fence();
    return os;
  }

} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
