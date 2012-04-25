#ifndef TILEDARRAY_ARRAY_IMPL_H__INCLUDED
#define TILEDARRAY_ARRAY_IMPL_H__INCLUDED

// This needs to be defined before world/worldreduce.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/distributed_storage.h>
#include <TiledArray/tensor.h>
#include <TiledArray/bitset.h>
#include <world/functional.h>
#include <world/nodefaults.h>

namespace TiledArray {
  namespace detail {

    template <typename T, typename CS>
    class ArrayImpl : private NO_DEFAULTS {
    private:

      template <typename, typename> friend class ArrayImpl;

    public:
      typedef ArrayImpl<T, CS> ArrayImpl_;

      typedef CS coordinate_system; ///< The array coordinate system
      typedef std::size_t volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef std::size_t ordinal_index; ///< Array ordinal index type
      typedef typename coordinate_system::size_array size_array; ///< Size array type

      typedef expressions::Tensor<T,StaticRange<typename ChildCoordinateSystem<coordinate_system>::coordinate_system> > value_type; ///< The tile type
      typedef DistributedStorage<value_type> storage_type;
      typedef typename storage_type::future future;
      typedef typename storage_type::pmap_interface pmap_interface;

      typedef typename storage_type::iterator iterator; ///< Local tile iterator
      typedef typename storage_type::const_iterator const_iterator; ///< Local tile const iterator

      typedef StaticTiledRange<CS> trange_type; ///< Tile range type
      typedef typename trange_type::range_type range_type; ///< Range type for the array
      typedef typename trange_type::tile_range_type tile_range_type; ///< Range type for elements of individual tiles and all elements

    protected:

      trange_type trange_;  ///< Tiled range object
      storage_type data_;   ///< Distributed container that holds tiles
      madness::Future<bool> ready_; ///< A future that is set once the array has been evaluated
      madness::AtomicInt initialized_; ///< A flag that indicates when the evaluation has started

    public:

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param v The version number of the array
      template <typename D>
      ArrayImpl(madness::World& w, const TiledRange<D>& tr, const std::shared_ptr<pmap_interface>& pmap) :
          trange_(tr.derived()),
          data_(w, tr.tiles().volume(), pmap),
          ready_(),
          initialized_()
      {
        initialized_ = 0;
      }

      virtual ~ArrayImpl() { }

      /// Execute one time initialization of this object. It is safe to call
      /// this function any number of times concurrently. The returned future
      /// will be called once the initialization has finished.
      madness::Future<bool> eval(const std::shared_ptr<ArrayImpl_>& pimpl) {
        // The initial value of initialized_ is zero. If it has any other value
        // then the initialization is running or has finished.
        if(! (initialized_++))
          ready_.set(this->initialize(pimpl));

        return ready_;
      }

      /// Begin iterator factory function

      /// \return An iterator to the first local tile.
      iterator begin() { return data_.begin(); }

      /// Begin const iterator factory function

      /// \return A const iterator to the first local tile.
      const_iterator begin() const { return data_.begin(); }

      /// End iterator factory function

      /// \return An iterator to one past the last local tile.
      iterator end() { return data_.end(); }

      /// End const iterator factory function

      /// \return A const iterator to one past the last local tile.
      const_iterator end() const { return data_.end(); }

      /// Find a tile at index \c i

      /// \tparam Index The type of the tile index
      /// \param i The index of the tile to search for
      /// \return A future to the tile. If it is a zero tile, the future is set
      /// to an empty tile. If the tile is local, the future points to the tile
      /// at \c i . If the tile is not local, a message is sent to the tile owner
      /// and the result will be placed in the returned future.
      template <typename Index>
      future find(const Index& i) const {
        TA_ASSERT(includes(i));
        TA_ASSERT(! is_zero(i));
        return data_[ord(i)];
      }

      /// Set the data of a tile in the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param first The first iterator for the tile data
      /// \param last The last iterator for the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      /// \throw std::runtime_error When \c first \c - \c last is not equal to the
      /// volume of the tile at \c i
      template <typename Index, typename InIter>
      void set(const Index& i, InIter first) {
        TA_ASSERT(includes(i));
        data_.set(ord(i), value_type(trange_.make_tile_range(i), first));
      }


      /// Set the data of a tile in the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param value The value that will be used to initialize the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      void set(const Index& i, const T& value) {
        const ordinal_index o = ord(i);
        TA_ASSERT(includes(i));
        TA_ASSERT(! is_zero(o));
        data_.set(o, value_type(trange_.make_tile_range(i), value));
      }

      /// Set the data of a tile in the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param v The value that will be used to initialize the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      void set(const Index& i, const future& f) {
        const ordinal_index o = ord(i);
        TA_ASSERT(includes(i));
        TA_ASSERT(! is_zero(o));
        data_.set(o, f);
      }

      /// Set the data of a tile in the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param v The value that will be used to initialize the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      void set(const Index& i, const value_type& v) {
        const ordinal_index o = ord(i);
        TA_ASSERT(includes(i));
        TA_ASSERT(! is_zero(o));
        data_.set(o, v);
      }

      template <typename Index>
      bool includes(const Index& i) const { return trange_.tiles().includes(i); }


      /// Tiled range accessor

      /// \return A const reference to the tiled range object for the array
      /// \throw nothing
      const trange_type& tiling() const { return trange_; }

      /// Tile range accessor

      /// \return A const reference to the range object for the array tiles
      /// \throw nothing
      const range_type& tiles() const { return trange_.tiles(); }

      /// Element range accessor

      /// \return A const reference to the range object for the array elements
      /// \throw nothing
      const tile_range_type& elements() const { return trange_.elements(); }

      /// Process map accessor

      /// \return A const shared pointer reference to the array process map
      /// \throw nothing
      template <typename Index>
      ProcessID owner(const Index& i) const { return data_.owner(ord(i)); }

      template <typename Index>
      bool is_local(const Index& i) const { return owner(i) == get_world().rank(); }

      const std::shared_ptr<pmap_interface>& get_pmap() const { return data_.get_pmap(); }

      template <typename Index>
      bool is_zero(const Index& i) const {
        TA_ASSERT(includes(i));
        return !(this->probe_remote_tile(ord(i)));
      }

      madness::World& get_world() const { return data_.get_world(); }

      template <typename Index, typename Value, typename Op, typename InIter>
      future reduce(const Index& i, const Value& value, Op op, InIter first, InIter last) {
        TA_ASSERT(! is_zero(i));
        const ordinal_index o = tiles().ord(i);
        future result = data_.reduce(o, value, op, first, last, owner(o));

        // Result returned on all nodes but only the root node has the final value.
        if(is_local(o))
          data_.set(o, result);

        return result;
      }

      virtual bool is_dense() const = 0;

      virtual const detail::Bitset<>& get_shape() const = 0;

      template <typename Index>
      bool insert(const Index& i) { return data_.insert(ord(i)); }

    private:

      virtual madness::Future<bool> initialize(const std::shared_ptr<ArrayImpl_>&) = 0;

      virtual bool probe_remote_tile(ordinal_index) const { return true; }

    protected:


      /// Calculate the ordinal index

      /// \tparam Index The index type, either index or ordinal_index type.
      /// \param i The index to convert
      /// \return The ordinal index of \c i
      /// \note No range checking is done in this function.
      template <typename Index>
      ordinal_index ord(const Index& i) const {
        return trange_.tiles().ord(i);
      }

    }; // class ArrayImpl

    template <typename T, typename CS>
    class DenseArrayImpl : public ArrayImpl<T, CS> {
    private:
      // shape_map_ is just a place holder. It should never be used since the
      // shape is always known.
      static const detail::Bitset<> shape_map_; ///< Empty bitset for all dense

    public:

      typedef ArrayImpl<T, CS> ArrayImpl_;
      typedef DenseArrayImpl<T, CS> DenseArrayImpl_;

      typedef CS coordinate_system; ///< The array coordinate system
      typedef std::size_t volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef std::size_t ordinal_index; ///< Array ordinal index type
      typedef typename coordinate_system::size_array size_array; ///< Size array type

      typedef expressions::Tensor<T,StaticRange<typename ChildCoordinateSystem<coordinate_system>::coordinate_system> > value_type; ///< The tile type
      typedef DistributedStorage<value_type> storage_type;
      typedef typename storage_type::future future;
      typedef typename storage_type::pmap_interface pmap_interface;

      typedef typename storage_type::iterator iterator; ///< Local tile iterator
      typedef typename storage_type::const_iterator const_iterator; ///< Local tile const iterator

      typedef StaticTiledRange<CS> trange_type; ///< Tile range type
      typedef typename trange_type::range_type range_type; ///< Range type for the array
      typedef typename trange_type::tile_range_type tile_range_type; ///< Range type for elements of individual tiles and all elements

      template <typename R>
      DenseArrayImpl(madness::World& w, const TiledRange<R>& tr,
          const std::shared_ptr<pmap_interface>& pmap) :
          ArrayImpl_(w, tr, pmap)
      { }

      virtual bool is_dense() const { return true; }

      virtual const detail::Bitset<>& get_shape() const { return shape_map_; }

    private:

      /// One time initialization

      /// Submit a task that will initialize the local tiles. The returned
      /// madness::Future will be set once the initialization has finished.
      /// \return A future to the initialization task.
      virtual madness::Future<bool> initialize(const std::shared_ptr<ArrayImpl_>& pimpl) {
        return ArrayImpl_::get_world().taskq.add(*this, & DenseArrayImpl_::insert_tiles,
            std::static_pointer_cast<DenseArrayImpl_>(pimpl),
            madness::TaskAttributes::hipri());
      }

      /// Parallel initialization operation functor
      struct InitLocalTiles {
        /// Construct the initialization functor

        /// \param pimpl The implementation pointer to the object that will be
        /// initialized
        InitLocalTiles(const std::shared_ptr<DenseArrayImpl_>& pimpl) : pimpl_(pimpl) { }

        /// Insert a local tile

        /// Insert tile \c i , if it is local.
        /// \param i The tile to be inserted
        /// \return \c true
        bool operator()(std::size_t i) const {
          if(pimpl_->is_local(i))
            pimpl_->insert(i);

          return true;
        }

        /// Serialization of this functor

        /// Serialization has not been implemented.
        /// \throw TiledArray::Exception always.
        template <typename Archive> void serialize(Archive&) { TA_ASSERT(false); }


      private:
        std::shared_ptr<DenseArrayImpl_> pimpl_; ///< shared pointer to the implementation object of the array that is being initialized.
      }; // struct InitLocalTiles

      /// Insert local tiles into the array.

      /// The inserted tiles are unassigned futures. It is the responsibility of
      /// the user to ensure that the tiles are eventually set. This function
      /// does not return until the parallel initialization has completed. It
      /// block and run tasks until the initialization has finished.
      /// \return true
      bool insert_tiles(const std::shared_ptr<DenseArrayImpl_>& pimpl) {
        // Get is called to force this function to block until the for_each
        // tasks have finished.
        return ArrayImpl_::get_world().taskq.for_each(
            madness::Range<std::size_t>(0, pimpl->tiling().tiles().volume()),
            InitLocalTiles(pimpl)).get();
      }

    }; // class DenseArrayImpl

    // DenseArrayImpl static member instantiation
    template <typename T, typename CS>
    const detail::Bitset<> DenseArrayImpl<T,CS>::shape_map_(0);

    template <typename T, typename CS>
    class SparseArrayImpl : public ArrayImpl<T, CS> {
    private:
      detail::Bitset<> shape_map_;

    public:

      typedef ArrayImpl<T, CS> ArrayImpl_;
      typedef SparseArrayImpl<T, CS> SparseArrayImpl_;

      typedef CS coordinate_system; ///< The array coordinate system
      typedef std::size_t volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef std::size_t ordinal_index; ///< Array ordinal index type
      typedef typename coordinate_system::size_array size_array; ///< Size array type

      typedef expressions::Tensor<T,StaticRange<typename ChildCoordinateSystem<coordinate_system>::coordinate_system> > value_type; ///< The tile type
      typedef DistributedStorage<value_type> storage_type;
      typedef typename storage_type::future future;
      typedef typename storage_type::pmap_interface pmap_interface;

      typedef typename storage_type::iterator iterator; ///< Local tile iterator
      typedef typename storage_type::const_iterator const_iterator; ///< Local tile const iterator

      typedef StaticTiledRange<CS> trange_type; ///< Tile range type
      typedef typename trange_type::range_type range_type; ///< Range type for the array
      typedef typename trange_type::tile_range_type tile_range_type; ///< Range type for elements of individual tiles and all elements


      /// Sparse array constructor

      /// \tparam R The tiled range type
      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param pmap The process map for tile distribution
      /// \param shape A bitset that defines the sparsity of the tensor.
      template <typename R>
      SparseArrayImpl(madness::World& w, const TiledRange<R>& tr,
          const std::shared_ptr<pmap_interface>& pmap, const Bitset<>& shape) :
          ArrayImpl_(w, tr, pmap),
          shape_map_(shape)
      {
        TA_ASSERT(shape.size() == ArrayImpl_::tiling().tiles().volume());
      }

      virtual bool is_dense() const { return false; }

      virtual const detail::Bitset<>& get_shape() const { return shape_map_; }

    private:

      /// One time initialization task

      /// Submit a task that will initialize the local tiles. The returned
      /// madness::Future will be set once the initialization has finished.
      /// \return A future to the initialization task.
      virtual madness::Future<bool> initialize(const std::shared_ptr<ArrayImpl_>& pimpl) {
        return this->get_world().taskq.add(*this, & SparseArrayImpl_::insert_tiles,
            std::static_pointer_cast<SparseArrayImpl_>(pimpl),
            madness::TaskAttributes::hipri());
      }

      /// Parallel initialization operation functor
      struct InitLocalTiles {

        /// Construct the initialization functor

        /// \param pimpl The implementation pointer to the object that will be
        /// initialized
        InitLocalTiles(const std::shared_ptr<SparseArrayImpl_>& pimpl) : pimpl_(pimpl) { }

        /// Insert a local tile

        /// Insert tile \c i , if it is local and non-zero.
        /// \param i The tile to be inserted
        /// \return \c true
        bool operator()(std::size_t i) const {
          if(pimpl_->is_local(i))
            if(pimpl_->shape_map_[i])
              pimpl_->insert(i);

          return true;
        }

        /// Serialization of this functor

        /// Serialization has not been implemented.
        /// \throw TiledArray::Exception always.
        template <typename Archive> void serialize(Archive&) { TA_ASSERT(false); }

      private:
        std::shared_ptr<SparseArrayImpl_> pimpl_;
      }; // struct InitLocalTiles


      /// Insert local tiles into the array.

      /// The inserted tiles are unassigned futures. It is the responsibility of
      /// the user to ensure that the tiles are eventually set. This function
      /// does not return until the parallel initialization has completed. It
      /// block and run tasks until the initialization has finished.
      /// \return true
      bool insert_tiles(const std::shared_ptr<SparseArrayImpl_>& pimpl) {
        madness::Future<bool> done = ArrayImpl_::get_world().taskq.for_each(
            madness::Range<std::size_t>(0, pimpl->tiling().tiles().volume()),
            InitLocalTiles(pimpl));

        return done.get();
      }

      virtual bool probe_remote_tile(ordinal_index i) const { return shape_map_[i]; }
    }; // class SparseArrayImpl

  } // detail
} // namespace TiledArray

namespace madness {
  namespace archive {

    template <typename Archive, typename T, typename CS>
    struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::detail::ArrayImpl<T,CS> > > {
      static inline void load(const Archive& ar, std::shared_ptr<TiledArray::detail::ArrayImpl<T,CS> > & ptr) {
        TA_ASSERT(false);
      }
    };

    template <typename Archive, typename T, typename CS>
    struct ArchiveStoreImpl<Archive,std::shared_ptr<TiledArray::detail::ArrayImpl<T,CS> > > {
      static inline void store(const Archive& ar, const std::shared_ptr<TiledArray::detail::ArrayImpl<T,CS> >&  ptr) {
        TA_ASSERT(false);
      }
    };


  }  // namespace archive
}  // namespace madness

#endif // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
