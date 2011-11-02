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
      typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
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

    private:

      trange_type trange_;  ///< Tiled range object
      storage_type data_;   ///< Distributed container that holds tiles

    public:

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param v The version number of the array
      template <typename D>
      ArrayImpl(madness::World& w, const TiledRange<D>& tr, const std::shared_ptr<pmap_interface>& pmap) :
          trange_(tr.derived()),
          data_(w, tr.tiles().volume(), pmap, false)
      { }

      virtual ~ArrayImpl() { }

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
        return data_.find(ord(i));
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
      bool insert(const Index& i) {
        return data_.insert(ord(i));
      }

      void process_pending() {
        data_.process_pending();
      }

    private:


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

      typedef CS coordinate_system; ///< The array coordinate system
      typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
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

      typedef CS coordinate_system; ///< The array coordinate system
      typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
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

      /// \tparam InIter The input iterator type
      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param first An input iterator that points to the a list of tiles to be
      /// added to the sparse array.
      /// \param last An input iterator that points to the last position in a list
      /// of tiles to be added to the sparse array.
      /// \param v The version number of the array
      template <typename R, typename InIter>
      SparseArrayImpl(madness::World& w, const TiledRange<R>& tr,
          const std::shared_ptr<pmap_interface>& pmap, InIter first, InIter last) :
          ArrayImpl_(w, tr, pmap),
          shape_map_(tr.tiles().volume())
      {

        for(; first != last; ++first)
          shape_map_.set(ArrayImpl_::ord(*first));

        // Construct the bitset for remote data

        ArrayImpl_::get_world().gop.bit_or(shape_map_.get(), shape_map_.num_blocks());
      }

      template <typename R>
      SparseArrayImpl(madness::World& w, const TiledRange<R>& tr,
          const std::shared_ptr<pmap_interface>& pmap, const Bitset<>& shape) :
          ArrayImpl_(w, tr, pmap),
          shape_map_(shape)
      { }

      virtual bool is_dense() const { return false; }

      virtual const detail::Bitset<>& get_shape() const { return shape_map_; }

    private:

      virtual bool probe_remote_tile(ordinal_index i) const { return shape_map_[i]; }
    }; // class SparseArrayImpl

  } // detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
