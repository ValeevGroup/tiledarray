#ifndef TILEDARRAY_ARRAY_IMPL_H__INCLUDED
#define TILEDARRAY_ARRAY_IMPL_H__INCLUDED

#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/indexed_iterator.h>
#include <TiledArray/versioned_pmap.h>
#include <TiledArray/dense_shape.h>
#include <TiledArray/sparse_shape.h>
#include <world/worldreduce.h>
#include <world/make_task.h>
#include <world/functional.h>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

namespace TiledArray {
  namespace detail {

    template <typename T, typename CS, typename P >
    class ArrayImpl : public madness::WorldReduce<ArrayImpl<T,CS,P>, typename CS::ordinal_index>, private boost::noncopyable {
    private:
      typedef P policy;

      template <typename, typename, typename> friend class ArrayImpl;

    public:
      typedef CS coordinate_system; ///< The array coordinate system
      typedef typename policy::value_type value_type; ///< The tile type

    private:
      typedef ArrayImpl<T, CS, P> ArrayImpl_;
      typedef madness::WorldObject<ArrayImpl_> WorldObject_;
      typedef madness::WorldReduce<ArrayImpl_, typename CS::ordinal_index> WorldReduce_;
      typedef madness::Future<value_type> data_type;
      typedef madness::ConcurrentHashMap<typename coordinate_system::ordinal_index, data_type> container_type;
      typedef detail::VersionedPmap<typename coordinate_system::ordinal_index> pmap_type;
      typedef DenseShape<CS> dense_shape_type;
      typedef SparseShape<CS> sparse_shape_type;

    public:
      typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
      typedef typename coordinate_system::size_array size_array; ///< Size array type
      typedef Shape<CS> shape_type; ///< Shape type
      typedef detail::IndexedIterator<typename container_type::iterator> iterator; ///< Local tile iterator
      typedef detail::IndexedIterator<typename container_type::const_iterator> const_iterator; ///< Local tile const iterator
      typedef detail::IndexedIterator<typename container_type::accessor> accessor; ///< Local tile accessor
      typedef detail::IndexedIterator<typename container_type::const_accessor> const_accessor; ///< Local tile const accessor

      typedef TiledRange<CS> tiled_range_type; ///< Tile range type
      typedef typename tiled_range_type::range_type range_type; ///< Range type for tiles
      typedef typename tiled_range_type::tile_range_type tile_range_type; ///< Range type for elements

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param v The version number of the array
      ArrayImpl(madness::World& w, const tiled_range_type& tr, unsigned int v) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new dense_shape_type(tiled_range_.tiles(), pmap_))),
          tiles_()
      { initialize_(); }

      /// Sparse array constructor

      /// \tparam InIter The input iterator type
      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param first An input iterator that points to the a list of tiles to be
      /// added to the sparse array.
      /// \param last An input iterator that points to the last position in a list
      /// of tiles to be added to the sparse array.
      /// \param v The version number of the array
      template <typename InIter>
      ArrayImpl(madness::World& w, const tiled_range_type& tr, InIter first, InIter last, unsigned int v,
        typename madness::enable_if<is_iterator<InIter>, void*>::type = NULL) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new sparse_shape_type(w, tiled_range_.tiles(), pmap_, first, last))),
          tiles_()
      { initialize_(); }

      /// Sparse array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param first An input iterator that points to the a list of tiles to be
      /// added to the sparse array.
      /// \param last An input iterator that points to the last position in a list
      /// of tiles to be added to the sparse array.
      /// \param v The version number of the array
      template <typename LT, typename LCS, typename LP, typename RT, typename RCS, typename RP>
      ArrayImpl(madness::World& w, const tiled_range_type& tr,
        const std::shared_ptr<ArrayImpl<LT, LCS, LP> >& left,
        const std::shared_ptr<ArrayImpl<RT, RCS, RP> >& right, unsigned int v) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(shape_union<CS>(w, tiled_range_.tiles(), pmap_, *(left->shape_), *(right->shape_))),
          tiles_()
      { initialize_(); }

      template <typename LeftArrayImpl, typename RightArrayImpl>
      ArrayImpl(madness::World& w, const tiled_range_type& tr,
        const std::shared_ptr<math::Contraction<ordinal_index> >& cont,
        const std::shared_ptr<LeftArrayImpl>& left, const std::shared_ptr<RightArrayImpl>& right,
        unsigned int v) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(shape_contract<CS>(w, tiled_range_.tiles(), pmap_, cont, *(left->shape_), *(right->shape_))),
          tiles_()
      { initialize_(); }

      template <typename ArrayArgImpl>
      ArrayImpl(madness::World& w, const tiled_range_type& tr,
        const std::shared_ptr<ArrayArgImpl>& arg, unsigned int v) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(shape_copy<CS>(w, tiled_range_.tiles(), pmap_, *(arg->shape_))),
          tiles_()
      { initialize_(); }

      template <typename ArrayArgImpl>
      ArrayImpl(madness::World& w, const tiled_range_type& tr,
        const Permutation<coordinate_system::dim>& perm,
        const std::shared_ptr<ArrayArgImpl>& arg, unsigned int v) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(shape_permute<CS>(w, tiled_range_.tiles(), pmap_, perm, *(arg->shape_))),
          tiles_()
      { initialize_(); }

      /// Version number accessor

      /// \return The current version number
      std::size_t version() const { return pmap_.version(); }


      /// Begin iterator factory function

      /// \return An iterator to the first local tile.
      iterator begin() { return iterator(tiles_.begin()); }

      /// Begin const iterator factory function

      /// \return A const iterator to the first local tile.
      const_iterator begin() const { return const_iterator(tiles_.begin()); }

      /// End iterator factory function

      /// \return An iterator to one past the last local tile.
      iterator end() { return iterator(tiles_.end()); }

      /// End const iterator factory function

      /// \return A const iterator to one past the last local tile.
      const_iterator end() const { return const_iterator(tiles_.end()); }

      data_type local_find(const ordinal_index& i) const {
        // Find local tiles
        typename container_type::const_iterator it = tiles_.find(i);

        // This should never happen, because zero tiles should be caught in
        // the first check and non-zero tiles are always present.
        TA_ASSERT(it != tiles_.end());

        return it->second;
      }

      data_type remote_find(const ordinal_index& i) const {
        // Find remote tiles (which may or may not be zero tiles).
        data_type result;
        WorldObject_::task(owner(i), & ArrayImpl_::find_handler, i,
            result.remote_ref(WorldObject_::get_world()));

        return result;
      }

      /// Find a tile at index \c i

      /// \tparam Index The type of the tile index
      /// \param i The index of the tile to search for
      /// \return A future to the tile. If it is a zero tile, the future is set
      /// to an empty tile. If the tile is local, the future points to the tile
      /// at \c i . If the tile is not local, a message is sent to the tile owner
      /// and the result will be placed in the returned future.
      template <typename Index>
      data_type find(const Index& i) const {
        const ordinal_index o = ord(i);

        TA_ASSERT(tiled_range_.tiles().includes(o));

        // Check for zero tiles
        if(shape_->is_local(o))
          if(!shape_->probe(o))
            return madness::Future<value_type>(value_type());

        return (is_local(o) ? local_find(o) : remote_find(o));
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
        set_value(i, policy::construct_value(tiled_range_.make_tile_range(i), first));
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
      void set(const Index& i, const T& v) {
        set_value(i, policy::construct_value(tiled_range_.make_tile_range(i), v));
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
      void set(const Index& i, const madness::Future<value_type>& f) {
        if(f.probe())
          set_value(i, f);
        else
          task(get_world().rank(), & ArrayImpl_::template set_value<Index>,
              i, f, madness::TaskAttributes::hipri());
      }

      /// Tiled range accessor

      /// \return A const reference to the tiled range object for the array
      /// \throw nothing
      const tiled_range_type& tiling() const { return tiled_range_; }

      /// Tile range accessor

      /// \return A const reference to the range object for the array tiles
      /// \throw nothing
      const range_type& tiles() const { return tiled_range_.tiles(); }

      /// Element range accessor

      /// \return A const reference to the range object for the array elements
      /// \throw nothing
      const tile_range_type& elements() const { return tiled_range_.elements(); }

      /// Process map accessor

      /// \return A const shared pointer reference to the array process map
      /// \throw nothing
      template <typename Index>
      ProcessID owner(const Index& i) const { return pmap_.owner(ord(i)); }

      template <typename Index>
      bool is_local(const Index& i) const { return owner(i) == get_world().rank(); }

      /// Shape accessor
      const shape_type& get_shape() const { return *shape_; }

      template <typename Index>
      bool is_zero(const Index& i) const {
        if(shape_->is_local(i))
          return ! shape_->probe(i);

        return false;
      }

      using WorldObject_::get_world;
      using WorldObject_::task;
      using WorldObject_::send;

    private:

      /// Insert a tile into the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param t The value that will be used to initialize the tile data
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      madness::Void set_value(const Index& i, const value_type& t) {
        if(is_local(i)) {
          TA_ASSERT(shape_->probe(i));

          typename container_type::accessor acc;
          TA_TEST(tiles_.find(acc, ord(i)));
          TA_ASSERT(! acc->second.probe());
          acc->second.set(t);
        } else {
          send(owner(i), & ArrayImpl_::template set_value<Index>, i, t);
        }

        return madness::None;
      }

      /// Add an empty tile to the array

      /// \param i The index of the new tile
      /// \return True if the tile is successfully added to the array, false
      /// otherwise.
      /// \throw std::runtime_error If the tile is not local.
      /// \throw std::runtime_error If the tile is not included in the shape.
      bool insert_tile(const ordinal_index& i) {
        TA_ASSERT(is_local(i));
        TA_ASSERT(shape_->probe(i));

        std::pair<typename container_type::iterator, bool> result =
            tiles_.insert(typename container_type::datumT(i, data_type()));
        return result.second;
      }

      /// Initialize the array container by inserting local tiles.

      /// \throw std::runtime_error If a tile is not added array.
      void initialize_() {
        for(typename tiled_range_type::range_type::volume_type it = 0; it != tiled_range_.tiles().volume(); ++it) {
          if(is_local(it)) {
            if(shape_->probe(it)) {
              TA_TEST(insert_tile(it));
            }
          }
        }

        WorldObject_::process_pending();
      }

      /// Calculate the ordinal index

      /// \tparam Index The index type, either index or ordinal_index type.
      /// \param i The index to convert
      /// \return The ordinal index of \c i
      /// \note No range checking is done in this function.
      template <typename Index>
      ordinal_index ord(const Index& i) const {
        return tiled_range_.tiles().ord(i);
      }

      static void find_return(const typename data_type::remote_refT& ref, const value_type& value) {
        data_type result(ref);
        result.set(value);
      }

      /// Handles find request
      madness::Void find_handler(const ordinal_index& i, const typename data_type::remote_refT& ref) const {
        TA_ASSERT(is_local(i));
        if(shape_->probe(i)) {
          data_type local_tile = local_find(i);
          if(local_tile.probe())
            // The local tile is ready, send it back now.
            find_return(ref, local_tile);
          else
            get_world().taskq.add(madness::make_task(& ArrayImpl_::find_return,
                ref, local_tile, madness::TaskAttributes::hipri()));

        } else
          // Tile is zero.
          find_return(ref, policy::construct_value());

        return madness::None;
      }

      tiled_range_type tiled_range_;        ///< Tiled range object
      pmap_type pmap_;                      ///< Versioned process map
      boost::scoped_ptr<shape_type> shape_; ///< Pointer to the shape object
      container_type tiles_;                ///< Distributed container that holds tiles
    }; // class ArrayImpl

  } // detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
