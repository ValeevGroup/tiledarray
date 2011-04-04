#ifndef TILEDARRAY_ARRAY_IMPL_H__INCLUDED
#define TILEDARRAY_ARRAY_IMPL_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/key.h>
#include <TiledArray/indexed_iterator.h>
#include <TiledArray/versioned_pmap.h>
#include <TiledArray/dense_shape.h>
#include <TiledArray/sparse_shape.h>
#include <TiledArray/pred_shape.h>
#include <boost/noncopyable.hpp>

namespace TiledArray {
  namespace detail {

    template <typename T, typename CS, typename P >
    class ArrayImpl : public madness::WorldObject<ArrayImpl<T,CS,P> >, private boost::noncopyable {
    public:
      typedef CS coordinate_system; ///< The array coordinate system

    private:
      typedef ArrayImpl<T, CS, P> ArrayImpl_;
      typedef madness::WorldObject<ArrayImpl_> WorldObject_;
      typedef P policy;
      typedef typename coordinate_system::key_type key_type;
      typedef madness::Future<typename policy::value_type> data_type;
      typedef madness::ConcurrentHashMap<key_type, data_type> container_type;
      typedef detail::VersionedPmap<key_type> pmap_type;
      typedef Shape<CS> shape_type;
      typedef DenseShape<CS> dense_shape_type;
      typedef SparseShape<CS> sparse_shape_type;

    public:
      typedef typename policy::value_type value_type; /// The array value type (i.e. tiles)
      typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
      typedef typename coordinate_system::size_array size_array; ///< Size array type
      typedef detail::IndexedIterator<typename container_type::iterator> iterator; ///< Local tile iterator
      typedef detail::IndexedIterator<typename container_type::const_iterator> const_iterator; ///< Local tile const iterator

      typedef TiledRange<CS> tiled_range_type; ///< Tile range type
      typedef typename tiled_range_type::range_type range_type; ///< Range type for tiles
      typedef typename tiled_range_type::tile_range_type tile_range_type; ///< Range type for elements

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param v The version number of the array
      ArrayImpl(madness::World& w, const tiled_range_type& tr, unsigned int v) :
          WorldObject_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new dense_shape_type(tiled_range_.tiles(), pmap_))),
          tiles_()
      { }

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param first An input iterator that points to the a list of tiles to be
      /// added to the sparse array.
      /// \param last An input iterator that points to the last position in a list
      /// of tiles to be added to the sparse array.
      /// \param v The version number of the array
      template <typename InIter>
      ArrayImpl(madness::World& w, const tiled_range_type& tr, InIter first, InIter last, unsigned int v) :
          WorldObject_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new sparse_shape_type(w, tiled_range_.tiles(), pmap_, first, last))),
          tiles_()
      { }

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param p The predicate for the array shape
      /// \param v The version number for the array
      template <typename Pred>
      ArrayImpl(madness::World& w, const tiled_range_type& tr, const Pred& p, unsigned int v) :
          WorldObject_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new PredShape<coordinate_system, Pred>(tiled_range_.tiles(), pmap_, p))),
          tiles_()
      { }

      /// Version number accessor

      /// \return The current version number
      std::size_t version() const { return pmap_->version(); }


      /// Begin iterator factory function

      /// \return An iterator to the first local tile.
      iterator begin() { return iterator(tiles_->begin()); }

      /// Begin const iterator factory function

      /// \return A const iterator to the first local tile.
      const_iterator begin() const { return const_iterator(tiles_->begin()); }

      /// End iterator factory function

      /// \return An iterator to one past the last local tile.
      iterator end() { return iterator(tiles_->end()); }

      /// End const iterator factory function

      /// \return A const iterator to one past the last local tile.
      const_iterator end() const { return const_iterator(tiles_->end()); }

      /// Tile future accessor

      /// Search for and return a future to the tile.
      template <typename Index>
      data_type local_find(const Index& i) const {
        TA_ASSERT(shape_->local_and_includes(), std::runtime_error,
            "Tile is not local or is not included in the array.");
        typename container_type::const_accessor acc;
        const bool found = tiles_.find(acc, i);
        TA_ASSERT(found, std::runtime_error,
            "A local tile search did not find a tile when it should have.");
        return *acc;
      }

      template <typename Index>
      data_type remote_find(const Index& i) const {
        if(shape_->local_and_includes())
          return local_find(i);

        madness::Future<typename container_type::iterator> fut_it =
            tiles_.find(i);
        madness::TaskAttributes attr();
        data_type tile = get_world().taskq.add(this, & find_handler, fut_it, attr);
      }

      /// Insert a tile into the array

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
      void set(const Index& i, InIter first, InIter last) {
        TA_ASSERT(shape_->local_and_inclues(i), std::range_error,
            "The given index i is not local or not included in the array shape.");

        std::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

        TA_ASSERT(volume_type(std::distance(first, last)) == r->volume(), std::runtime_error,
            "The number of elements in [first, last) is not equal to the tile volume.");

        typename container_type::accessor acc;
        bool found = tiles_.find(acc, i);
        TA_ASSERT(found, std::runtime_error, "The tile should be present,");
        acc->set(policy::construct_value(i, r, first, last));
      }


      /// Insert a tile into the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param v The value that will be used to initialize the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      void set(const Index& i, const T& v = T()) {
        TA_ASSERT(shape_->local_and_inclues(i), std::runtime_error,
            "The given index i is not local and included in the array.");

        std::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

        typename container_type::accessor acc;
        bool found = tiles_.find(acc, i);
        TA_ASSERT(found, std::runtime_error, "The tile should be present,");
        acc->set(policy::construct_value(i, r, v));
      }

      /// Insert a tile into the array

      /// \tparam Index The type of the index (valid types are: Array::index or
      /// Array::ordinal_index)
      /// \tparam InIter Input iterator type for the data
      /// \param i The index where the tile will be inserted
      /// \param t The value that will be used to initialize the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      void set(const Index& i, const value_type& t) {
        TA_ASSERT(shape_->local_and_inclues(i), std::runtime_error,
            "The given index i is not local and included in the array.");

        std::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

        typename container_type::accessor acc;
        bool found = tiles_.find(acc, i);
        TA_ASSERT(found, std::runtime_error, "The tile should be present,");
        acc->set(t);
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
      const pmap_type& pmap() const { return pmap_; }

      /// Shape accessor
      const shape_type& get_shape() const { return shape_; }

      /// World accessor
      madness::World& get_world() const { return tiles_.get_world(); }

    private:


      /// Initialize the array container by inserting local tiles.
      void initialize_() {
        for(typename tiled_range_type::range_type::const_iterator it =
            tiled_range_.tiles().begin(); it != tiled_range_.tiles().end(); ++it) {
          if(shape_->is_local_and_includes(*it))
            tiles_.replace(key_(*it), data_type());
        }

        tiles_.process_pending();
      }

      /// Calculate the ordinal index

      /// \param i The ordinal index to convert
      /// \return The ordinal index of \c i
      /// \note This function is a pass through function. It is only here to make
      /// life easier with templates.
      /// \note No range checking is done in this function.
      ordinal_index ord_(const ordinal_index& i) const { return i; }

      /// Calculate the ordinal index

      /// \param i The coordinate index to convert
      /// \return The ordinal index of \c i
      /// \note No range checking is done in this function.
      ordinal_index ord_(const index& i) const {
        return coordinate_system::calc_ordinal(i, tiled_range_.tiles().weight(),
            tiled_range_.tiles().start());
      }

      /// Calculate the ordinal index

      /// \param k The key to convert to an ordinal index
      /// \return The ordinal index of \c k
      /// \note No range checking is done in this function.
      ordinal_index ord_(const key_type& k) const {
        if((k.keys() & 1) == 0)
          return ord_(k.key2());

        return k.key1();
      }

      /// Construct a complete key

      /// \c k may contain key1, key2, or both. If one of the keys is missing, it
      /// is added to the key before being returned. If both keys are present, it
      /// is returned as is.
      /// \param k The key to convert to a complete key
      /// \return A key that contains both key1 and key2
      key_type key_(const key_type& k) const {
        return coordinate_system::key(k, tiled_range_.tiles().weight(),
            tiled_range_.tiles().start());
      }

      value_type find_handler(const typename container_type::iterator& it) const {
        if(it == tiles_.end())
          return value_type();

        return it->second;
      }

      tiled_range_type tiled_range_; ///< Tiled range object
      pmap_type pmap_;     ///< Versioned process map
      std::shared_ptr<shape_type> shape_;                             ///< Pointer to the shape object
      container_type tiles_;                          ///< Distributed container that holds tiles
    }; // class ArrayImpl

  } // detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
