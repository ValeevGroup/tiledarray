#ifndef TILEDARRAY_ARRAY_IMPL_H__INCLUDED
#define TILEDARRAY_ARRAY_IMPL_H__INCLUDED

#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/indexed_iterator.h>
#include <TiledArray/versioned_pmap.h>
#include <TiledArray/dense_shape.h>
#include <TiledArray/sparse_shape.h>
#include <TiledArray/pred_shape.h>
#include <world/worldreduce.h>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

namespace TiledArray {
  namespace detail {

    template <typename T, typename CS, typename P >
    class ArrayImpl : public madness::WorldReduce<ArrayImpl<T,CS,P>, typename CS::ordinal_index>, private boost::noncopyable {
    private:
      typedef P policy;

    public:
      typedef CS coordinate_system; ///< The array coordinate system
      typedef typename policy::value_type value_type;

    private:
      typedef ArrayImpl<T, CS, P> ArrayImpl_;
      typedef madness::WorldObject<ArrayImpl_> WorldObject_;
      typedef madness::WorldReduce<ArrayImpl_, typename CS::ordinal_index> WorldReduce_;
      typedef typename coordinate_system::key_type key_type;
      typedef madness::Future<value_type> data_type;
      typedef madness::ConcurrentHashMap<key_type, data_type> container_type;
      typedef detail::VersionedPmap<key_type> pmap_type;
      typedef Shape<CS> shape_type;
      typedef DenseShape<CS> dense_shape_type;
      typedef SparseShape<CS> sparse_shape_type;

    public:
      typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
      typedef typename coordinate_system::index index; ///< Array coordinate index type
      typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
      typedef typename coordinate_system::size_array size_array; ///< Size array type
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
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new sparse_shape_type(w, tiled_range_.tiles(), pmap_, first, last))),
          tiles_()
      { initialize_(); }

      /// Dense array constructor

      /// \param w The world where the array will live.
      /// \param tr The tiled range object that will be used to set the array tiling.
      /// \param p The predicate for the array shape
      /// \param v The version number for the array
      template <typename Pred>
      ArrayImpl(madness::World& w, const tiled_range_type& tr, const Pred& p, unsigned int v) :
          WorldReduce_(w),
          tiled_range_(tr),
          pmap_(w.size(), v),
          shape_(static_cast<shape_type*>(new PredShape<coordinate_system, Pred>(tiled_range_.tiles(), pmap_, p))),
          tiles_()
      { initialize_(); }

      virtual ~ArrayImpl() { }

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

      /// Tile future accessor

      /// Search for and return a future to the tile.
      /// \return If found true, otherwise false.
      template <typename Index>
      madness::Future<value_type> local_find(const Index& i) const {
        key_type key = key_(i);
        TA_ASSERT(pmap_.owner(key) == get_world().rank(), std::runtime_error,
            "Do not do a remote find on local tiles.");
        TA_ASSERT(tiled_range_.tiles().includes(key), std::out_of_range,
            "Element is out of range.");

        typename container_type::const_iterator it = tiles_.find(key);

        TA_ASSERT(it == tiles_.end(), std::runtime_error,
            "A tile that should have been in the array was not found.");

        return it->second;
      }

      template <typename Index>
      madness::Future<value_type> remote_find(const Index& i) const {
        key_type key = key_(i);
        const ProcessID me = get_world().rank();
        const ProcessID dest = owner(key);
        TA_ASSERT(dest != me, std::runtime_error,
            "Do not do a remote find on local tiles.");
        TA_ASSERT(tiled_range_.tiles().includes(key), std::out_of_range,
            "Element is out of range.");

        // If the tile existence data is stored locally and shape says it does
        // not exist, then we return an empty tile.
        if(shape_->is_local(key))
          if(!shape_->probe(key))
            return madness::Future<value_type>(value_type());

        madness::Future<value_type> result;
        WorldObject_::send(dest, & find_handler, key, result.remote_ref(get_world()));
        return result;
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
      void set(const Index& i, InIter first, InIter last) {
        set_value(policy::construct_value(tiled_range_.make_tile_range(i), first, last));
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
      madness::Void set(const Index& i, const T& v) {
        set_value(i, policy::construct_value(tiled_range_.make_tile_range(i), v));
        return madness::None;
      }

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
          TA_ASSERT(shape_->probe(i), std::runtime_error,
              "The given index i is not included in the array shape.");

          typename container_type::accessor acc;
          const bool found = tiles_.find(acc, key_(i));
          TA_ASSERT(found, std::runtime_error, "The tile should be present,");
          acc->second.set(t);
        } else {
          WorldObject_::send(owner(i), & ArrayImpl_::template set_value<Index>, i, t);
        }

        return madness::None;
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
      ProcessID owner(const Index& i) const { return pmap_.owner(key_(i)); }

      template <typename Index>
      bool is_local(const Index& i) const { return owner(i) == get_world().rank(); }

      /// Shape accessor
      const shape_type& get_shape() const { return shape_; }

      using WorldObject_::get_world;

    private:

      bool insert_tile(const key_type& key) {
        TA_ASSERT(key.keys() & 3, std::runtime_error,
            "A full key must be used to insert a tile into the array.");
        TA_ASSERT(is_local(key), std::runtime_error,
            "Tile must be owned by this node.");

        std::pair<typename container_type::iterator, bool> result =
            tiles_.insert(typename container_type::datumT(key, data_type()));
        return result.second;
      }

      /// Initialize the array container by inserting local tiles.
      void initialize_() {
        ordinal_index o = 0;
        for(typename tiled_range_type::range_type::const_iterator it = tiled_range_.tiles().begin(); it != tiled_range_.tiles().end(); ++it, ++o) {
          key_type key(o, *it);
          if(is_local(key) && shape_->is_local(key)) {
            if(shape_->probe(key)) {
              bool success = insert_tile(key_(key));
              TA_ASSERT(success, std::runtime_error,
                  "For some reason the tile was not inserted into the container.");
            }
          }
        }

        WorldObject_::process_pending();
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
      template <typename Index>
      key_type key_(const Index& i) const {
        return coordinate_system::key(i, tiled_range_.tiles().weight(),
            tiled_range_.tiles().start());
      }

      value_type find_handler(const typename container_type::iterator& it) const {
        if(it == tiles_.end())
          return value_type();

        return it->second;
      }

      /// Handles find request
      void find_handler(const key_type& key, const madness::RemoteReference< madness::FutureImpl<value_type> >& ref) const {
        typename container_type::const_iterator it = tiles_.find(key);
        data_type result(ref);

        // Since the tile is local, shape will definitely contain local existence data.
        if(shape_->probe(key))
          result.set(local_find(key));
        else
          result.set(policy::construct_value());
      }

      tiled_range_type tiled_range_;        ///< Tiled range object
      pmap_type pmap_;                      ///< Versioned process map
      boost::scoped_ptr<shape_type> shape_; ///< Pointer to the shape object
      container_type tiles_;                ///< Distributed container that holds tiles
    }; // class ArrayImpl

  } // detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
