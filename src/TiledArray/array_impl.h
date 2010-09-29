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
    class ArrayImpl : private boost::noncopyable {
    private:
      typedef madness::WorldObject<ArrayImpl<T, CS, P> > WorldObject_;
      typedef P policy;
      typedef detail::Key<typename CS::ordinal_index, typename CS::index> key_type;
      typedef madness::Future<typename policy::value_type> data_type;
      typedef madness::WorldDCPmapInterface< key_type > pmap_interface_type;
      typedef madness::WorldContainer<key_type, data_type> container_type;
      typedef detail::VersionedPmap<key_type> pmap_type;
      typedef Shape<CS, key_type> shape_type;
      typedef DenseShape<CS, key_type> dense_shape_type;
      typedef SparseShape<CS, key_type> sparse_shape_type;

    public:
      typedef typename policy::value_type value_type; /// The array value type (i.e. tiles)
      typedef CS coordinate_system; ///< The array coordinate system
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
      ArrayImpl(madness::World& w, const tiled_range_type& tr, unsigned int v = 0) :
          tiled_range_(tr),
          pmap_(make_pmap_(w, v)),
          shape_(NULL),
          tiles_(w, pmap_, false)
      { }

      /// Set the array shape.
      madness::Void set_shape(const madness::SharedPtr<pmap_interface_type>& s) {
        TA_ASSERT(shape_.get() == NULL, std::runtime_error,
            "Array shape has already been set.");

        // Set the array shape and add local tiles to the distributed container.
        shape_ = s;
        initialize_();

        return madness::None;
      }

      /// Version number accessor

      /// \return The current version number
      unsigned int version() const {
        // Todo: This has a runtime hit taht we don't want, but it can't be avoided
        // until madness shared pointers change to tr1 versions.
        pmap_type* pmap = dynamic_cast<pmap_type*>(pmap_.get());
        return pmap->version();
      }


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

        boost::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

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

        boost::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

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
      /// \param v The value that will be used to initialize the tile data
      /// \throw std::out_of_range When \c i is not included in the array range
      /// \throw std::range_error When \c i is not included in the array shape
      template <typename Index>
      void set(const Index& i, const value_type& t) {
        TA_ASSERT(shape_->local_and_inclues(i), std::runtime_error,
            "The given index i is not local and included in the array.");

        boost::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

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
      const madness::SharedPtr< pmap_interface_type >& get_pmap() const { return shape_->pmap(); }

      /// Shape accessor
      const boost::shared_ptr<shape_type>& get_shape() const { return shape_; }

      /// World accessor
      madness::World& get_world() const { return tiles_.get_world(); }

      /// Construct a dense shape

      /// \param r The tile range object
      /// \param pmap The array process map
      static madness::SharedPtr<pmap_interface_type>
      make_shape(const typename tiled_range_type::tile_range_type& r) {
#ifdef NDEBUG
        // Optimize away the dynamic cast checking
        madness::SharedPtr<shape_type> result(static_cast<shape_type*>(new dense_shape_type(r)));
#else
        madness::SharedPtr<shape_type> result();
        dense_shape_type* s = new dense_shape_type(r);
        try {
          result = madness::SharedPtr<shape_type>(dynamic_cast<shape_type*>(s));
          if(result.get() == NULL)
            throw std::bad_cast("Shape cast failed.");
        } catch(...) {
          delete s;
          throw;
        }
#endif
        return result;
      }

      /// Construct a sparse shape

      /// \tparam InIter Input iterator type for tile list
      /// \param pmap The array process map
      /// \param first An input iterator pointing to the first element in a list
      /// of tiles to be added to the shape.
      /// \param last An input iterator pointing to the last element in a list of
      /// tiles to be added to the shape.
      /// \note InIter::value_type may be Array::index, Array::ordinal_index, or
      /// Array::key_type types.
      template <typename InIter>
      static madness::SharedPtr<pmap_interface_type>
      make_shape(madness::World& w, const typename tiled_range_type::tile_range_type& r,
          const madness::SharedPtr<pmap_interface_type>& pmap, InIter first, InIter last) {
#ifdef NDEBUG
        // Optimize away the dynamic cast checking
        madness::SharedPtr<shape_type> result(
            static_cast<shape_type*>(new sparse_shape_type(w, r, pmap, first, last)));
#else
        madness::SharedPtr<shape_type> result;
        sparse_shape_type* s = new sparse_shape_type(w, r, pmap, first, last);
        try {
          result = madness::SharedPtr<shape_type>(dynamic_cast<shape_type*>(s));
          if(result.get() == NULL)
            throw std::bad_cast("Shape cast failed.");
        } catch(...) {
          delete s;
          throw;
        }
#endif
        return result;
      }

      /// Construct a predicated shape

      /// \tparam Pred The predicate type
      /// \param pmap The array process map
      /// \param p The predicate used to construct the predicated shape
      template <typename Pred>
      static madness::SharedPtr<pmap_interface_type>
      make_shape(const typename tiled_range_type::tile_range_type& r, const Pred& p) {
#ifdef NDEBUG
        // Optimize away the dynamic_cast runtime check
        madness::SharedPtr<shape_type> result(
            static_cast<shape_type*>(new PredShape<coordinate_system, key_type, Pred>(r, p)));
#else
        madness::SharedPtr<shape_type> result;
        sparse_shape_type* s = new PredShape<coordinate_system, key_type, Pred>(r, p);
        try {
          result = madness::SharedPtr<shape_type>(dynamic_cast<shape_type*>(s));
          if(result.get() == NULL)
            throw std::bad_cast("Shape cast failed.");
        } catch(...) {
          delete s;
          throw;
        }
#endif
        return result;
      }

    private:

      static madness::SharedPtr<pmap_interface_type>
      make_pmap_(madness::World& w, unsigned int v) {
#ifdef NDEBUG
        // Optimize away the dynamic_cast runtime check
        madness::SharedPtr<pmap_interface_type> result(
            static_cast<pmap_interface_type*>(new pmap_type(w, v)));

#else
        madness::SharedPtr<pmap_interface_type> result;
        pmap_type* p = new pmap_type(w, v);
        try {
          result = madness::SharedPtr<pmap_interface_type>(dynamic_cast<pmap_interface_type*>(p));
          if(result.get() == NULL)
            throw std::bad_cast("Shape cast failed.");
        } catch(...) {
          delete p;
          throw;
        }
#endif
        return result;
      }

      /// Initialize the array container by inserting local tiles.
      void initialize_() const {
        for(typename tiled_range_type::range_type::const_iterator it =
            tiled_range_.tiles().begin(); it != tiled_range_.tiles().end(); ++it) {
          if(shape_->is_local_and_includes(*it))
            tiles_.replace(*it, data_type());
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
        if(k.keys() == 1)
          return key_(k.key1());
        else if(k.keys() == 2)
          return key_(k.key2());

        return k;
      }


      /// Construct a complete key from an index

      /// \param k The index of the key
      /// \return A key that contains both key1 and key2
      key_type key_(const index& k) const {
        return key_type(coordinate_system::calc_ordinal(k,
            tiled_range_.tiles().weight(), tiled_range_.tiles().start()), k);
      }

      /// Construct a complete key from an ordinal index

      /// \param k The ordinal index of the key
      /// \return A key that contains both key1 and key2
      key_type key_(const ordinal_index& k) const {
        return key_type(k, coordinate_system::calc_index(k,
            tiled_range_.tiles().weight()));
      }

      value_type find_handler(const typename container_type::iterator& it) const {
        if(it == tiles_.end())
          return value_type();

        return it->second;
      }

      TiledRange<CS> tiled_range_;                      ///< Tiled range object
      madness::SharedPtr<pmap_interface_type> pmap_;    ///< Versioned process map
      boost::shared_ptr<shape_type> shape_;             ///< Pointer to the shape object
      container_type tiles_;                            ///< Distributed container that holds tiles
    };

  } // detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
