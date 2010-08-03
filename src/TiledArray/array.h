#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/key.h>
#include <TiledArray/shape.h>
#include <TiledArray/tile.h>
#include <TiledArray/indexed_iterator.h>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/has_virtual_destructor.hpp>

namespace TiledArray {

  template <typename>
  class Shape;

  namespace detail {
    template <typename, typename>
    class ArrayPolicy;
  }

  /// An n-dimensional, tiled array

  /// \tparam T The element type of for array tiles
  /// \tparam Coordinate system type
  /// \tparam Policy class for the array
  template <typename T, typename CS, typename P = detail::ArrayPolicy<T, CS> >
  class Array : madness::WorldObject<Array<T, CS, P> > {
  private:
    typedef madness::WorldObject<Array<T, CS, P> > WorldObject_;
    typedef P array_policy;
    typedef typename array_policy::key_type key_type;
    typedef typename array_policy::data_type data_type;
    typedef typename array_policy::container_type container_type;
    typedef typename array_policy::pmap_interface_type pmap_interface_type;

  public:
    typedef Array<T, CS, P> Array_; ///< This object's type
    typedef CS coordinate_system; ///< The array coordinate system

    typedef typename coordinate_system::volume_type volume_type; ///< Array volume type
    typedef typename coordinate_system::index index; ///< Array coordinate index type
    typedef typename coordinate_system::ordinal_index ordinal_index; ///< Array ordinal index type
    typedef typename coordinate_system::size_array size_array; ///< Size array type

    typedef typename array_policy::value_type value_type; ///< Tile type
    typedef typename array_policy::reference reference; ///< Reference to tile type
    typedef typename array_policy::const_reference const_reference; ///< Const reference to tile type

    typedef typename array_policy::tiled_range_type tiled_range_type; ///< Tile range type
    typedef typename array_policy::range_type range_type; ///< Range type for tiles
    typedef typename array_policy::tile_range_type tile_range_type; ///< Range type for elements

    typedef detail::IndexedIterator<typename container_type::iterator> iterator; ///< Local tile iterator
    typedef detail::IndexedIterator<typename container_type::const_iterator> const_iterator; ///< Local tile const iterator

  private:
    Array();
    Array(const Array_&);
    Array_& operator=(const Array_&);

  public:

    Array(madness::World& w, const tiled_range_type& tr, const boost::shared_ptr<Shape<CS> >& s) :
        WorldObject_(w), tiled_range_(tr), shape_(s), tiles_(array_policy::make_tile_container(w, s->pmap()))
    { }

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
    void insert(const Index& i, InIter first, InIter last) {
      TA_ASSERT(shape_->inclues(i), std::range_error,
          "The given index i is not included in the array shape.");

      boost::shared_ptr<tile_range_type> r = tiled_range_.make_tile_range(i);

      TA_ASSERT(volume_type(std::distance(first, last)) == r->volume(), std::runtime_error,
          "The number of elements in [first, last) is not equal to the tile volume.");
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
    void insert(const Index& i, value_type v = value_type()) {
      TA_ASSERT(tiled_range_.tiles().includes(i), std::runtime_error,
          "The given index i is not included in the array range.");
      TA_ASSERT(shape_->inclues(i), std::runtime_error,
          "The given index i is not included in the array shape.");

    }

    /// Tiled range accessor

    /// \return A const reference to the tiled range object for the array
    /// \throw nothing
    const tile_range_type& tiling() const { return tiled_range_; }

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
    const madness::SharedPtr< pmap_interface_type >& pmap() const{ return shape_->pmap(); }

    /// Create an annotated tile

    /// \param v A string with a comma-separated list of variables
    /// \return An annotated array object that references this array
    expressions::AnnotatedArray<Array_> operator ()(const std::string& v) {
      return expressions::AnnotatedArray<Array_>(*this,
          expressions::VariableList(v));
    }

    /// Create an annotated tile

    /// \param v A string with a comma-separated list of variables
    /// \return An annotated array object that references this array
    const expressions::AnnotatedArray<Array_> operator ()(const std::string& v) const {
      return expressions::AnnotatedArray<Array_>(* const_cast<Array_*>(this),
          expressions::VariableList(v));
    }

    /// Create an annotated tile

    /// \param v A variable list object
    /// \return An annotated array object that references this array
    expressions::AnnotatedArray<Array_> operator ()(const expressions::VariableList& v) {
      return expressions::AnnotatedArray<Array_>(*this, v);
    }

    /// Create an annotated tile

    /// \param v A variable list object
    /// \return An annotated array object that references this array
    const expressions::AnnotatedArray<Array_> operator ()(const expressions::VariableList& v) const {
      return expressions::AnnotatedArray<Array_>(* const_cast<Array_*>(this), v);
    }

  private:
    TiledRange<CS> tiled_range_;                      ///< Tiled range object
    boost::shared_ptr<Shape<ordinal_index> > shape_;  ///< Pointer to the shape object
    boost::shared_ptr<container_type> tiles_;         ///< Distributed container that holds tiles
  }; // class Array

  namespace detail {
    /// Defer cleanup of object

    /// This class is used to automate deferred cleanup of objects. as a the deletion functor with shared pointers. It
    /// will defer cleanup of the object until the next synchronization point.
    /// If \c D is a function pointer it must be \c void(*)(T*) (the default
    /// type) or a function object which defines \c D::operator()(T*).
    /// \tparam T The object type to be deleted (This type must be derived from
    /// madness::DeferredCleanupInterface and have a virtual destructor).
    /// \tparam D The deleter object for the (Default = \c void(*)(T*) )
    template<typename T, typename D = void(*)(T*)>
    class DeferedDelete {
    private:
      BOOST_STATIC_ASSERT( (boost::is_base_of<madness::DeferredCleanupInterface, T>::value) );
      BOOST_STATIC_ASSERT( (boost::has_virtual_destructor<T>::value) );

      struct Enabler { };

    public:
      typedef DeferedDelete<T,D> DeferedDelete_;
      typedef D delete_func;

      /// Constructor

      /// This constructor is used when \c D is a function pointer.
      /// \param w A \c World object that is responsible for the deferred clean-up
      /// \param d The deleter function for the T* pointer (Default = A function
      /// that calls \c delete when the pointer should be destroyed.).
      DeferedDelete(madness::World& w, delete_func d = &deleter,
          typename boost::enable_if<boost::is_same<delete_func, void(*)(T*)>, Enabler>::type = Enabler()) :
          world_(&w), deleter_(d)
      { }

      /// Constructor

      /// This constructor is used when \c D is a functor.
      /// \param w A \c World object that is responsible for the deferred clean-up
      /// \param d The deleter function for the T* pointer (Default = The
      /// default constructor of the deleter object).
      DeferedDelete(madness::World& w, delete_func d = delete_func(),
          typename boost::enable_if<boost::is_class<delete_func>, Enabler>::type = Enabler()) :
          world_(&w), deleter_(d)
      { }

      /// Copy constructor

      /// \param other The object to be copied
      DeferedDelete(const DeferedDelete_& other) :
          world_(other.world_), deleter_(other.deleter_)
      { }

      /// Assignment operator

      /// \param other The object to be copied
      DeferedDelete_& operator=(const DeferedDelete_& other) {
        deleter_ = other.deleter_;
        world_ = other.world_;
        return *this;
      }

      /// Cleanup function

      /// When this function is called (i.e. When the pointer is ready for
      /// cleanup), the pointer is put in a new shared pointer and place in a
      /// world object's list for deferred cleanup. The
      void operator()(T* p) {
        world_->deferred_cleanup(madness::SharedPtr<T>(p, deleter_));
      }

    private:
      /// Default cleanup function
      static void deleter(T* p) { delete p; }

      madness::World* world_; ///< The world object that handles cleanup deferment
      D deleter_;             ///< The deleter function that will cleanup the pointer
    }; // class DeferedDelete

    template <typename T, typename CS>
    class ArrayPolicy {
    public:
      typedef Tile<T, CS> value_type;
      typedef value_type& reference;
      typedef const value_type& const_reference;

      typedef detail::Key<typename CS::ordinal_index, typename CS::index> key_type;
      typedef madness::Future<value_type> data_type;
      typedef madness::WorldContainer<key_type, data_type> container_type;
      typedef madness::WorldDCPmapInterface< key_type > pmap_interface_type;
      typedef madness::WorldDCDefaultPmap<key_type> pmap_type;

      typedef TiledRange<CS> tiled_range_type; ///< Tile range type
      typedef typename tiled_range_type::range_type range_type; ///< Range type for tiles
      typedef typename tiled_range_type::tile_range_type tile_range_type; ///< Range type for elements

      static boost::shared_ptr<container_type> make_tile_container(madness::World& w,
          const madness::SharedPtr<pmap_type>& m)
      {
        return boost::shared_ptr<container_type>(
            new madness::WorldContainer<key_type, data_type>(w, m), DeferedDelete<container_type>(w));
      }

      template <typename Index, typename InIter>
      static void add_tile(const Index& i, const tiled_range_type& r,
          const container_type& c, InIter first, InIter last)
      {
        TA_ASSERT(c.is_local(key(i)), std::runtime_error,
            "Index i is not stored locally.");

        data_type data;
        typename container_type::accessor a;
        c.replace(make_key(i,r.tiles().start(), r.tiles().weight()), data);

      }

    private:
      template <typename InIter>
      static value_type make_tile(const boost::shared_ptr<tile_range_type>& r, InIter first, InIter last) {
        return value_type(r, first, last);
      }

      template <typename InIter>
      static value_type make_tile(const boost::shared_ptr<tile_range_type>& r, const T& v) {
        return value_type(r, v);
      }

      static key_type key(const typename CS::ordinal_index& i) {
        return key_type(i);
      }

      static key_type key(const typename CS::index& i) {
        return key_type(i);
      }

      static const key_type& key(const key_type& i) {
        return i;
      }

      static key_type make_key(const key_type& i, const typename range_type::index& s,
          const typename range_type::size_array& w)
      {
        switch(i.keys()) {
          case 1u:
            return make_key(i.key1(), s, w);
            break;
          case 2u:
            return make_key(i.key2(), s, w);
            break;
          case 3u:
            return i;
        }

        return key_type(); // keep the compiler happy
      }

      static key_type make_key(const typename range_type::index& i,
          const typename range_type::index& s, const typename range_type::size_array& w) {
        return key_type(CS::calc_ordinal(i - s, w), i);
      }

      static key_type make_key(const typename range_type::ordinal_index& i,
          const typename range_type::index& s, const typename range_type::size_array& w) {
        return key_type(i, CS::calc_index(i, w) + s);
      }

    }; // class TilePolicy

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
