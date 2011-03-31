#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/tile.h>
#include <TiledArray/array_impl.h>
#include <world/sharedptr.h>

namespace TiledArray {

  namespace detail {
    template <typename, typename>
    class DefaultArrayPolicy;
  }

  namespace math {
    template <typename, typename, typename, template <typename> class>
    class BinaryOp;

    template <typename, typename, template <typename> class>
    class UnaryOp;
  }



  /// An n-dimensional, tiled array

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

  private:

    template <typename, typename, typename, template <typename> class>
    friend class math::BinaryOp;

    template <typename, typename, template <typename> class>
    friend class math::UnaryOp;

  public:

    /// Dense array constructor

    ///
    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    Array(madness::World& w, const tiled_range_type& tr) :
        pimpl_(std::make_shared<impl_type>(w, tr, 0u))
    { }

    /// Sparse array constructor

    /// \tparam InIter Input iterator type
    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param first An input iterator that points to the a list of tiles to be
    /// added to the sparse array.
    /// \param last An input iterator that points to the last position in a list
    /// of tiles to be added to the sparse array.
    template <typename InIter>
    Array(madness::World& w, const tiled_range_type& tr, InIter first, InIter last) :
        pimpl_(std::make_shared<impl_type>(w, tr, first, last, 0u))
    { }

    /// Predicated array constructor

    /// \param w The world where the array will live.
    /// \param tr The tiled range object that will be used to set the array tiling.
    /// \param p The shape predicate.
    template <typename Pred>
    Array(madness::World& w, const tiled_range_type& tr, Pred p) :
        pimpl_(std::make_shared<impl_type>(w, tr, p, 0u))
    { }

    /// Array copy constructor

    /// Performs a shallow copy of \c other array
    /// \param other The array to be copied
    Array(const Array_& other) :
        pimpl_(other.pimpl_)
    { }

    /// Array assignment operator

    /// Performs a shallow copy of \c other array
    /// \param other The array to be copied
    /// \return A reference to this object
    Array_& operator=(const Array_& other) {
      pimpl_ = other.pimpl_;
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

    template <typename Index, typename InIter>
    void set(const Index& i, InIter first, InIter last) { pimpl_->set(i, first, last); }

    template <typename Index>
    void set(const Index& i, const T& v = T()) { pimpl_->set(i, v); }

    /// Tiled range accessor

    /// \return A const reference to the tiled range object for the array
    /// \throw nothing
    const tiled_range_type& tiling() const { return pimpl_->tiling(); }

    /// Tile range accessor

    /// \return A const reference to the range object for the array tiles
    /// \throw nothing
    const range_type& tiles() const { return pimpl_->tiles(); }

    /// Element range accessor

    /// \return A const reference to the range object for the array elements
    /// \throw nothing
    const tile_range_type& elements() const { return pimpl_->elements(); }

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

    madness::World& get_world() const { return pimpl_->get_world(); }

  private:

    void preassign(madness::World& w, const tiled_range_type& tr) {
      pimpl_ = std::make_shared<impl_type>(w, tr);
    }

    std::shared_ptr<impl_type> pimpl_;
  }; // class Array

  namespace detail {

    template <typename T, typename CS>
    class DefaultArrayPolicy {
    public:
      typedef Tile<T, CS> value_type;

    }; // class TilePolicy

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
