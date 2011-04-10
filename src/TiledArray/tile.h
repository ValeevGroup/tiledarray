#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/range.h>
#include <TiledArray/annotated_array.h>
#include <TiledArray/type_traits.h>
#include <Eigen/Core>
#include <world/sharedptr.h>
#include <world/archive.h>
#include <iterator>
#include <string>
#include <algorithm>
#include <iosfwd>

namespace TiledArray {

  // Forward declarations
  template <typename, typename, typename>
  class Tile;
  template <typename T, typename CS, typename A>
  void swap(Tile<T, CS, A>&, Tile<T, CS, A>&);
  template <typename T, typename CS, typename A>
  Tile<T, CS, A> operator ^(const Permutation<CS::dim>&, const Tile<T, CS, A>&);

  namespace detail {

  } // namespace detail


  /// Tile is an N-dimensional, dense array.

  /// \tparam T The value type of the array.
  /// \tparam CS The coordinate system type (it must conform to TiledArray
  /// coordinate system requirements)
  /// \tparam A The allocator type that conforms to standard C++ allocator
  /// requirements (Default: Eigen::aligned_allocator<T>)
  template <typename T, typename CS, typename A = Eigen::aligned_allocator<T> >
  class Tile : private A {
  private:
    typedef A alloc_type;

  public:
    typedef Tile<T,CS> Tile_;                         ///< This object's type
    typedef CS coordinate_system;                     ///< The array coordinate system

    typedef typename CS::volume_type volume_type;     ///< Array volume type
    typedef typename CS::index index;                 ///< Array coordinate index type
    typedef typename CS::ordinal_index ordinal_index; ///< Array ordinal index type
    typedef typename CS::size_array size_array;       ///< Size array type

    typedef T value_type;                             ///< Array element type
    typedef T * iterator;                             ///< Element iterator type
    typedef const T * const_iterator;                 ///< Element const iterator type
    typedef T & reference;                            ///< Element reference type
    typedef const T & const_reference;                ///< Element reference type
    typedef typename alloc_type::pointer pointer;     ///< Element pointer type
    typedef typename alloc_type::const_pointer const_pointer; ///< Element const pointer type

    typedef Range<coordinate_system> range_type;      ///< Tile range type

  public:
    /// Default constructor

    /// Constructs a tile with zero size.
    /// \note You must call resize() before attempting to access any elements.
    Tile() :
        alloc_type(), range_(std::make_shared<range_type>()), first_(NULL), last_(NULL)
    { }

    /// Copy constructor

    /// \param other The tile to be copied.
    Tile(const Tile_& other) :
        alloc_type(other), range_(std::make_shared<range_type>(* other.range_)),
        first_(alloc_type::allocate(other.range_->volume())),
        last_(first_ + other.range_->volume())
    {
      try {
        std::uninitialized_copy(other.first_, other.last_, first_);
      } catch(...) {
        alloc_type::deallocate(first_, other.range_->volume());
        throw;
      }
    }

    /// Constructs a new tile

    /// The tile will have the dimensions specified by the range object \c r and
    /// the elements of the new tile will be equal to \c v. The provided
    /// allocator \c a will allocate space for only for the tile data.
    /// \param r A shared pointer to the range object that will define the tile
    /// dimensions
    /// \param val The fill value for the new tile elements ( default: value_type() )
    /// \param a The allocator object for the tile data ( default: alloc_type() )
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    /// \throw anything Any exception that can be thrown by \c T type default or
    /// copy constructors
    Tile(const std::shared_ptr<range_type>& r, const value_type& val = value_type(), const alloc_type& a = alloc_type()) :
        alloc_type(a), range_(r), first_(alloc_type::allocate(r->volume())),
        last_(first_ + r->volume())
    {
      try {
        std::uninitialized_fill(first_, last_, val);
      } catch (...) {
        alloc_type::deallocate(first_, r->volume());
        throw;
      }
    }


    /// Constructs a new tile

    /// The tile will have the dimensions specified by the range object \c r and
    /// the elements of the new tile will be equal to \c v. The provided
    /// allocator \c a will allocate space for only for the tile data.
    /// \tparam InIter An input iterator type.
    /// \param r A shared pointer to the range object that will define the tile
    /// dimensions
    /// \param first An input iterator to the beginning of the data to copy.
    /// \param last An input iterator to one past the end of the data to copy.
    /// \param a The allocator object for the tile data ( default: alloc_type() )
    /// \throw std::bad_alloc There is not enough memory available for the
    /// target tile
    /// \throw anything Any exceptions that can be thrown by \c T type default
    /// or copy constructors
    template <typename InIter>
    Tile(const std::shared_ptr<range_type>& r, InIter first, InIter last, const alloc_type& a = alloc_type()) :
        alloc_type(a), range_(r), first_(alloc_type::allocate(r->volume())),
        last_(first_ + r->volume())
    {
      try {
        TA_ASSERT(volume_type(std::distance(first, last)) == r->volume(), std::runtime_error,
            "The distance between initialization iterators must be equal to the tile volume.");
        std::uninitialized_copy(first, last, first_);
      } catch (...) {
        alloc_type::deallocate(first_, r->volume());
        throw;
      }
    }

    /// Assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    Tile_& operator =(const Tile_& other) {
      Tile_ temp(other);
      swap(temp);

      return *this;
    }

    /// destructor
    ~Tile() {
      destroy_(first_, last_);
      alloc_type::deallocate(first_, range_->volume());
    }

    /// In-place permutation of tile elements.

    /// \param p A permutation object.
    /// \return A reference to this object
    /// \warning This function modifies the shared range object.
    Tile_& operator ^=(const Permutation<coordinate_system::dim>& p) {
      typedef detail::UninitializedCopy<typename Tile<T,CS,A>::const_iterator, A> copy_op;

      const typename range_type::volume_type v = range_->volume();
      range_type r(p ^ *range_);

      // Allocate some space for the permuted data.
      pointer result_first = alloc_type::allocate(v);
      pointer result_last = result_first + v;

      try {
        // create a permuted copy of the tile data
        detail::Permute<CS, copy_op> f_perm(*range_, copy_op(first_, last_, *this));
        f_perm(p, result_first, result_last);

      } catch(...) {
        alloc_type::deallocate(result_first, range_->volume());
        throw;
      }

      // Swap the current range and data with the permuted copies.
      TiledArray::swap(*range_, r);
      std::swap(first_, result_first);
      std::swap(last_, result_last);

      // clean-up the temporary storage (which now holds the old data).
      destroy_(result_first, result_last);
      alloc_type::deallocate(result_first, range_->volume());

      return *this;
    }

    /// Resize the array to the specified dimensions.

    /// \param r The range object that specifies the new size.
    /// \param val The value that will fill any new elements in the array
    /// ( default: value_type() ).
    /// \return A reference to this object.
    /// \note The current data common to both arrays is maintained.
    /// \note This function cannot change the number of tile dimensions.
    Tile_& resize(const std::shared_ptr<range_type>& r, value_type val = value_type()) {
      Tile_ temp(r, val);
      if(first_ != NULL) {
        // replace Range with ArrayDim?
        range_type range_common = *r & (*range_);

        for(typename range_type::const_iterator it = range_common.begin(); it != range_common.end(); ++it)
          temp[ *it ] = operator[]( *it ); // copy common data.
      }
      swap(temp);
      return *this;
    }

    /// Returns a raw pointer to the array elements. Elements are ordered from
    /// least significant to most significant dimension.
    value_type * data() {
      return first_;
    }

    /// Returns a constant raw pointer to the array elements. Elements are
    /// ordered from least significant to most significant dimension.
    const value_type * data() const {
      return first_;
    }

    // Iterator factory functions.
    iterator begin() { // no throw
      return first_;
    }

    iterator end() { // no throw
      return last_;
    }

    const_iterator begin() const { // no throw
      return first_;
    }

    const_iterator end() const { // no throw
      return last_;
    }

    /// Returns a reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    reference at(const Index& i) {
      if(! range_->includes(i))
        throw std::out_of_range("DenseArrayStorage<...>::at(...): Element is not in range.");

      return first_[ord_(i)];
    }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    const_reference at(const Index& i) const {
      if(! range_->includes(i))
        throw std::out_of_range("DenseArrayStorage<...>::at(...) const: Element is not in range.");

      return first_[ord_(i)];
    }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    template <typename Index>
    reference operator[](const Index& i) { // no throw for non-debug
#ifdef NDEBUG
      return first_[ord_(i)];
#else
      return at(i);
#endif
    }

    /// Returns a constant reference to element i. No error checking is performed.
    template <typename Index>
    const_reference operator[](const Index& i) const { // no throw for non-debug
#ifdef NDEBUG
      return first_[ord_(i)];
#else
      return at(i);
#endif
    }

    /// Tile range accessor

    /// \return A const reference to the tile range object.
    /// \throw nothing
    const range_type& range() const { return *range_; }

    /// Tile range shared pointer accessor

    /// \return A shared pointer to the tile's range object.
    /// \throw nothing
    std::shared_ptr<range_type> range_ptr() const { return range_; }

    /// Create an annotated tile

    /// \param v A string with a comma-separated list of variables.
    expressions::AnnotatedArray<Tile_> operator ()(const std::string& v) {
      return expressions::AnnotatedArray<Tile_>(*this,
          expressions::VariableList(v));
    }

    /// Create an annotated tile

    /// \param v A string with a comma-separated list of variables.
    const expressions::AnnotatedArray<Tile_> operator ()(const std::string& v) const {
      return expressions::AnnotatedArray<Tile_>(* const_cast<Tile_*>(this),
          expressions::VariableList(v));
    }

    /// Create an annotated tile

    /// \param v A variable list object.
    expressions::AnnotatedArray<Tile_> operator ()(const expressions::VariableList& v) {
      return expressions::AnnotatedArray<Tile_>(*this, v);
    }

    /// Create an annotated tile

    /// \param v A variable list object.
    const expressions::AnnotatedArray<Tile_> operator ()(const expressions::VariableList& v) const {
      return expressions::AnnotatedArray<Tile_>(* const_cast<Tile_*>(this), v);
    }

    /// Exchange the content of this object with other.

    /// \param other The other Tile to swap with this object
    /// \throw nothing
    void swap(Tile_& other) {
      std::swap<alloc_type>(*this, other);
      std::swap(range_, other.range_);
      std::swap(first_, other.first_);
      std::swap(last_, other.last_);
    }

  private:

    /// Forwards the ordinal index.

    /// \param i An ordinal index
    /// \return The given ordinal index i.
    /// \throw nothing
    /// \note No range checking is done in this function.
    inline const ordinal_index& ord_(const ordinal_index& i) const { return i; }

    /// Convert an index to an ordinal index

    /// This function converts a coordinate index to the equivalent ordinal index.
    /// \param i index to be converted to an ordinal index
    /// \return The ordinal index of the index.
    /// \throw nothing
    /// \note No range checking is done in this function.
    inline ordinal_index ord_(const index& i) const {
      return coordinate_system::calc_ordinal(i, range_->weight(), range_->start());
    }

    /// Call the destructor for a range of data.

    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    void destroy_(pointer first, pointer last) {
      destroy_aux_(first, last, std::has_trivial_destructor<value_type>());
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a non-trivial destructor function.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    void destroy_aux_(pointer first, pointer last, std::false_type) {
      for(; first != last; ++first)
        alloc_type::destroy(&*first);
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a trivial destructor functions.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    void destroy_aux_(pointer, pointer, std::true_type) { }

    friend Tile_ operator ^ <>(const Permutation<coordinate_system::dim>&, const Tile_&);
    template <class, class>
    friend struct madness::archive::ArchiveStoreImpl;
    template <class, class>
    friend struct madness::archive::ArchiveLoadImpl;

    std::shared_ptr<range_type> range_; ///< Shared pointer to the range data for this tile
    pointer first_;                       ///< Pointer to the beginning of the data range
    pointer last_;                        ///< Pointer to the end of the data range
  }; // class Tile


  /// Swap the data of the two arrays.
  template <typename T, typename CS, typename A>
  void swap(Tile<T, CS, A>& first, Tile<T, CS, A>& second) { // no throw
    first.swap(second);
  }

  /// Permutes the content of the n-dimensional array.
  template <typename T, typename CS, typename A>
  Tile<T,CS,A> operator ^(const Permutation<CS::dim>& p, const Tile<T,CS,A>& t) {
    typedef detail::AssignmentOp<typename Tile<T,CS,A>::iterator, typename Tile<T,CS,A>::const_iterator> assign_op;

    std::shared_ptr<typename Tile<T,CS,A>::range_type> r =
        std::make_shared<typename Tile<T,CS,A>::range_type>(p ^ t.range());
    Tile<T,CS,A> result(r);

    // create a permuted copy of the tile data
    detail::Permute<CS, assign_op> f_perm(t.range(), assign_op(t.begin(), t.end()));
    f_perm(p, result.begin(), result.end());

    return result;
  }

  /// ostream output orperator.
  template <typename T, typename CS, typename A>
  std::ostream& operator <<(std::ostream& out, const Tile<T, CS, A>& t) {
    typedef Tile<T, CS, A> tile_type;
    typedef typename detail::CoordIterator<const typename tile_type::size_array,
        tile_type::coordinate_system::order>::iterator weight_iterator;

    typename tile_type::ordinal_index i = 0;
    weight_iterator weight_begin_1 = tile_type::coordinate_system::begin(t.range().weight()) + 1;
    weight_iterator weight_end = tile_type::coordinate_system::end(t.range().weight());
    weight_iterator weight_it;

    out << "{";
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(weight_it = weight_begin_1; weight_it != weight_end; ++weight_it) {
        if((i % *weight_it) == 0)
          out << "{";
      }

      out << *it << " ";

      for(weight_it = weight_begin_1; weight_it != weight_end; ++weight_it) {
        if(((i + 1) % *weight_it) == 0)
          out << "}";
      }
    }
    out << "}";
    return out;
  }

} // namespace TiledArray

namespace madness {
  namespace archive {

    template <class Archive, class T>
    struct ArchiveStoreImpl;
    template <class Archive, class T>
    struct ArchiveLoadImpl;

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveStoreImpl<Archive, TiledArray::Tile<T, CS, A> > {
      static void store(const Archive& ar, const TiledArray::Tile<T, CS, A>& t) {
        ar & static_cast<const typename TiledArray::Tile<T, CS, A>::alloc_type&>(t) & t.range()
            & wrap(t.data(), t.range().volume());
      }
    };

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveLoadImpl<Archive, TiledArray::Tile<T, CS, A> > {
      typedef TiledArray::Tile<T, CS, A> tile_type;

      static void load(const Archive& ar, tile_type& t) {
        if(t.first_ != NULL) {
          t.destroy_(t.first_, t.last_);
          t.deallocate(t.first_, t.range_->volume());
        }

        ar & static_cast<typename TiledArray::Tile<T, CS, A>::alloc_type&>(t);
        ar & (* t.range_);
        t.first_ = t.allocate(t.range_->volume());
        t.last_ = t.first_ + t.range_->volume();
        t.uninitialized_fill_(t.first_, t.last_, typename TiledArray::Tile<T, CS, A>::value_type());
        ar & wrap(t.first_, t.range_->volume());
      }
    };

    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, Eigen::aligned_allocator<T> > {
      static void store(const Archive&, const Eigen::aligned_allocator<T>&) { }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, Eigen::aligned_allocator<T> > {
      static void load(const Archive&, Eigen::aligned_allocator<T>&) { }
    };

    template <class Archive, typename T>
    struct ArchiveStoreImpl<Archive, std::allocator<T> > {
      static void store(const Archive&, const std::allocator<T>&) { }
    };

    template <class Archive, typename T>
    struct ArchiveLoadImpl<Archive, std::allocator<T> > {
      static void load(const Archive&, std::allocator<T>&) { }
    };
  }
}
#endif // TILEDARRAY_TILE_H__INCLUDED
