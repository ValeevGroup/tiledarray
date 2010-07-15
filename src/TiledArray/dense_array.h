#ifndef TILEDARRAY_ARRAY_STORAGE_H__INCLUDED
#define TILEDARRAY_ARRAY_STORAGE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/range.h>
#include <TiledArray/permutation.h>
#include <Eigen/Core>
#include <boost/utility/enable_if.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/has_trivial_destructor.hpp>
#include <iterator>

namespace TiledArray {

  // Forward declarations
  template <typename, typename, typename>
  class Tile;
  template <typename T, typename CS, typename A>
  void swap(Tile<T, CS, A>&, Tile<T, CS, A>&);
  template <unsigned int DIM, typename T, typename CS, typename A>
  Tile<T,CS, A> operator ^(const Permutation<DIM>&, const Tile<T,CS, A>&);

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
        alloc_type(), range_(boost::make_shared<range_type>()), first_(NULL), last_(NULL)
    { }

    /// Copy constructor
    Tile(const Tile_& other) :
        alloc_type(other), range_(other.range_), first_(NULL), last_(NULL)
    {
      first_ = alloc_type::allocate(other.range_->volume());
      last_ = first_ + other.range_->volume();
      uninitialized_copy_(other.first_, other.last_, first_);
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

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move constructor

    /// \param other The tile object to move
    /// \throw anything Throws anything the allocator move/copy constructor can throw.
    Tile(Tile_&& other) :
        allocator_type(std::move(other)), range_(other.range_), first_(other.first_), last_(other.last_)
    {
      other.range_.reset();
      other.first_ = NULL;
      other.last_ = NULL;
    }

    /// Move assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw nothing
    Tile_& operator =(Tile_&& other) {
      swap(other);
      return *this;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Constructs a new tile

    /// The tile will have the dimensions specified by the range object \c r and
    /// the elements of the new tile will be equal to \c v. The provided
    /// allocator \c a will allocate space for only for the tile data.
    /// \param r A shared pointer to the range object that will define the tile
    /// dimensions
    /// \param v The fill value for the new tile elements ( default: value_type() )
    /// \param a The allocator object for the tile data ( default: alloc_type() )
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    /// \throw anything Any exception that can be thrown by \c T type default or
    /// copy constructors
    Tile(const boost::shared_ptr<range_type>& r, const value_type& v = value_type(), const alloc_type& a = alloc_type()) :
        alloc_type(a), range_(r), first_(NULL), last_(NULL)
    {
      first_ = alloc_type::allocate(r->volume());
      last_ = first_ + r->volume();
      uninitialized_fill_(first_, last_, v);
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
    Tile(const boost::shared_ptr<range_type>& r, InIter first, InIter last, const alloc_type& a = alloc_type()) :
        alloc_type(a), range_(r), first_(create_(first, last))
    {
      first_ = alloc_type::allocate(r->volume());
      last_ = first_ + r->volume();
      uninitialized_copy_(first, last, first_);
    }

    /// Destructor
    ~Tile() {
      destroy_(first_, last_);
      alloc_type::deallocate(first_, range_->volume());
    }

    /// In place permutation of tile elements.

    /// \param p A permutation object.
    /// \return A reference to this object
    Tile_& operator ^=(const Permutation<coordinate_system::dim>& p) {

      if(first_ != NULL) {
        Tile_ temp = p ^ (*this);
        swap(temp);
      }
      return *this;
    }

    /// Resize the array to the specified dimensions.

    /// \param r The range object that specifies the new size.
    /// \param val The value that will fill any new elements in the array
    /// ( default: value_type() ).
    /// \return A reference to this object.
    /// \note The current data common to both arrays is maintained.
    /// \note This function cannot change the number of tile dimensions.
    Tile_& resize(const boost::shared_ptr<range_type>& r, value_type val = value_type()) {
      Tile_ temp(r, val);
      if(first_ != NULL) {
        // replace Range with ArrayDim?
        range_type range_common = r & (*range_);

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

    /// Copy iterator range into an uninitialized memory

    /// This function is for data with a non-trivial copy operation (i.e. a
    /// a simple memory copy is not safe).
    /// \tparam InIter Input iterator type for the data to copy.
    /// \param first An input iterator to the beginning of the data to copy.
    /// \param last An input iterator to one past the end of the data to copy.
    /// \param result A pointer to the first element where the data will be copied.
    /// \return A pointer to the end of the copied data.
    /// \throw anything This function will rethrow any thing that is thrown
    /// by the T copy constructor.
    template<typename InIter>
    typename boost::disable_if<boost::has_trivial_copy<value_type>, pointer >::type
    uninitialized_copy_(InIter first, InIter last, pointer result) {
      pointer cur = result;
      try {
        for(; first != last; ++first, ++cur)
          alloc_type::construct(&*cur, *first);
      } catch(...) {
        destroy(result, cur);
        throw;
      }

      return cur;
    }

    /// Copy iterator range into an uninitialized memory

    /// This function is for data with a trivial copy operation (i.e. a simple
    /// memory copy is safe).
    /// \tparam InIter Input iterator type for the data to copy.
    /// \param first An input iterator to the beginning of the data to copy.
    /// \param last An input iterator to one past the end of the data to copy.
    /// \param result A pointer to the first element where the data will be copied.
    /// \return A pointer to the end of the copied data.
    /// \throw nothing
    template<typename InIter>
    typename boost::enable_if<boost::has_trivial_copy<value_type>, pointer >::type
    uninitialized_copy_(InIter first, InIter last, pointer result) {
      return std::copy(first, last, result);
    }

    /// Fill a range of memory with the given value.

    /// \param first A pointer to the first element in the memory range to fill
    /// \param last A pointer to one past the last element in the memory range to fill
    /// \param v The value to be copied into the memory range
    void uninitialized_fill_(pointer first, pointer last, const value_type& v) {
      uninitialized_fill_aux_(first, last, v, boost::has_trivial_copy<value_type>());
    }

    /// Fill a range of memory with the given value.

    /// This is a helper function for filling data with a non-trivial copy operation.
    /// \param first A pointer to the first element in the memory range to fill
    /// \param last A pointer to one past the last element in the memory range to fill
    /// \param v The value to be copied into the memory range
    /// \throw anything This function will rethrow any thing that is thrown
    /// by the T copy constructor.
    void uninitialized_fill_aux_(pointer first, pointer last, const value_type& v, boost::false_type) {
      ForIter cur = first;
      try {
        for(; n > 0; --n, ++cur)
          alloc_type::construct(&*cur, v);
      } catch(...) {
        destroy_(first, cur);
        throw;
      }
    }

    /// Fill a range of memory with the given value.

    /// This is a helper function for filling data with a trivial copy operation.
    /// \param first A pointer to the first element in the memory range to fill
    /// \param last A pointer to one past the last element in the memory range to fill
    /// \param v The value to be copied into the memory range
    /// \throw nothing
    void uninitialized_fill_aux_(pointer first, pointer last, const value_type& v, boost::true_type) {
      std::fill(first, last, v);
    }

    /// Call the destructor for a range of data.

    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    void destroy_(pointer first, pointer last) {
      destroy_aux_(first, last, boost::has_trivial_destructor<value_type>());
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a non-trivial destructor function.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    void destroy_aux_(pointer first, pointer last, boost::false_type) {
      for(; first != last; ++first)
        alloc_type::destroy(&*first);
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a trivial destructor functions.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    void destroy_aux_(pointer, pointer, boost::true_type) { }

    /// Exchange the content of this object with other.

    /// \param other The other Tile to swap with this object
    /// \throw nothing
    void swap(Tile_& other) {
      std::swap<alloc_type>(*this, other);
      boost::swap(range_, other.range_);
      std::swap(first_, other.first_);
      std::swap(last_, other.last_);
    }

    friend void TiledArray::swap<>(Tile_& first, Tile_& second);

    boost::shared_ptr<range_type> range_; ///< Shared pointer to the range data for this tile
    pointer first_;                       ///< Pointer to the beginning of the data range
    pointer last_;                        ///< Pointer to the end of the data range
  }; // class DenseArrayStorage


  /// Swap the data of the two arrays.
  template <typename T, typename CS, typename A>
  void swap(Tile<T, CS, A>& first, Tile<T, CS, A>& second) { // no throw
    first.swap(second);
  }

  /// Permutes the content of the n-dimensional array.
  template <typename T, typename CS, typename A>
  Tile<T,CS,A> operator ^(const Permutation<CS::dim>& p, const Tile<T,CS,A>& s) {
    Tile<T,CS,A> result(p ^ s.size());
    detail::Permute<Tile<T,CS,A> > f_perm(s);
    f_perm(p, result.begin(), result.end());

    return result;
  }

  /// ostream output orperator.
  template <typename T, typename CS, typename A>
  std::ostream& operator <<(std::ostream& out, const Tile<T, CS, A>& t) {
    typedef Tile<T,CS> tile_type;
    const typename tile_type::size_array& weight = t.data_.weight();

    out << "{";
    typename CS::const_iterator d ;
    typename tile_type::ordinal_index i = 0;
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(d =  CS::begin(), ++d; d != CS::end(); ++d) {
        if((i % weight[*d]) == 0)
          out << "{";
      }

      out << *it << " ";


      for(d = CS::begin(), ++d; d != CS::end(); ++d) {
        if(((i + 1) % weight[*d]) == 0)
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
        ar & static_cast<const TiledArray::Tile::alloc_type&>(t) & t.range()
            & wrap(t.data(), t.range().volume());
      }
    };

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveLoadImpl<Archive, TiledArray::Tile<T, CS, A> > {
      static void load(const Archive& ar, TiledArray::Tile<T, CS, A>& a) {
        if(a.first_ != NULL) {
          t.destroy_(a.first_, a.last_);
          t.deallocate(t.first_, t.range_.volume());
        }

        ar & static_cast<TiledArray::Tile::alloc_type&>(t);
        t.range_ = boost::make_shared<typename TiledArray::Tile<T, CS, A>::range_type>();
        ar & (* t.range_);
        t.allocate(t.first_, t.range_->volume());
        t.last_ = t.first_ + t.range_->volume();
        t.uninitialized_fill_(t.first_, t.last_, TiledArray::Tile::value_type());
        ar & wrap(t.first_, t.range_->volume());
      }
    };

  }
}

#endif // TILEDARRAY_ARRAY_STORAGE_H__INCLUDED
