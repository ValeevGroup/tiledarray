#ifndef TILEDARRAY_TILE_BASE_H__INCLUDED
#define TILEDARRAY_TILE_BASE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <world/archive.h>
#include <Eigen/Core>
#include <cstddef>

namespace TiledArray {

  /// DenseStorage is an N-dimensional, dense array.

  /// \tparam T DenseStorage element type.
  /// \tparam CS A \c CoordinateSystem type
  /// \tparam A A C++ standard library compliant allocator (Default:
  /// \c Eigen::aligned_allocator<T>)
  template <typename T, typename A = Eigen::aligned_allocator<T> >
  class DenseStorage : private A {
  public:
    typedef DenseStorage<T,A> DenseStorage_;                          ///< This object's type

    typedef std::size_t size_type;                                    ///< Array volume type
    typedef A allocator_type;                                         ///< Allocator type
    typedef typename allocator_type::value_type value_type;           ///< Array element type
    typedef typename allocator_type::reference reference;             ///< Element reference type
    typedef typename allocator_type::const_reference const_reference; ///< Element reference type
    typedef typename allocator_type::pointer pointer;                 ///< Element pointer type
    typedef typename allocator_type::const_pointer const_pointer;     ///< Element const pointer type
    typedef typename allocator_type::difference_type difference_type; ///< Difference type
    typedef pointer iterator;                                         ///< Element iterator type
    typedef const_pointer const_iterator;                             ///< Element const iterator type

  private:
    struct Enabler { };

  public:

    /// Default constructor

    /// Constructs a tile with zero size.
    /// \note You must call resize() before attempting to access any elements.
    DenseStorage() : allocator_type(), first_(NULL), last_(NULL) { }

    /// Copy constructor

    /// \param other The tile to be copied.
    DenseStorage(const DenseStorage_& other) :
        allocator_type(other),
        first_(NULL),
        last_(NULL)
    {
      const size_type n = other.size();
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          construct_iterator(first_, last_, other.begin());
        } catch(...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }

    /// Constructs dense storage object

    /// The storage object will contain \c n elements that have \c val , and are
    /// allocated with the allocator \c a .
    /// \param r A shared pointer to the range object that will define the tile
    /// dimensions
    /// \param val The fill value for the new tile elements ( default: value_type() )
    /// \param a The allocator object for the tile data ( default: allocator_type() )
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    /// \throw anything Any exception that can be thrown by \c T type default or
    /// copy constructors
    DenseStorage(size_type n, const value_type& val = value_type(), const allocator_type& a = allocator_type()) :
        allocator_type(a),
        first_(NULL),
        last_(NULL)
    {
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          construct_value(first_, last_, val);
        } catch (...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }


    /// Constructs a new tile

    /// The tile will have the dimensions specified by the range object \c r and
    /// the elements of the new tile will be equal to \c v. The provided
    /// allocator \c a will allocate space for only for the tile data.
    /// \tparam InIter An input iterator type.
    /// \param n the size of the storage area
    /// \param first An input iterator to the beginning of the data to copy
    /// \param a The allocator object for the tile data ( default: allocator_type() )
    /// \throw std::bad_alloc There is not enough memory available for the
    /// target tile
    template <typename InIter>
    DenseStorage(size_type n, InIter first, const allocator_type& a = allocator_type(),
        typename madness::enable_if<detail::is_iterator<InIter>, Enabler>::type = Enabler()) :
        allocator_type(a),
        first_(NULL),
        last_(NULL)
    {
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          construct_iterator(first_, last_, first);
        } catch (...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }

    /// destructor
    ~DenseStorage() {
      destroy_(*this, first_, last_);
      allocator_type::deallocate(first_, size());
    }

    /// Assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    DenseStorage_& operator =(const DenseStorage_& other) {
      if(this != &other) // check for self assignment
        DenseStorage_(other).swap(*this);

      return *this;
    }

    template <typename Arg>
    DenseStorage_& operator+=(const Arg& other) {
      typename Arg::const_iterator it_other = other.begin();
      for(iterator it = begin(); it != end(); ++it, ++it_other)
        *it += *it_other;

      return *this;
    }

    template <typename Arg>
    DenseStorage_& operator-=(const Arg& other) {
      typename Arg::const_iterator it_other = other.begin();
      for(iterator it = begin(); it != end(); ++it, ++it_other)
        *it -= *it_other;

      return *this;
    }

    DenseStorage_& operator+=(const value_type& value) {
      for(iterator it = begin(); it != end(); ++it)
        *it += value;

      return *this;
    }

    DenseStorage_& operator-=(const value_type& value) {
      for(iterator it = begin(); it != end(); ++it)
        *it -= value;

      return *this;
    }

    DenseStorage_& operator*=(const value_type& value) {
      for(iterator it = begin(); it != end(); ++it)
        *it *= value;

      return *this;
    }

    /// Copy the content of this data into dest
    template <typename Dest>
    void eval_to(Dest& dest) const {
      TA_ASSERT(! empty());
      const size_type end = size();
      TA_ASSERT(dest.size() == end);
      for(size_type i = 0; i < end; ++i)
        dest[i] = operator[](i);
    }

    /// Returns a raw pointer to the array elements. Elements are ordered from
    /// least significant to most significant dimension.
    pointer data() { return first_; }

    /// Returns a constant raw pointer to the array elements. Elements are
    /// ordered from least significant to most significant dimension.
    const_pointer data() const { return first_; }

    // Iterator factory functions.
    iterator begin() {  return first_; }
    iterator end() { return last_; }
    const_iterator begin() const { return first_; }
    const_iterator end() const { return last_; }

    /// Returns a reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    reference at(size_type i) {
      if(i >= size())
        TA_EXCEPTION("Element is out of range.");

      return first_[i];
    }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    const_reference at(size_type i) const {
      if(i >= size())
        TA_EXCEPTION("Element is out of range.");

      return first_[i];
    }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    reference operator[](size_type i) { // no throw for non-debug
#ifdef NDEBUG
      return first_[i];
#else
      return at(i);
#endif
    }

    /// Returns a constant reference to element i. No error checking is performed.
    const_reference operator[](size_type i) const { // no throw for non-debug
#ifdef NDEBUG
      return first_[i];
#else
      return at(i);
#endif
    }

    /// DenseStorage size accessor

    /// \return The number of elemenst stored
    /// \throw nothing
    size_type size() const { return last_ - first_; }

    /// Empty storage check

    /// \return \c true when no data is stored, otherwise false.
    bool empty() const { return first_ == NULL; }

    /// Exchange the content of this object with other.

    /// \param other The other DenseStorage to swap with this object
    /// \throw nothing
    void swap(DenseStorage_& other) {
      std::swap<allocator_type>(*this, other);
      std::swap(first_, other.first_);
      std::swap(last_, other.last_);
    }

    template <typename Archive>
    void load(const Archive& ar) {
      size_type n = 0;
      ar & n;
      DenseStorage_ temp(n);
      ar & madness::archive::wrap(temp.data(), temp.size());
      temp.swap(*this);
    }

    template <typename Archive>
    void store(const Archive& ar) const {
      ar & size() & madness::archive::wrap(first_, size());
    }

  private:

    template <typename InIter>
    void construct_iterator(pointer first, pointer last, InIter data) {
      pointer it = first;
      try {
        for(; it != last; ++it, ++data)
          allocator_type::construct(it, *data);
      } catch (...) {
        destroy_(*this, first, it);
        throw;
      }
    }

    void construct_value(pointer first, pointer last, const value_type& value) {
      pointer it = first;
      try {
        for(; it != last; ++it)
          allocator_type::construct(it, value);
      } catch (...) {
        destroy_(*this, first, it);
        throw;
      }
    }

    /// Call the destructor for a range of data.

    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    static void destroy_(const allocator_type& alloc, pointer first, pointer last) {
      destroy_aux_(alloc, first, last, std::has_trivial_destructor<value_type>());
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a non-trivial destructor function.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    static void destroy_aux_(const allocator_type& alloc, pointer first, pointer last, std::false_type) {
      for(; first != last; ++first)
        alloc.destroy(&*first);
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a trivial destructor functions.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    static void destroy_aux_(const allocator_type&, pointer, pointer, std::true_type) { }

    template <class, class>
    friend struct madness::archive::ArchiveStoreImpl;
    template <class, class>
    friend struct madness::archive::ArchiveLoadImpl;

    pointer first_;     ///< Pointer to the beginning of the data range
    pointer last_;      ///< Pointer to the end of the data range
  }; // class DenseStorage

  template <typename T, typename A>
  void swap(DenseStorage<T, A> t1, DenseStorage<T, A> t2) {
    t1.swap(t2);
  }

} // namespace TiledArray

namespace madness {
  namespace archive {

    template <typename Archive, typename T>
    struct ArchiveStoreImpl;
    template <typename Archive, typename T>
    struct ArchiveLoadImpl;

    template <typename Archive, typename T, typename A>
    struct ArchiveStoreImpl<Archive, TiledArray::DenseStorage<T, A> > {
      static void store(const Archive& ar, const TiledArray::DenseStorage<T, A>& t) {
        t.store(ar);
      }
    };

    template <typename Archive, typename T, typename A>
    struct ArchiveLoadImpl<Archive, TiledArray::DenseStorage<T, A> > {

      static void load(const Archive& ar, TiledArray::DenseStorage<T, A>& t) {
        t.load(ar);
      }
    };
  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_TILE_BASE_H__INCLUDED
