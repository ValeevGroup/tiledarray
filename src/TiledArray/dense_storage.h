/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_TILE_BASE_H__INCLUDED
#define TILEDARRAY_TILE_BASE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/math/math.h>
#include <world/archive.h>
#include <Eigen/Core>
#include <cstddef>

namespace TiledArray {

  namespace detail {

    template <typename T, typename P>
    struct optimized_init : public std::false_type { };

    template <typename T>
    struct optimized_init<T, T*> : public detail::is_numeric<T> { };

    template <typename T>
    struct optimized_init<T, const T*> : public detail::is_numeric<T> { };


    template <typename T, typename P1, typename P2>
    struct optimized_pair_init : public std::false_type { };

    template <typename T>
    struct optimized_pair_init<T, T*, T*> : public detail::is_numeric<T> { };

    template <typename T>
    struct optimized_pair_init<T, const T*, T*> : public detail::is_numeric<T> { };

    template <typename T>
    struct optimized_pair_init<T, T*, const T*> : public detail::is_numeric<T> { };

    template <typename T>
    struct optimized_pair_init<T, const T*, const T*> : public detail::is_numeric<T> { };

  } // namespace detail

  /// DenseStorage is an N-dimensional, dense array.

  /// \tparam T DenseStorage element type.
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
          construct_iterator(other.begin());
        } catch(...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }

    /// Constructs dense storage object

    /// The storage object will contain \c n elements that have \c val , and are
    /// allocated with the allocator \c a .
    /// \param n The number of elements to be stored
    /// \param a The allocator object for the tile data ( default: allocator_type() )
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    /// \throw anything Any exception that can be thrown by \c T type default or
    /// copy constructors
    explicit DenseStorage(size_type n, const allocator_type& a = allocator_type()) :
      allocator_type(a),
      first_(NULL),
      last_(NULL)
    {
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          default_init(detail::is_numeric<value_type>());
        } catch (...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }

    /// Constructs dense storage object

    /// The storage object will contain \c n elements that have \c val , and are
    /// allocated with the allocator \c a .
    /// \param n The number of elements to be stored
    /// \param val The fill value for the new tile elements
    /// \param a The allocator object for the tile data [default: allocator_type()]
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    /// \throw anything Any exception that can be thrown by \c T type default or
    /// copy constructors
    DenseStorage(size_type n, const value_type& val, const allocator_type& a = allocator_type()) :
      allocator_type(a),
      first_(NULL),
      last_(NULL)
    {
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          construct_value(val);
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
          construct_iterator(first);
        } catch (...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }

    /// Constructs a new tile

    /// This constructor will allocate memory for \c n elements. Each element
    /// will be initialized as:
    /// \code
    /// for(int i = 0; i < n; ++i)
    ///    data[i] = op(*first++);
    /// \endcode
    /// \tparam InIter An input iterator type for the argument.
    /// \tparam Op A unary operation type
    /// \param n the size of the storage area
    /// \param first An input iterator for the argument
    /// \param op The unary operation to be applied to the argument data
    /// \param a The allocator object for the tile data ( default: allocator_type() )
    /// \throw std::bad_alloc There is not enough memory available for the
    /// target tile
    template <typename InIter, typename Op>
    DenseStorage(size_type n, InIter first, const Op& op, const allocator_type& a = allocator_type(),
      typename madness::enable_if_c<detail::is_iterator<InIter>::value &&
        ! detail::is_iterator<Op>::value, Enabler>::type = Enabler()) :
      allocator_type(a),
      first_(NULL),
      last_(NULL)
    {
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          construct_unary_op(first, op);
        } catch (...) {
          allocator_type::deallocate(first_, n);
          throw;
        }
      }
    }

    /// Constructs a new tile

    /// This constructor will allocate memory for \c n elements. Each element
    /// will be initialized as:
    /// \code
    /// for(int i = 0; i < n; ++i)
    ///    data[i] = op(*it1++, *it2++);
    /// \endcode
    /// \tparam InIter1 An input iterator type for the left-hand argument.
    /// \tparam InIter2 An input iterator type for the right-hand argument.
    /// \tparam Op A binary operation type
    /// \param n the size of the storage area
    /// \param it1 An input iterator for the left-hand argument
    /// \param it2 An input iterator for the right-hand argument
    /// \param op The binary operation to be applied to the argument data
    /// \param a The allocator object for the tile data ( default: allocator_type() )
    /// \throw std::bad_alloc There is not enough memory available for the
    /// target tile
    template <typename InIter1, typename InIter2, typename Op>
    DenseStorage(size_type n, InIter1 it1, InIter2 it2, const Op& op,
        const allocator_type& a = allocator_type(),
        typename madness::enable_if_c<detail::is_iterator<InIter1>::value &&
            detail::is_iterator<InIter2>::value, Enabler>::type = Enabler()) :
      allocator_type(a),
      first_(NULL),
      last_(NULL)
    {
      TA_ASSERT(n < allocator_type::max_size());
      if(n) {
        first_ = allocator_type::allocate(n);
        last_ = first_ + n;
        try {
          construct_binary_op(it1, it2, op);
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

    /// \return The number of elements stored
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

      // Deallocate existing memory
      if(first_ != NULL) {
        const std::size_t size = last_ - first_;
        if(size < n) {
          destroy_(*this, first_, last_);
          allocator_type::deallocate(first_, size);

          // Allocate new memory
          first_ = allocator_type::allocate(n);
          last_ = first_ + n;
          default_init(detail::is_numeric<value_type>());
        } else if(size > n) {
          value_type* const new_last = first_ + n;
          destroy_(*this, new_last, last_);
          last_ = new_last;
        }
      }

      ar & madness::archive::wrap(first_, n);
    }

    template <typename Archive>
    void store(const Archive& ar) const {
      ar & size() & madness::archive::wrap(first_, size());
    }

  private:

    void default_init(std::true_type) { }


    void default_init(std::false_type) {
      const value_type temp = value_type();
      for(pointer it = first_; it != last_; ++it)
        allocator_type::construct(it, temp);
    }

    /// Initialize tile data with an input iterator

    /// \tparam InIter Input iterator type
    /// \param in_it Input iterator to the initialization data
    template <typename InIter>
    typename madness::disable_if<detail::optimized_init<value_type, InIter> >::type
    construct_iterator(InIter in_it) {
      pointer it = first_;
      try {
        for(; it != last_; ++it)
          allocator_type::construct(it, *in_it++);
      } catch (...) {
        destroy_(*this, first_, it);
        throw;
      }
    }

    /// Initialize data with an input iterator

    /// \param in_it Input iterator to the initialization data
    void construct_iterator(const_pointer in_it) {
      memcpy(first_, in_it, sizeof(value_type) * (last_ - first_));
    }

    /// Initialize data with an input iterator + unary operation

    /// Each element is initialized with: \c op(*in_it)
    /// \tparam InIter Input iterator type
    /// \param in_it Input iterator to the initialization data
    /// \param op The operation to be applied to the input
    template <typename InIter, typename Op>
    typename madness::disable_if<detail::optimized_init<value_type, InIter> >::type
    construct_unary_op(InIter in_it, const Op& op) {
      pointer it = first_;
      try {
        for(; it != last_; ++it)
          allocator_type::construct(it, op(*in_it++));
      } catch (...) {
        destroy_(*this, first_, it);
        throw;
      }
    }

    /// Initialize data with an input iterator + unary operation

    /// Each element is initialized with: \c op(*in_it)
    /// \tparam InIter Input iterator type
    /// \param in_it Input iterator to the initialization data
    /// \param op The operation to be applied to the input
    template <typename InIter, typename Op>
    typename madness::enable_if<detail::optimized_init<value_type, InIter> >::type
    construct_unary_op(InIter in_it, const Op& op) {
      math::vector_op(last_ - first_, in_it, first_, op);
    }

    /// Initialize data with an input iterator + unary operation

    /// Each element is initialized with: \c op(*in_it1, *in_it2)
    /// \tparam InIter1 An input iterator type for the left-hand argument.
    /// \tparam InIter2 An input iterator type for the right-hand argument.
    /// \tparam Op A binary operation type
    /// \param in_it1 An input iterator for the left-hand argument
    /// \param in_it2 An input iterator for the right-hand argument
    /// \param op The operation to be applied to the input data
    template <typename InIter1, typename InIter2, typename Op>
    typename madness::disable_if_c<
        detail::optimized_init<value_type, InIter1>::value &&
        detail::optimized_init<value_type, InIter2>::value >::type
    construct_binary_op(InIter1 in_it1, InIter2 in_it2, const Op& op) {
      pointer it = first_;
      try {
        for(; it != last_; ++it)
          allocator_type::construct(it, op(*in_it1++, *in_it2++));
      } catch (...) {
        destroy_(*this, first_, it);
        throw;
      }
    }

    /// Initialize data with an input iterator + unary operation

    /// Each element is initialized with: \c op(*in_it1,*in_it2)
    /// \tparam InIter1 An input iterator type for the left-hand argument.
    /// \tparam InIter2 An input iterator type for the right-hand argument.
    /// \tparam Op A binary operation type
    /// \param in_it1 An input iterator for the left-hand argument
    /// \param in_it2 An input iterator for the right-hand argument
    /// \param op The operation to be applied to the input data
    template <typename InIter1, typename InIter2, typename Op>
    typename madness::enable_if_c<
        detail::optimized_init<value_type, InIter1>::value &&
        detail::optimized_init<value_type, InIter2>::value >::type
    construct_binary_op(InIter1 in_it1, InIter2 in_it2, const Op& op) {
      math::vector_op(last_ - first_, in_it1, in_it2, first_, op);
    }

    void construct_value(const value_type& value) {
      pointer it = first_;
      try {
        for(; it != last_; ++it)
          allocator_type::construct(it, value);
      } catch (...) {
        destroy_(*this, first_, it);
        throw;
      }
    }

    /// Call the destructor for a range of data.

    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    static void destroy_(allocator_type& alloc, pointer first, pointer last) {
      // NOTE: destroy_aux_ is necessary because destroy has no template parameters.
      destroy_aux_(alloc, first, last, detail::is_numeric<value_type>());
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a non-trivial destructor function.
    /// \param first A pointer to the first element in the memory range to destroy
    /// \param last A pointer to one past the last element in the memory range to destroy
    /// \throw nothing
    static void destroy_aux_(allocator_type& alloc, pointer first, pointer last, std::false_type) {
      for(; first != last; ++first)
        alloc.destroy(first);
    }

    /// Call the destructor for a range of data.

    /// This is a helper function for data with a trivial destructor functions.
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
