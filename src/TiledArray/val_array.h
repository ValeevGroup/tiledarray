/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  val_array.h
 *  Feb 17, 2014
 *
 */

#ifndef TILEDARRAY_SHARED_BUFFER_H__INCLUDED
#define TILEDARRAY_SHARED_BUFFER_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/size_array.h>

namespace TiledArray {
namespace detail {

/// Value array

/// This is minimal wrapper around a dynamically allocated array. The array
/// may be shared and includes a reference counter.
/// \tparam T The element type of the array
template <typename T>
class ValArray : private SizeArray<T> {
 public:
  typedef ValArray<T> ValArray_;                         ///< This object type
  typedef typename SizeArray<T>::size_type size_type;    ///< size type
  typedef typename SizeArray<T>::value_type value_type;  ///< Element type
  typedef typename SizeArray<T>::reference reference;    ///< Reference type
  typedef typename SizeArray<T>::const_reference
      const_reference;                             ///< Const reference type
  typedef typename SizeArray<T>::pointer pointer;  ///< Data pointer type
  typedef typename SizeArray<T>::const_pointer
      const_pointer;  ///< Const data pointer type
  typedef typename SizeArray<T>::difference_type
      difference_type;                               ///< Difference type
  typedef typename SizeArray<T>::iterator iterator;  ///< Iterator type
  typedef typename SizeArray<T>::const_iterator
      const_iterator;  ///< Const iterator type

  static const std::size_t alignment = TILEDARRAY_ALIGN_SIZE;

 private:
  /// The pointer to reference counter
  mutable madness::AtomicInt* counter_ = nullptr;

  template <typename U>
  typename std::enable_if<detail::is_scalar_v<U>>::type default_construct(
      const size_type, U* MADNESS_RESTRICT) {}

  template <typename U>
  typename std::enable_if<!detail::is_scalar_v<U>>::type default_construct(
      const size_type n, U* MADNESS_RESTRICT u) {
    size_type i = 0ul;
    try {
      for (; i < n; ++i) new (u + i) U();
    } catch (...) {
      math::destroy_vector(i, u);
      throw;
    }
  }

  void deallocate() {
    if (counter_) {
      const int count = (*counter_)--;
      if (count == 1) {
        math::destroy_vector(SizeArray<T>::size(), SizeArray<T>::begin());
        free(counter_);
      }
    }
  }

  void init(const size_type n) {
    typedef std::integral_constant<
        size_type,
        sizeof(madness::AtomicInt) +
            ((alignment - (sizeof(madness::AtomicInt) & (alignment - 1ul))) &
             ~alignment)>
        sizeof_aligned_atomic_int;

    // Allocate buffer
    void* buffer = nullptr;
    if (posix_memalign(
            &buffer, alignment,
            (n * sizeof(value_type)) + sizeof_aligned_atomic_int::value) != 0)
      throw std::bad_alloc();

    // Initialize counter
    counter_ = reinterpret_cast<madness::AtomicInt*>(buffer);
    new (counter_) madness::AtomicInt;
    *counter_ = 1;

    // Initialize the array
    pointer const array = reinterpret_cast<pointer>(
        reinterpret_cast<char*>(buffer) + sizeof_aligned_atomic_int::value);
    SizeArray<T>::set(array, n);
  }

 public:
  ValArray() = default;

  explicit ValArray(const size_type n) {
    init(n);
    default_construct(n, SizeArray<T>::data());
  }

  template <typename Value, typename std::enable_if<std::is_convertible<
                                value_type, Value>::value>::type* = nullptr>
  ValArray(const size_type n, const Value& value) {
    init(n);
    math::uninitialized_fill_vector(n, value, SizeArray<T>::data());
  }

  template <typename Arg>
  ValArray(const size_type n, const Arg* const arg) {
    init(n);
    math::uninitialized_copy_vector(n, arg, SizeArray<T>::data());
  }

  template <typename Arg, typename Op>
  ValArray(const size_type n, const Arg* MADNESS_RESTRICT const arg,
           const Op& op) {
    init(n);
    math::uninitialized_unary_vector_op(n, arg, SizeArray<T>::data(), op);
  }

  template <typename U, typename Op>
  ValArray(const ValArray<U>& arg, const Op& op) {
    init(arg.size());
    math::uninitialized_unary_vector_op(arg.size(), arg.data(),
                                        SizeArray<T>::data(), op);
  }

  template <typename Left, typename Right, typename Op>
  ValArray(const size_type n, const Left* MADNESS_RESTRICT const left,
           const Right* MADNESS_RESTRICT const right, const Op& op) {
    init(n);
    math::uninitialized_binary_vector_op(n, left, right, SizeArray<T>::data(),
                                         op);
  }

  template <typename U, typename V, typename Op>
  ValArray(const ValArray<U>& left, const ValArray<V>& right, const Op& op) {
    TA_ASSERT(left.size() == right.size());
    init(left.size());
    math::uninitialized_binary_vector_op(left.size(), left.data(), right.data(),
                                         SizeArray<T>::data(), op);
  }

  ValArray(const ValArray_& other)
      : SizeArray<T>(const_cast<pointer>(other.begin()),
                     const_cast<pointer>(other.end())),
        counter_(other.counter_) {
    if (counter_) (*counter_)++;
  }

  ValArray_& operator=(const ValArray_& other) {
    if (counter_ != other.counter_) {
      // Cache pointers from other
      madness::AtomicInt* const counter = other.counter_;
      pointer const first = const_cast<pointer>(other.begin());
      pointer const last = const_cast<pointer>(other.end());

      // Increment the reference counter for other
      if (counter) (*counter)++;

      // Destroy this object
      deallocate();

      // Set the data
      counter_ = counter;
      SizeArray<T>::set(first, last);
    }

    return *this;
  }

  ~ValArray() { deallocate(); }

  // Import SizeArray interface

  using SizeArray<T>::begin;
  using SizeArray<T>::end;
  using SizeArray<T>::rbegin;
  using SizeArray<T>::rend;
  using SizeArray<T>::operator[];
  using SizeArray<T>::at;
  using SizeArray<T>::front;
  using SizeArray<T>::back;
  using SizeArray<T>::size;
  using SizeArray<T>::empty;
  using SizeArray<T>::max_size;
  using SizeArray<T>::data;
  using SizeArray<T>::assign;
  using SizeArray<T>::binary;
  using SizeArray<T>::unary;
  using SizeArray<T>::reduce;
  using SizeArray<T>::row_reduce;
  using SizeArray<T>::col_reduce;
  using SizeArray<T>::outer;
  using SizeArray<T>::outer_fill;

  // ValArray wrappers for vector operations

  /// Binary vector operation

  /// Perform a binary vector operation where this array is the left-hand
  /// argument and arg is the right-hand argument. The data elements are
  /// modified by: <tt>op(*this[i], arg[i])</tt>.
  /// \tparam U The element type of \c arg
  /// \tparam Op The binary operation
  /// \param arg The right-hand argument
  /// \param op The binary operation
  /// \throw TiledArray::Exception When the size of \c arg is not equal to
  /// the size of this array.
  template <typename U, typename Op>
  void binary(const ValArray<U>& arg, const Op& op) {
    TA_ASSERT(arg.size() == SizeArray<T>::size());
    SizeArray<T>::binary(arg.data(), op);
  }

  /// Binary vector operation

  /// Perform a binary vector operation with \c left and \c right, and store
  /// the result in this array. The data elements are given by:
  /// <tt>*this[i] = op(left[i], right[i])</tt>.
  /// \tparam U The element type of \c arg
  /// \tparam Op The binary operation
  /// \param left The left-hand argument
  /// \param right The right-hand argument
  /// \param op The binary operation
  /// \throw TiledArray::Exception When the sizes of left and right are not
  /// equal to the size of this array.
  template <typename U, typename V, typename Op>
  void binary(const ValArray<U>& left, const ValArray<V>& right, const Op& op) {
    TA_ASSERT(left.size() == SizeArray<T>::size());
    TA_ASSERT(right.size() == SizeArray<T>::size());
    SizeArray<T>::binary(left.data(), right.data(), op);
  }

  /// Unary vector operation

  /// Perform a unary vector operation, and store the result in this array.
  /// The data elements are given by: <tt>*this[i] = op(arg[i])</tt>.
  /// \tparam U The element type of \c arg
  /// \tparam Op The binary operation
  /// \param arg The right-hand argument
  /// \param op The binary operation
  /// \throw TiledArray::Exception When the size of \c arg is not equal to
  /// the size of this array.
  template <typename U, typename Op>
  void unary(const ValArray<U>& arg, const Op& op) {
    TA_ASSERT(arg.size() == SizeArray<T>::size());
    SizeArray<T>::unary(arg.data(), op);
  }

  /// Binary reduce operation

  /// Binary reduction operation where this object is the left-hand
  /// argument type. The reduced result is computed by
  /// <tt>op(result, *this[i], arg[i])</tt>.
  /// \tparam U The element type of \c arg
  /// \tparam Result The result type of the reduction
  /// \tparam Op The reduction operation.
  /// \param arg The right-hand array argument
  /// \param result The initial value of the reduction
  /// \param op The binary reduction operation
  /// \return The reduced value
  /// \throw TiledArray::Exception When <tt>arg.size() != size()</tt>.
  template <typename U, typename Result, typename Op>
  Result reduce(const ValArray<U>& arg, Result& result, const Op& op) {
    TA_ASSERT(arg.size() == SizeArray<T>::size());
    return SizeArray<T>::reduce(arg.data(), result, op);
  }

  /// Reduce row operation

  /// Reduce rows of \c left to this array, where the size of rows is equal
  /// to \c right.size(). The reduced result is computed by
  /// <tt>op(*this[i], left[i][j], right[j])</tt>.
  /// \tparam U The element type of \c left
  /// \tparam V The element type of \c right
  /// \tparam Op The reduction operation
  /// \param left The array to be reduced of size \c size()*right.size()
  /// \param right The right-hand array
  /// \param op The reduction operation
  /// \throw TiledArray::Exception When <tt>left.size() != (size() *
  /// right.size())</tt>.
  template <typename U, typename V, typename Op>
  void row_reduce(const ValArray<U>& left, const ValArray<V>& right,
                  const Op& op) {
    TA_ASSERT(left.size() == (SizeArray<T>::size() * right.size()));
    SizeArray<T>::row_reduce(right.size(), left.data(), right.data(), op);
  }

  /// Reduce row operation

  /// Reduce rows of \c arg to this array, where a row have
  /// \c arg.size()/size() elements. The reduced result is computed by
  /// <tt>op(*this[i], arg[i][j])</tt>.
  /// \tparam U The element type of \c arg
  /// \tparam Op The reduction operation
  /// \param arg The array to be reduced
  /// \param op The reduction operation
  /// \throw TiledArray::Exception When <tt>(arg.size() % size()) != 0</tt>.
  template <typename U, typename Op>
  void row_reduce(const ValArray<U>& arg, const Op& op) {
    TA_ASSERT((arg.size() % SizeArray<T>::size()) == 0ul);
    SizeArray<T>::row_reduce(arg.size() / SizeArray<T>::size(), arg, op);
  }

  /// Reduce column operation

  /// Reduce columns of \c left to this array, where columns have
  /// \c right.size() elements. The reduced result is computed by
  /// <tt>op(*this[j], left[i][j], right[i])</tt>.
  /// \tparam U The element type of \c left
  /// \tparam V The element type of \c right
  /// \tparam Op The reduction operation
  /// \param left The array to be reduced of size \c size()*right.size()
  /// \param right The right-hand array
  /// \param op The reduction operation
  /// \throw TiledArray::Exception When <tt>left.size() != (size() *
  /// right.size())</tt>.
  template <typename U, typename V, typename Op>
  void col_reduce(const ValArray<U>& left, const ValArray<V>& right,
                  const Op& op) {
    TA_ASSERT(left.size() == (SizeArray<T>::size() * right.size()));
    SizeArray<T>::col_reduce(right.size(), left.data(), right.data(), op);
  }

  /// Reduce column operation

  /// Reduce columns of \c arg to this array, where a columns have
  /// \c arg.size()/size() elements. The reduced result is computed by
  /// <tt>op(*this[i], arg[i][j])</tt>.
  /// \tparam U The element type of \c arg
  /// \tparam Op The reduction operation
  /// \param arg The array to be reduced
  /// \param op The reduction operation
  /// \throw TiledArray::Exception When <tt>(arg.size() % size()) != 0</tt>.
  template <typename U, typename Op>
  void col_reduce(const ValArray<U>& arg, const Op& op) {
    TA_ASSERT((arg.size() % SizeArray<T>::size()) == 0ul);
    SizeArray<T>::col_reduce(arg.size() / SizeArray<T>::size(), arg, op);
  }

  /// Outer operation

  /// This function use two arrays, \c left and \c right,
  /// to modify this array ( which is treated as a matrix of size
  /// <tt>left.size() * right.size()</tt> ). Elements of this array are
  /// modified by <tt>op(*this[i][j], left[i], right[j])</tt>.
  /// \tparam U The left-hand argument type
  /// \tparam V The right-hand argument type
  /// \param left The left-hand array
  /// \param right The right-hand array
  /// \param op The outer operation
  /// \throw TiledArray::Exception When <tt>size() != (left.size() *
  /// right.size())</tt>.
  template <typename U, typename V, typename Op>
  void outer(const ValArray<U>& left, const ValArray<V>& right, const Op& op) {
    TA_ASSERT(SizeArray<T>::size() == (left.size() * right.size()));
    SizeArray<T>::outer(left.size(), right.size(), left.data(), right.data(),
                        op);
  }

  /// Outer fill operation

  /// This function use two arrays, \c left and \c right,
  /// to fill this array ( which is treated as a matrix of size
  /// <tt>left.size() * right.size()</tt> ). Elements of this array are
  /// filled by <tt>op(*this[i][j], left[i], right[j])</tt>.
  /// \tparam U The left-hand argument type
  /// \tparam V The right-hand argument type
  /// \param left The left-hand array
  /// \param right The right-hand array
  /// \param op The outer operation
  /// \throw TiledArray::Exception When <tt>size() != (left.size() *
  /// right.size())</tt>.
  template <typename U, typename V, typename Op>
  void outer_fill(const ValArray<U>& left, const ValArray<V>& right,
                  const Op& op) {
    TA_ASSERT(SizeArray<T>::size() == (left.size() * right.size()));
    SizeArray<T>::outer_fill(left.size(), right.size(), left.data(),
                             right.data(), op);
  }

  /// Outer operation

  /// This function use two arrays, \c left and \c right,
  /// to modify this array ( which is treated as a matrix of size
  /// <tt>left.size() * right.size()</tt> ). Elements of this array are
  /// modified by <tt>op(*this[i][j], left[i], right[j])</tt>.
  /// \tparam U The left-hand argument type
  /// \tparam V The right-hand argument type
  /// \param[in] left The left-hand array
  /// \param[in] right The right-hand array
  /// \param[out] a The array that will hold the result
  /// \param[in] op The outer operation
  /// \throw TiledArray::Exception When <tt>size() != (left.size() *
  /// right.size())</tt>.
  template <typename U, typename V, typename A, typename Op>
  void outer_fill(const ValArray<U>& left, const ValArray<V>& right,
                  const ValArray<A>& a, const Op& op) {
    TA_ASSERT(SizeArray<T>::size() == (left.size() * right.size()));
    TA_ASSERT(a.size() == SizeArray<T>::size());
    SizeArray<T>::outer_fill(left.size(), right.size(), left.data(),
                             right.data(), a.data(), op);
  }

  void swap(ValArray_& other) {
    std::swap(counter_, other.counter_);
    pointer const first = other.begin();
    pointer const last = other.end();
    other.set(begin(), end());
    SizeArray<T>::set(first, last);
  }

  // Comparison operators
  template <typename U>
  bool operator==(const ValArray<U>& other) const {
    return SizeArray<T>::operator==(other);
  }

  template <typename U>
  bool operator!=(const ValArray<U>& other) const {
    return SizeArray<T>::operator!=(other);
  }

  /// Serialization

  /// \tparam Archive An output archive type
  /// \param[out] ar an Archive object
  template <typename Archive,
            typename = std::enable_if_t<madness::is_output_archive_v<Archive>>>
  void serialize(Archive& ar) const {
    // need to write size first to be able to init when deserializing
    ar& size() & madness::archive::wrap(data(), size());
  }

  /// (De)serialization

  /// \tparam Archive An input archive type
  /// \param[out] ar an Archive object
  template <typename Archive,
            typename = std::enable_if_t<madness::is_input_archive_v<Archive>>>
  void serialize(Archive& ar) {
    size_t sz = 0;
    ar & sz;
    init(sz);
    ar& madness::archive::wrap(data(), size());
  }

};  // class ValArray

template <typename Char, typename CharTraits, typename T>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os, const ValArray<T>& val_array) {
  print_array(os, val_array);
  return os;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_SHARED_BUFFER_H__INCLUDED
