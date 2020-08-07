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

#ifndef TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED
#define TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED

#include <TiledArray/external/madness.h>
#include <cstddef>

namespace TiledArray {
namespace detail {

namespace {

/// Pointer proxy

/// This object is a proxy object for the iterator pointer dereference.
/// It is necessary for the transform iterators because they are
/// dereferenced to values, which is not compatible with the standard
/// iterator interface.
/// \tparam T The value type of the pointer proxy
template <typename T>
class PointerProxy {
 public:
  /// Constructor

  /// \param value The pointer value
  PointerProxy(const T& value) : value_(value) {}

  /// Arrow operator

  /// \return A pointer to the value
  T* operator->() const { return &value_; }

  /// Type conversion operator
  operator T*() const { return &value_; }

 private:
  mutable T value_;  ///< The value of the pointer
};                   // class PointerProxy
}  // namespace

/// Binary transform iterator

/// This iterator holds a pair of iterators that are transformed when
/// dereferenced with a binary transform object that is provided by the user.
/// The first iterator is the left-hand argument and the second is the
/// right-hand argument for the transform operator. The iterator dereferences
/// to the result type of the transform operations.
/// \tparam Iter1 First base iterator type of the transform iterator.
/// \tparam Iter2 Second base iterator type of the transform iterator.
/// \tparam Op The transform operator type.
template <typename Iter1, typename Iter2, typename Op>
class BinaryTransformIterator {
 private:
  // Give access to other Transform iterator types.
  template <typename, typename, typename>
  friend class BinaryTransformIterator;

 public:
  typedef std::ptrdiff_t difference_type;  ///< Difference type
  typedef typename madness::detail::result_of<Op>::type
      value_type;  ///< Iterator dereference value type
  typedef PointerProxy<value_type> pointer;  ///< Pointer type to iterator value
  typedef value_type reference;  ///< Reference type to iterator value
  typedef std::input_iterator_tag
      iterator_category;  ///< Iterator category type
  typedef BinaryTransformIterator<Iter1, Iter2, Op>
      this_type;                 ///< This object type
  typedef Iter1 base_iterator1;  ///< First base iterator type
  typedef Iter2 base_iterator2;  ///< Second base iterator type

  /// Constructor

  /// \param it1 First base iterator
  /// \param it2 Second base iterator
  /// \param op The transform operator
  BinaryTransformIterator(Iter1 it1, Iter2 it2, Op op = Op())
      : it1_(it1), it2_(it2), op_(op) {}

  /// Copy constructor

  /// \param other The transform iterator to copy
  BinaryTransformIterator(const this_type& other)
      : it1_(other.it1_), it2_(other.it2_), op_(other.op_) {}

  /// Copy conversion constructor

  /// This constructor allows copying when the base iterators are implicitly
  /// convertible with each other.
  /// \tparam It1 An iterator type that is implicitly convertible to
  /// \c base_iterator1 type.
  /// \tparam It2 An iterator type that is implicitly convertible to
  /// \c base_iterator2 type.
  /// \param other The transform iterator to copy
  /// \note The operation type must be the same for both transform iterators.
  template <typename It1, typename It2>
  BinaryTransformIterator(const BinaryTransformIterator<It1, It2, Op>& other)
      : it1_(other.it1_), it2_(other.it2_), op_(other.op_) {}

  /// Copy operator

  /// \param other The transform iterator to copy
  /// \return A reference to this object
  this_type& operator=(const this_type& other) {
    it1_ = other.it1_;
    it2_ = other.it2_;
    op_ = other.op_;

    return *this;
  }

  /// Copy conversion operator

  /// This operator allows copying when the base iterators are implicitly
  /// convertible with each other.
  /// \tparam It1 An iterator type that is implicitly convertible to
  /// \c base_iterator1 type.
  /// \tparam It2 An iterator type that is implicitly convertible to
  /// \c base_iterator2 type.
  /// \param other The transform iterator to copy
  /// \note The operation type must be the same for both transform iterators.
  template <typename It1, typename It2>
  this_type& operator=(const BinaryTransformIterator<It1, It2, Op>& other) {
    it1_ = other.it1_;
    it2_ = other.it2_;
    op_ = other.op_;

    return *this;
  }

  /// Prefix increment operator

  /// \return A reference to this object after it has been incremented.
  this_type& operator++() {
    increment();
    return *this;
  }

  /// Post-fix increment operator

  /// \return A copy of this object before it is incremented.
  this_type operator++(int) {
    this_type tmp(*this);
    increment();
    return tmp;
  }

  /// Equality operator

  /// \tparam It1 An iterator type that is implicitly convertible to
  /// \c base_iterator1 type.
  /// \tparam It2 An iterator type that is implicitly convertible to
  /// \c base_iterator2 type.
  /// \param other The iterator to compare to this iterator.
  /// \return True when both base iterators are equal to that of \c other,
  /// otherwise false.
  template <typename It1, typename It2>
  bool operator==(const BinaryTransformIterator<It1, It2, Op>& other) const {
    return equal(other);
  }

  /// Inequality operator

  /// \tparam It1 An iterator type that is implicitly convertible to
  /// \c base_iterator1 type.
  /// \tparam It2 An iterator type that is implicitly convertible to
  /// \c base_iterator2 type.
  /// \param other The iterator to compare to this iterator.
  /// \return True when both base iterators are not equal to that of \c other,
  /// otherwise false.
  template <typename It1, typename It2>
  bool operator!=(const BinaryTransformIterator<It1, It2, Op>& other) const {
    return !equal(other);
  }

  /// Dereference operator

  /// \return A transformed copy of the current base iterators.
  reference operator*() { return dereference(); }

  /// Arrow dereference operator

  /// \return A pointer to a transformed copy of the current base iterators.
  pointer operator->() { return pointer(dereference()); }

  /// First base iterator accessor

  /// \return A copy of the first base iterator.
  base_iterator1 base1() const { return it1_; }

  /// Second base iterator accessor

  /// \return A copy of the second base iterator.
  base_iterator2 base2() const { return it2_; }

 private:
  /// Increment the base iterators
  void increment() {
    ++it1_;
    ++it2_;
  }

  /// Compare base iterators

  /// \tparam It1 An iterator type that is implicitly convertible to
  /// \c base_iterator1 type.
  /// \tparam It2 An iterator type that is implicitly convertible to
  /// \c base_iterator2 type.
  /// \param other The iterator to compare to this iterator.
  /// \return True when the base iterators are equal to each other, otherwise
  /// false.
  template <typename It1, typename It2>
  bool equal(const BinaryTransformIterator<It1, It2, Op>& other) const {
    return (it1_ == other.it1_) && (it2_ == other.it2_);
  }

  /// Iterator dereference

  /// \return A transformed copy of the current base iterators.
  reference dereference() const { return op_(*it1_, *it2_); }

  base_iterator1 it1_;  ///< First base iterator
  base_iterator2 it2_;  ///< Second base iterator
  Op op_;               ///< Transform operation object
};                      // class BinaryTransformIterator

/// Unary transform iterator

/// This iterator holds an iterator that is transformed when dereferenced
/// with a unary transform object that is provided by the user. The iterator
/// dereferences to the result type of the transform operations.
/// \tparam Iter The base iterator type of the transform iterator.
/// \tparam Op The transform operator type.
template <typename Iter, typename Op>
class UnaryTransformIterator {
 private:
  // Give access to other Transform iterator types.
  template <typename, typename>
  friend class UnaryTransformIterator;

 public:
  typedef ptrdiff_t difference_type;  ///< Difference type
  typedef typename madness::detail::result_of<Op>::type
      value_type;  ///< Iterator dereference value type
  typedef PointerProxy<value_type> pointer;  ///< Pointer type to iterator value
  typedef value_type reference;  ///< Reference type to iterator value
  typedef std::input_iterator_tag
      iterator_category;  ///< Iterator category type
  typedef UnaryTransformIterator<Iter, Op> this_type;  ///< This object type
  typedef Iter base_iterator;  ///< The base iterator type

  /// Constructor

  /// \param it The base iterator
  /// \param op The transform operator
  UnaryTransformIterator(Iter it, Op op = Op()) : it_(it), op_(op) {}

  /// Copy constructor

  /// \param other The transform iterator to copy
  UnaryTransformIterator(const this_type& other)
      : it_(other.it_), op_(other.op_) {}

  /// Copy conversion constructor

  /// This constructor allows copying when the base iterators are implicitly
  /// convertible with each other.
  /// \tparam It An iterator type that is implicitly convertible to
  /// \c base_iterator type.
  /// \param other The transform iterator to copy
  /// \note The operation type must be the same for both transform iterators.
  template <typename It>
  UnaryTransformIterator(const UnaryTransformIterator<It, Op>& other)
      : it_(other.it_), op_(other.op_) {}

  /// Copy operator

  /// \param other The transform iterator to copy
  /// \return A reference to this object
  this_type& operator=(const this_type& other) {
    it_ = other.it_;
    op_ = other.op_;

    return *this;
  }

  /// Copy conversion operator

  /// This operator allows copying when the base iterators are implicitly
  /// convertible with each other.
  /// \tparam It An iterator type that is implicitly convertible to
  /// \c base_iterator .
  /// \param other The transform iterator to copy
  /// \note The operation type must be the same for both transform iterators.
  template <typename It>
  this_type& operator=(const UnaryTransformIterator<It, Op>& other) {
    it_ = other.it_;
    op_ = other.op_;

    return *this;
  }

  /// Prefix increment operator

  /// \return A reference to this object after it has been incremented.
  this_type& operator++() {
    increment();
    return *this;
  }

  /// Post-fix increment operator

  /// \return A copy of this object before it is incremented.
  this_type operator++(int) {
    this_type tmp(*this);
    increment();
    return tmp;
  }

  /// Equality operator

  /// \tparam It An iterator type that is implicitly convertible to
  /// \c base_iterator type.
  /// \param other The iterator to compare to this iterator.
  /// \return True when the base iterators are equal to each other, otherwise
  /// false.
  template <typename It>
  bool operator==(const UnaryTransformIterator<It, Op>& other) const {
    return equal(other);
  }

  /// Inequality operator

  /// \tparam It An iterator type that is implicitly convertible to
  /// \c base_iterator type.
  /// \param other The iterator to compare to this iterator.
  /// \return True when the base iterators are not equal to each other,
  /// otherwise false.
  template <typename It>
  bool operator!=(const UnaryTransformIterator<It, Op>& other) const {
    return !equal(other);
  }

  /// Dereference operator

  /// \return A transformed copy of the current base iterator value.
  reference operator*() const { return dereference(it_); }

  /// Arrow dereference operator

  /// \return A pointer to a transformed copy of the current base iterator
  /// value.
  pointer operator->() const { return pointer(dereference(it_)); }

  /// Base iterator accessor

  /// \return A copy of the base iterator.
  base_iterator base() const { return it_; }

 private:
  /// Increment the base iterator
  void increment() { ++it_; }

  /// Compare base iterators

  /// \tparam It An iterator type that is implicitly convertible to
  /// \c base_iterator type.
  /// \param other The iterator to compare to this iterator.
  /// \return True when the base iterators are equal to each other, otherwise
  /// false.
  template <typename It>
  bool equal(const UnaryTransformIterator<It, Op>& other) const {
    return it_ == other.it_;
  }

  /// Iterator dereference

  /// \return A transformed copy of the current value of the base iterator.
  template <typename It>
  std::enable_if_t<!std::is_integral_v<It>, reference> dereference(
      It it) const {
    return op_(*it);
  }

  /// Iterator dereference

  /// \return A transformed copy of the current value of the base iterator.
  template <typename It>
  std::enable_if_t<std::is_integral_v<It>, reference> dereference(It it) const {
    return op_(it);
  }

  base_iterator it_;  ///< The base iterator
  Op op_;             ///< The transform operation object
};                    // class UnaryTransformIterator

/// Binary Transform iterator factory

/// \tparam Iter1 First iterator type
/// \tparam Iter2 Second iterator type
/// \tparam Op The binary transform type
/// \param it1 First iterator
/// \param it2 Second iterator
/// \param op The binary transform object
/// \return A binary transform iterator
template <typename Iter1, typename Iter2, typename Op>
BinaryTransformIterator<Iter1, Iter2, Op> make_tran_it(Iter1 it1, Iter2 it2,
                                                       Op op) {
  return BinaryTransformIterator<Iter1, Iter2, Op>(it1, it2, op);
}

/// Unary Transform iterator factory

/// \tparam Iter The iterator type
/// \tparam Op The binary transform type
/// \param it The iterator
/// \param op The binary transform object
/// \return A unary transform iterator
template <typename Iter, typename Op>
UnaryTransformIterator<Iter, Op> make_tran_it(Iter it, Op op) {
  return UnaryTransformIterator<Iter, Op>(it, op);
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TRANSFORM_ITERATOR_H__INCLUDED
