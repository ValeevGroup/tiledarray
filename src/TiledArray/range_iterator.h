/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_RANGE_ITERATOR_H__INCLUDED
#define TILEDARRAY_RANGE_ITERATOR_H__INCLUDED

#include <TiledArray/error.h>
#include <iterator>
#include <vector>

namespace TiledArray {
namespace detail {

template <typename, typename>
class RangeIterator;

}  // namespace detail
}  // namespace TiledArray

namespace std {
template <typename T, typename Container>
void advance(
    TiledArray::detail::RangeIterator<T, Container>&,
    typename TiledArray::detail::RangeIterator<T, Container>::difference_type);

template <typename T, typename Container>
typename TiledArray::detail::RangeIterator<T, Container>::difference_type
distance(const TiledArray::detail::RangeIterator<T, Container>&,
         const TiledArray::detail::RangeIterator<T, Container>&);

}  // namespace std

namespace TiledArray {
namespace detail {

/// Coordinate index iterate

/// This is an input iterator that is used to iterate over the coordinate
/// indexes of a \c Range.
/// \tparam T The value type of the iterator
/// \tparam Container The container that the iterator references
/// \note The container object must define the function
/// \c Container::increment(T&) \c const, and be accessible to
/// \c RangeIterator.
template <typename T, typename Container>
class RangeIterator {
 public:
  typedef RangeIterator<T, Container> RangeIterator_;  ///< This class type

  // Standard iterator typedefs
  typedef typename Container::index value_type;  ///< Iterator value type
  typedef const value_type& reference;           ///< Iterator reference type
  typedef const value_type* pointer;             ///< Iterator pointer type
  typedef std::input_iterator_tag iterator_category;  /// Iterator category tag
  typedef std::ptrdiff_t difference_type;  ///< Iterator difference type

  /// Copy constructor

  /// \param other The other iterator to be copied
  RangeIterator(const RangeIterator_& other)
      : container_(other.container_), current_(other.current_) {}

  /// Construct an index iterator

  /// \param v The initial value of the iterator index
  /// \param c The container that the iterator will reference
  RangeIterator(const T* v, const Container* c)
      : container_(c), current_(v, v + c->rank()) {}

  /// Copy constructor

  /// \param other The other iterator to be copied
  /// \return A reference to this object
  RangeIterator_& operator=(const RangeIterator_& other) {
    current_ = other.current_;
    container_ = other.container_;

    return *this;
  }

  const Container* container() const { return container_; }

  /// Dereference operator

  /// \return A \c reference to the current data
  /// \note this asserts that the iterator is valid
  reference operator*() const {
    TA_ASSERT(container_->includes(current_));
    return current_;
  }

  /// Increment operator

  /// Increment the iterator
  /// \return The modified iterator
  RangeIterator_& operator++() {
    TA_ASSERT(container_->includes(current_));
    container_->increment(current_);
    return *this;
  }

  /// Increment operator

  /// Increment the iterator
  /// \return An unmodified copy of the iterator
  RangeIterator_ operator++(int) {
    TA_ASSERT(container_->includes(current_));
    RangeIterator_ temp(*this);
    container_->increment(current_);
    return temp;
  }

  /// Pointer operator

  /// \return A \c pointer to the current data
  pointer operator->() const {
    TA_ASSERT(container_->includes(current_));
    return &current_;
  }

  void advance(difference_type n) {
    TA_ASSERT(container_->includes(current_));
    container_->advance(current_, n);
  }

  difference_type distance_to(const RangeIterator_& other) const {
    TA_ASSERT(container_ == other.container_);
    TA_ASSERT(container_->includes(current_));
    return container_->distance_to(current_, other.current_);
  }

 private:
  const Container* container_;  ///< The container that the iterator references
  typename Container::index current_;  ///< The current value of the iterator

  template <typename U, typename C>
  friend bool operator==(const RangeIterator<U, C>& left_it,
                         const RangeIterator<U, C>& right_it);

  /// \return A \c reference to the current data
  /// \note this does not check that the iterator is valid, hence only useful
  /// for implementing other methods
  reference value() const { return current_; }

};  // class RangeIterator

/// Equality operator

/// Compares the iterators for equality. They must reference the same range
/// object to be considered equal.
/// \tparam T The value type of the iterator
/// \tparam Container The container that the iterator references
/// \param left_it The left-hand iterator to be compared
/// \param right_it The right-hand iterator to be compared
/// \return \c true if the value and container are equal for the \c left_it
/// and \c right_it , otherwise \c false .
template <typename T, typename Container>
bool operator==(const RangeIterator<T, Container>& left_it,
                const RangeIterator<T, Container>& right_it) {
  return (left_it.value() == right_it.value()) &&
         (left_it.container() == right_it.container());
}

/// Inequality operator

/// Compares the iterators for inequality.
/// \tparam T The value type of the iterator
/// \tparam Container The container that the iterator references
/// \param left_it The left-hand iterator to be compared
/// \param right_it The right-hand iterator to be compared
/// \return \c true if the value or container are not equal for the
/// \c left_it and \c right_it , otherwise \c false .
template <typename T, typename Container>
bool operator!=(const RangeIterator<T, Container>& left_it,
                const RangeIterator<T, Container>& right_it) {
  //      return (left_it.value() != right_it.value()) ||
  //          (left_it.container() != right_it.container());
  return !(left_it == right_it);
}

}  // namespace detail
}  // namespace TiledArray

namespace std {
template <typename T, typename Container>
void advance(
    TiledArray::detail::RangeIterator<T, Container>& it,
    typename TiledArray::detail::RangeIterator<T, Container>::difference_type
        n) {
  it.advance(n);
}

template <typename T, typename Container>
typename TiledArray::detail::RangeIterator<T, Container>::difference_type
distance(const TiledArray::detail::RangeIterator<T, Container>& first,
         const TiledArray::detail::RangeIterator<T, Container>& last) {
  return first.distance_to(last);
}

}  // namespace std

#endif  // TILEDARRAY_RANGE_ITERATOR_H__INCLUDED
