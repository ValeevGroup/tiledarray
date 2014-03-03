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

#ifndef TILEDARRAY_SIZE_ARRAY_H__INCLUDED
#define TILEDARRAY_SIZE_ARRAY_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/madness.h>
#include <TiledArray/math/outer.h>
#include <TiledArray/math/partial_reduce.h>
#include <algorithm>
#include <vector>
#include <iterator>
#include <cstddef>

namespace TiledArray {
  namespace detail {

    /// Array wrapper

    /// This object is a wrapper for raw memory buffers so that it has the same
    /// interface, and can be used in place of, standard containers (e.g.
    /// std::vector). SizeArray does not own the buffer, therefore it is the
    /// user's responsibility to manage (allocate, free, etc.) the memory buffer.
    /// \tparam T The type of the array referenced by this array object
    template <typename T>
    class SizeArray {
    private:
      T* first_; ///< First element of the array
      T* last_; ///< Last element of the array

      // Not allowed
      SizeArray(const SizeArray&);

    public:
      // type definitions
      typedef T              value_type;
      typedef T*             iterator;
      typedef const T*       const_iterator;
      typedef T&             reference;
      typedef const T&       const_reference;
      typedef T*             pointer;
      typedef const T*       const_pointer;
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;

      SizeArray() : first_(NULL), last_(NULL) { }

      SizeArray(pointer const first, pointer const last) :
        first_(first), last_(last)
      { }

      void set(pointer const first, const size_type n) {
        first_ = first;
        last_ = first + n;
      }

      void set(pointer const first, pointer const last) {
        first_ = first;
        last_ = last;
      }

      SizeArray<T>& operator=(const SizeArray<T>& other) {
        TA_ASSERT(size() == other.size());
        math::copy_vector(last_ - first_, other.data(), first_);
        return *this;
      }

      template <typename U>
      SizeArray<T>& operator=(const U& other) {
        TA_ASSERT(size() == other.size());
        std::copy(other.begin(), other.end(), first_);
        return *this;
      }

      template <typename U>
      operator std::vector<U> () const {
        return std::vector<U>(first_, last_);
      }

      template <typename U, std::size_t N>
      operator std::array<U, N> () const {
        TA_ASSERT(N == size());
        std::array<U, N> temp;
        math::copy_vector(last_ - first_, first_, temp.begin());

        return temp;
      }

      // iterator support
      iterator begin() { return first_; }
      const_iterator begin() const { return first_; }
      iterator end() { return last_; }
      const_iterator end() const { return last_; }

      // reverse iterator support
      typedef std::reverse_iterator<iterator> reverse_iterator;
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

      reverse_iterator rbegin() { return reverse_iterator(end()); }
      const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
      reverse_iterator rend() { return reverse_iterator(begin()); }
      const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

      // operator[]
      reference operator[](size_type i) { return first_[i]; }
      const_reference operator[](size_type i) const { return first_[i]; }

      // at() with range check
      reference at(size_type i) {
        TA_ASSERT(i < size());
        return first_[i];
      }

      const_reference at(size_type i) const {
        TA_ASSERT(i < size());
        return first_[i];
      }

      // front() and back()
      reference front() {
        TA_ASSERT(first_);
        return first_[0];
      }

      const_reference front() const {
        TA_ASSERT(first_);
        return first_[0];
      }

      reference back() {
        TA_ASSERT(first_);
        return *(last_ - 1);
      }

      const_reference back() const {
        TA_ASSERT(first_);
        return *(last_ - 1);
      }

      // size is constant
      size_type size() const { return last_ - first_; }
      bool empty() const { return first_ == NULL; }
      size_type max_size() const { return last_ - first_; }

      // swap (note: linear complexity in N, constant for given instantiation)
      void swap (SizeArray<T>& other) {
        std::swap_ranges(begin(), end(), other.begin());
      }

      // direct access to data (read-only)
      const_pointer data() const { return first_; }

      // use array as C array (direct read/write access to data)
      pointer data() { return first_; }

      // assign one value to all elements
      void assign (const_reference value) { math::fill_vector(last_ - first_, value, first_); }

      // Comparison operators
      template <typename U>
      bool operator==(const SizeArray<U>& other) const {
        if(first_ == other.data())
          return true;

        const std::size_t n = size();
        if(n != other.size())
          return false;

        for(std::size_t i = 0; i < n; ++i)
          if(first_[i] != other[i])
            return false;

        return true;
      }

      template <typename U>
      bool operator!=(const SizeArray<U>& other) const {
        if(first_ == other.data())
          return false;

        const std::size_t n = size();
        if(n != other.size())
          return true;

        for(std::size_t i = 0; i < n; ++i)
          if(first_[i] == other[i])
            return false;

        return true;
      }

      // Vector operations

      /// Binary vector operation

      /// Binary operation where this object is the left-hand argument. The
      /// values of this array is set by <tt>op(*this[i], arg[i])</tt>.
      /// \tparam Arg The right-hand argument type
      /// \tparam Op The binary operation type
      /// \param arg The right-hand argument
      /// \param op The binary operation
      template <typename Arg, typename Op>
      void binary(const Arg* const arg, const Op& op) {
        math::binary_vector_op(last_ - first_, arg, first_, op);
      }

      /// Binary vector operation

      /// Binary operation sets the values of this array such that,
      /// <tt>*this[i] = op(left[i], right[i])</tt>. The result type of \c op
      /// must be \c value_type or implicitly convertible to \c value_type.
      /// \tparam Left The left-hand argument type
      /// \tparam Right The right-hand argument type
      /// \tparam Op The binary operation type
      /// \param left The left-hand argument
      /// \param right The right-hand argument
      /// \param op The binary operation
      template <typename Left, typename Right, typename Op>
      void binary(const Left* const left, const Right* const right, const Op& op) {
        math::binary_vector_op(last_ - first_, left, right, first_, op);
      }

      /// Unary vector operation

      /// Unary operation where this object is the argument. The
      /// values of this array is set by <tt>op(*this[i])</tt>.
      /// \tparam Op The binary operation type
      /// \param op The binary operation
      template <typename Op>
      void unary(const Op& op) {
        math::unary_vector_op(last_ - first_, first_, op);
      }

      /// Unary vector operation

      /// Unary operation sets the values of this array such that,
      /// <tt>*this[i] = op(arg[i])</tt>. The result type of \c op must be
      /// \c value_type or implicitly convertible to \c value_type.
      /// \tparam Arg The argument type
      /// \tparam Op The unary operation type
      /// \param arg The argument
      template <typename Arg, typename Op>
      void unary(const Arg* const arg, const Op& op) {
        math::unary_vector_op(last_ - first_, arg, first_, op);
      }

      /// Binary reduce vector operation

      /// Binary reduction operation where this object is the left-hand
      /// argument type. The reduced result is computed by
      /// <tt>op(result, *this[i], arg[i])</tt>.
      /// \tparam Arg The right-hand argument type
      /// \tparam Result The reduction result type
      /// \tparam Op The binary operation type
      /// \param arg The right-hand argument
      /// \param result The initial value of the reduction
      /// \param op The binary reduction operation
      /// \return The reduced value
      template <typename Arg, typename Result, typename Op>
      Result reduce(const Arg* const arg, Result result, const Op& op) const {
        math::reduce_vector_op(last_ - first_, first_, arg, result, op);
        return result;
      }

      /// Unary reduce vector operation

      /// Unary reduction operation where this object is the argument. The
      /// reduced result is computed by <tt>op(result, *this[i])</tt>.
      /// \tparam Result The reduction result type
      /// \tparam Op The binary reduction operation type
      /// \param value The initial value of the reduction
      /// \param result The initial value of the reduction
      /// \param op The unary reduction operation
      /// \return The reduced value
      template <typename Result, typename Op>
      Result reduce(Result result, const Op& op) const {
        math::reduce_vector_op(last_ - first_, first_, result, op);
        return result;
      }

      /// Row reduce operation

      /// Reduce rows of \c left matrix to this array. The reduced result is
      /// computed by <tt>op(*this[i], left[i][j], right[j])</tt>.
      /// \tparam Left The matrix element type
      /// \tparam Right The vector element type
      /// \tparam Op The reduction operation
      /// \param n The number of columns in the matrix
      /// \param left The matrix pointer of size \c size()*n
      /// \param right The vector pointer of size \c n
      /// \param op The reduction operation
      template <typename Left, typename Right, typename Op>
      void row_reduce(const size_type n, const Left* const left, const Right* right, const Op& op) {
        math::row_reduce(last_ - first_, n, left, right, first_, op);
      }

      /// Row reduce operation

      /// Reduce rows of \c arg matrix to this array. The reduced result is
      /// computed by <tt>op(*this[i], arg[i][j])</tt>.
      /// \tparam Arg The matrix element type
      /// \tparam Op The reduction operation
      /// \param n The number of columns in the matrix
      /// \param arg The matrix pointer of size \c size()*n
      /// \param op The reduction operation
      template <typename Arg, typename Op>
      void row_reduce(const size_type n, const Arg* const arg, const Op& op) {
        math::row_reduce(last_ - first_, n, arg, first_, op);
      }


      /// Column reduce operation

      /// Reduce columns of \c left matrix to this array. The reduced result is
      /// computed by <tt>op(*this[j], left[i][j], right[i])</tt>.
      /// \tparam Left The matrix element type
      /// \tparam Right The vector element type
      /// \tparam Op The reduction operation
      /// \param m The number of rows in the matrix
      /// \param left The matrix pointer of size \c m*size()
      /// \param right The vector pointer of size \c m
      /// \param op The reduction operation
      template <typename Left, typename Right, typename Op>
      void col_reduce(const size_type m, const Left* const left, const Right* right, const Op& op) {
        math::col_reduce(m, last_ - first_, left, right, first_, op);
      }

      /// Columns reduce operation

      /// Reduce columns of \c arg matrix to this array. The reduced result is
      /// computed by <tt>op(*this[j], arg[i][j])</tt>.
      /// \tparam Arg The matrix element type
      /// \tparam Op The reduction operation
      /// \param m The number of rows in the matrix
      /// \param arg The matrix pointer of size \c m*size()
      /// \param op The reduction operation
      template <typename Arg, typename Op>
      void col_reduce(const size_type m, const Arg* const arg, const Op& op) {
        math::col_reduce(m, last_ - first_, arg, first_, op);
      }

      /// Outer operation

      /// This function use two vectors, \c left and \c right, and this vector
      /// to modify this vector ( which is treated as a matrix of size \c m*n )
      /// such that <tt>op(*this[i][j], left[i], right[j])</tt>.
      /// \tparam Left The left-hand argument type
      /// \tparam Right The right-hand argument type
      /// \param m The size of the left-hand array
      /// \param n The size of the right-hand array
      /// \param left A pointer to the left-hand array
      /// \param right A pointer to the right-hand array
      /// \param op The outer operation
      template <typename Left, typename Right, typename Op>
      void outer(const size_type m, const size_type n, const Left* const left,
          const Right* const right, const Op& op)
      {
        TA_ASSERT((m * n) == size());
        math::outer(m, n, left, right, first_, op);
      }

      /// Outer fill operation

      /// This function use two vectors, \c left and \c right, to fill this
      /// vector ( which is treated as a matrix of size \c m*n ) such that
      /// <tt>*this[i][j] = op(left[i], right[j])</tt>.
      /// \tparam Left The left-hand array type
      /// \tparam Right The right-hand array type
      /// \param m The size of the left-hand array
      /// \param n The size of the right-hand array
      /// \param left A pointer to the left-hand array
      /// \param right A pointer to the right-hand array
      /// \param op The outer operation
      template <typename Left, typename Right, typename Op>
      void outer_fill(const size_type m, const size_type n, const Left* const left,
          const Right* const right, const Op& op)
      {
        TA_ASSERT((m * n) == size());
        math::outer_fill(m, n, left, right, first_, op);
      }

      /// Outer operation

      /// This function use two vectors, \c left and \c right, and base array
      /// to modify this array ( which are treated as a matrix of size \c m*n )
      /// such that <tt>*this[i][j] = op(base[i][j], left[i], right[j])</tt>.
      /// \tparam Left The left-hand argument type
      /// \tparam Right The right-hand argument type
      /// \tparam Base The base argument type
      /// \param m The size of the left-hand array
      /// \param n The size of the right-hand array
      /// \param left A pointer to the left-hand array
      /// \param right A pointer to the right-hand array
      /// \param base A pointer to the base array
      /// \param op The outer operation
      template <typename Left, typename Right, typename Base, typename Op>
      void outer_fill(const size_type m, const size_type n, const Left* const left,
          const Right* const right, const Base* const base, const Op& op)
      {
        TA_ASSERT((m * n) == size());
        math::outer_fill(m, n, left, right, base, first_, op);
      }

      // Common reduction operations



    }; // class SizeArray

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_SIZE_ARRAY_H__INCLUDED
