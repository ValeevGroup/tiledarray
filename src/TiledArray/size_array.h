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
        typedef std::size_t    size_type;
        typedef std::ptrdiff_t difference_type;

        SizeArray() : first_(NULL), last_(NULL) { }

        SizeArray(value_type* first, value_type* last) :
          first_(first), last_(last)
        { }

        void set(value_type* const first, const size_type n) {
          first_ = first;
          last_ = first + n;
        }

        SizeArray<T>& operator=(const SizeArray<T>& other) {
          TA_ASSERT(size() == other.size());
          std::copy(other.begin(), other.end(), first_);
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
          std::copy(first_, last_, temp.begin());

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
        const T* data() const { return first_; }

        // use array as C array (direct read/write access to data)
        T* data() { return first_; }

        // assign one value to all elements
        void assign (const T& value) { std::fill(begin(), end(), value); }

        // Comparison operators
        template <typename U>
        bool operator==(const SizeArray<U>& other) const {
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
          const std::size_t n = size();
          if(n != other.size())
            return true;

          for(std::size_t i = 0; i < n; ++i)
            if(first_[i] == other[i])
              return false;

          return true;
        }

    }; // class SizeArray

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_SIZE_ARRAY_H__INCLUDED
