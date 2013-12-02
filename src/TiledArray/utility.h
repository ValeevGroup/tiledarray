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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  utility.h
 *  Oct 18, 2013
 *
 */

#ifndef TILEDARRAY_UTILITY_H__INCLUDED
#define TILEDARRAY_UTILITY_H__INCLUDED

#include <TiledArray/madness.h>

namespace TiledArray {
  namespace detail {

    /// Array begin iterator accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \param a A c-style array
    /// \return A pointer to the first element of \c a
    template <typename T, std::size_t N>
    inline T* begin(T (&a)[N]) { return a; }

    /// Array const begin iterator accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \param a A c-style array
    /// \return A pointer to the first element of \c a
    template <typename T, std::size_t N>
    inline const T* cbegin(const T (&a)[N]) { return a; }

    /// Array end iterator accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \param a A c-style array
    /// \return A pointer to one past the last element of \c a
    template <typename T, std::size_t N>
    inline T* end(T (&a)[N]) { return (a + N); }

    /// Array const end iterator accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \param a A c-style array
    /// \return A pointer to one past the last element of \c a
    template <typename T, std::size_t N>
    inline const T* cend(const T (&a)[N]) { return (a + N); }

    /// Array size accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \param a A c-style array
    /// \return The size of array \c a
    template <typename T, std::size_t N>
    inline std::size_t size(T (&)[N]) { return N; }

    /// Array begin iterator accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The begin iterator of \c a
    template <typename T>
    inline typename T::iterator begin(T &a) { return a.begin(); }

    /// Array const begin iterator accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The begin const_iterator of \c a
    template <typename T>
    inline typename T::const_iterator begin(const T &a) { return a.begin(); }

    /// Array const begin iterator accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The begin cosnt_iterator of \c a
    template <typename T>
    inline typename T::const_iterator cbegin(const T &a) { return a.begin(); }

    /// Array const begin iterator accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The begin cosnt_iterator of \c a
    template <typename T>
    inline typename T::iterator end(T &a) { return a.end(); }

    /// Array const end iterator accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The end cosnt_iterator of \c a
    template <typename T>
    inline typename T::const_iterator end(const T &a) { return a.end(); }

    /// Array const end iterator accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The end cosnt_iterator of \c a
    template <typename T>
    inline typename T::const_iterator cend(const T &a) { return a.end(); }

    /// Array size accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The size of array \c a
    template <typename T>
    inline typename madness::disable_if<std::is_array<T>, std::size_t>::type
    size(const T &a) { return a.size(); }

    template <typename A>
    void print_array(std::ostream& out, const A& a) {
      std::size_t n = a.size();
      out << "[";
      for(std::size_t i = 0; i < n; ++i) {
        out << a[i];
        if (i != (n - 1))
          out << ",";
      }
      out << "]";
    }

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_UTILITY_H__INCLUDED
