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
#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <iosfwd>
#include <vector>
#include <array>
#include <initializer_list>

namespace TiledArray {
  namespace detail {


    /// Array size accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \return The size of c-stype array
    template <typename T, std::size_t N>
    inline constexpr std::size_t size(T (&)[N]) { return N; }

    /// Array size accessor

    /// \tparam T The array type
    /// \tparam N The size of the array
    /// \return The size of c-stype array
    template <typename T, std::size_t N>
    inline constexpr std::size_t size(const std::array<T, N>&) { return N; }

    /// Array size accessor

    /// \tparam T The array type
    /// \param a An array object
    /// \return The size of array \c a
    template <typename T,
        typename std::enable_if<! std::is_array<T>::value>::type* = nullptr>
    inline auto size(const T &a){ return a.size(); }

    /// Array size accessor

    /// \tparam T The initializer list element type
    /// \param a An initializer_list object
    /// \return The size of initializer_list \c a
    template <typename T>
    inline auto size(std::initializer_list<T> a) { return a.size(); }

    /// Container data pointer accessor

    /// \tparam T The container type
    /// \param t A container object
    /// \return A pointer to the first element of the container, \c v
    template <typename T,
        typename std::enable_if<! std::is_pointer<T>::value>::type* = nullptr>
    inline auto data(T& t)
    { return t.data(); }



    /// Container data pointer accessor

    /// \tparam T The container type
    /// \param t A container object
    /// \return A pointer to the first element of the container, \c v
    template <typename T,
        typename std::enable_if<! std::is_pointer<T>::value>::type* = nullptr>
    inline auto data(const T& t)
    { return t.data(); }

    /// Pointer data adapter

    /// \tparam T The container type
    /// \param t A pointer
    /// \return \c t (pass through)
    template <typename T,
        typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
    inline T data(T t) { return t; }

    /// Array data pointer accessor

    /// \tparam T The array type
    /// \param a The c-style array object
    /// \return A pointer to the first element of the array
    template <typename T, std::size_t N>
    inline T* data(T (&a)[N]) { return a; }

    /// Array data pointer accessor

    /// \tparam T The array type
    /// \param a The c-style array object
    /// \return A pointer to the first element of the array
    template <typename T, std::size_t N>
    inline const T* data(const T (&a)[N]) { return a; }


    /// Initializer list data pointer accessor

    /// \tparam T The initializer list element type
    /// \param l An initializer list object
    /// \return A pointer to the first element of the initializer list, \c l
    template <typename T>
    inline T* data(std::initializer_list<T>& l) { return l.begin(); }

    /// Initializer list const data pointer accessor

    /// \tparam T The initializer list element type
    /// \param l An initializer list object
    /// \return A const pointer to the first element of the initializer list, \c l
    template <typename T>
    inline const T* data(const std::initializer_list<T>& l) { return l.begin(); }

    /// Print the content of an array like object

    /// \tparam A The array container type
    /// \param out A standard output stream
    /// \param a The array-like container to be printed
    /// \param n The number of elements in the array.
    template <typename A>
    inline void print_array(std::ostream& out, const A& a, const std::size_t n) {
      out << "[";
      for(std::size_t i = 0; i < n; ++i) {
        out << a[i];
        if (i != (n - 1))
          out << ",";
      }
      out << "]";
    }

    /// Print the content of an array like object

    /// \tparam A The array container type
    /// \param out A standard output stream
    /// \param a The array-like container to be printed
    template <typename A>
    inline void print_array(std::ostream& out, const A& a) {
      print_array(out, a, size(a));
    }

  } // namespace detail
} // namespace TiledArray

namespace std {

  /// Vector output stream operator
  template <typename T, typename A>
  inline std::ostream& operator<<(std::ostream& os, const std::vector<T, A>& vec) {
    TiledArray::detail::print_array(os, vec);
    return os;
  }

} // namespace std

#endif // TILEDARRAY_UTILITY_H__INCLUDED
