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
#include <iostream>
#include <vector>

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
    /// \param a An array object
    /// \return The size of array \c a
    template <typename T>
    inline typename madness::disable_if<std::is_array<T>, std::size_t>::type
    size(const T &a) { return a.size(); }
    /// Print the content of an array like object

    /// \tparam A The array container type
    /// \param out A standard output stream
    /// \param a The array-like container to be printed
    template <typename A>
    void print_array(std::ostream& out, const A& a) {
      const std::size_t n = size(a);
      out << "[";
      for(std::size_t i = 0; i < n; ++i) {
        out << a[i];
        if (i != (n - 1))
          out << ",";
      }
      out << "]";
    }


    template <typename T, typename Op, typename Result>
    class BinderFirstHelper {
      const T& restrict t_;
      const Op& op_;

    public:
      typedef Result result_type;

      BinderFirstHelper(const T& restrict t, const Op& op) : t_(t), op_(op) { }

      template <typename Arg>
      result_type operator()(const Arg& restrict arg) const {
        return op_(t_, arg);
      }
    };


    template <typename T, typename Op>
    class BinderFirstHelper<T, Op, void> {
      const T& restrict t_;
      const Op& op_;

    public:
      typedef void result_type;

      BinderFirstHelper(const T& restrict t, const Op& op) : t_(t), op_(op) { }

      template <typename Result, typename Arg>
      void operator()(Result& restrict result, const Arg& restrict arg) const {
        op_(result, t_, arg);
      }
    };

    template <typename T, typename Op>
    class BinderFirst : public BinderFirstHelper<T, Op, typename madness::detail::result_of<Op>::type> {
      typedef BinderFirstHelper<T, Op, typename madness::detail::result_of<Op>::type> BinderFirstHelper_;
    public:
      typedef typename BinderFirstHelper_::result_type result_type;

      BinderFirst(const T& restrict t, const Op& op) :
        BinderFirstHelper_(t, op)
      { }

    };

    template <typename T, typename Op>
    inline BinderFirst<T, Op> bind_first(const T& restrict t, const Op& op) {
      return BinderFirst<T, Op>(t, op);
    }

    template <typename T, typename Op, typename Result>
    class BinderSecondHelper {
      const T& restrict t_;
      const Op& op_;

    public:
      typedef Result result_type;

      BinderSecondHelper(const T& restrict t, const Op& op) : t_(t), op_(op) { }

      template <typename Arg>
      result_type operator()(const Arg& restrict arg) const {
        return op_(arg, t_);
      }
    };


    template <typename T, typename Op>
    class BinderSecondHelper<T, Op, void> {
      const T& restrict t_;
      const Op& op_;

    public:
      typedef void result_type;

      BinderSecondHelper(const T& restrict t, const Op& op) : t_(t), op_(op) { }

      template <typename Result, typename Arg>
      void operator()(Result& restrict result, const Arg& restrict arg) const {
        op_(result, arg, t_);
      }
    };

    template <typename T, typename Op>
    class BinderSecond : public BinderSecondHelper<T, Op, typename madness::detail::result_of<Op>::type> {
      typedef BinderSecondHelper<T, Op, typename madness::detail::result_of<Op>::type> BinderSecondHelper_;
    public:
      typedef typename BinderSecondHelper_::result_type result_type;

      BinderSecond(const T& restrict t, const Op& op) :
        BinderSecondHelper_(t, op)
      { }

    };

    template <typename T, typename Op>
    inline BinderSecond<T, Op> bind_second(const T& restrict t, const Op& op) {
      return BinderSecond<T, Op>(t, op);
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
