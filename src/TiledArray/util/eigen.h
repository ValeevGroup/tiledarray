/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  eigen.h
 *  Oct 1, 2019
 *
 */

#ifndef TILEDARRAY_UTIL_EIGEN_H__INCLUDED
#define TILEDARRAY_UTIL_EIGEN_H__INCLUDED

//
// Configure Eigen and include Eigen/Core, all dependencies should include this
// before including any Eigen headers
//

#include <TiledArray/external/eigen.h>

namespace std {

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
constexpr auto begin(
    Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& m) {
  return m.data();
}

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
constexpr auto begin(
    const Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& m) {
  return m.data();
}

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
constexpr auto end(Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& m) {
  return m.data() + m.size();
}

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
constexpr auto end(
    const Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& m) {
  return m.data() + m.size();
}

}  // namespace std

#include <TiledArray/type_traits.h>

namespace TiledArray {

namespace eigen {

template <typename Range,
          typename = std::enable_if_t<detail::is_integral_range_v<Range> &&
                                      detail::is_sized_range_v<Range>>>
Eigen::Matrix<detail::value_t<Range>, Eigen::Dynamic, 1> iv(Range&& rng) {
  Eigen::Matrix<detail::value_t<Range>, Eigen::Dynamic, 1> result(
      std::size(rng));
  long count = 0;
  for (auto&& i : rng) {
    result(count) = i;
    ++count;
  }
  return result;
}

// TODO constrain this to run only at compile time
// template <typename Range, typename =
// std::enable_if_t<detail::is_integral_range_v<Range>>> constexpr auto
// ivn(Range&& rng) {
//  Eigen::Matrix<detail::value_t<Range>, std::size(rng), 1> result;
//  long count = 0;
//  for(auto && i : rng) {
//    result(count) = i;
//    ++count;
//  }
//  return result;
//}

template <typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
Eigen::Matrix<Int, Eigen::Dynamic, 1> iv(std::initializer_list<Int> list) {
  Eigen::Map<const Eigen::Matrix<Int, Eigen::Dynamic, 1>> result(data(list),
                                                                 size(list));
  return result;
}

namespace detail {
template <typename Mat, typename T, typename... Ts>
void iv_assign(Mat& m, int i, T v, Ts... vrest) {
  m(i) = v;
  if constexpr (sizeof...(Ts) > 0) {
    iv_assign(m, i + 1, vrest...);
  }
}
}  // namespace detail

template <typename Int, typename... Ints,
          typename = std::enable_if_t<std::is_integral_v<Int> &&
                                      (std::is_integral_v<Ints> && ...)>>
constexpr auto iv(Int i0, Ints... rest) {
  Eigen::Matrix<Int, sizeof...(Ints) + 1, 1> result;
  detail::iv_assign(result, 0, i0, rest...);
  return result;
}

/// evaluates an Eigen expression
template <typename Derived>
auto iv(const Eigen::MatrixBase<Derived>& mat) {
  Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,
                Eigen::internal::traits<Derived>::RowsAtCompileTime,
                Eigen::internal::traits<Derived>::ColsAtCompileTime>
      result = mat;
  return result;
}

}  // namespace eigen

}  // namespace TiledArray

#include <boost/range.hpp>

namespace boost {

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
struct range_mutable_iterator<
    Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>> {
  typedef _Scalar* type;
};

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
struct range_const_iterator<
    Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>> {
  typedef const _Scalar* type;
};

}  // namespace boost
namespace Eigen {

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
auto range_begin(
    const Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& mat) {
  return mat.data();
}
template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
auto range_begin(Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& mat) {
  return mat.data();
}

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
auto range_end(
    const Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& mat) {
  return mat.data() + mat.size();
}
template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
auto range_end(Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& mat) {
  return mat.data() + mat.size();
}

template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
auto range_calculate_size(
    const Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& mat) {
  return mat.size();
}
template <typename _Scalar, int _Rows, int _Options, int _MaxRows>
auto range_calculate_size(
    Eigen::Matrix<_Scalar, _Rows, 1, _Options, _MaxRows, 1>& mat) {
  return mat.size();
}

}  // namespace Eigen

#endif  // TILEDARRAY_UTIL_EIGEN_H__INCLUDED
