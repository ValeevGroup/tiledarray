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
 *  reduce_wrapper.h
 *  Oct 7, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_REDUCE_INTERFACE_H__INCLUDED
#define TILEDARRAY_TILE_OP_REDUCE_INTERFACE_H__INCLUDED

#include <TiledArray/external/madness.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
namespace math {

/// Unary reduction wrapper class that handles lazy tile evaluation

/// This allows reduction functors to handle lazy tile types.
/// \tparam Tile The tile argument type
/// \tparam Op The base reduction operation type
template <typename Tile, typename Op>
class UnaryReduceWrapper : public Op {
 public:
  // typedefs
  typedef UnaryReduceWrapper<Tile, Op>
      UnaryReduceWrapper_;                       ///< This class type
  typedef typename Op::result_type result_type;  ///< The reduction result type
  typedef Tile argument_type;  ///< The reduction argument type

  // Constructors
  UnaryReduceWrapper() : Op() {}
  UnaryReduceWrapper(const Op& op) : Op(op) {}
  UnaryReduceWrapper(const UnaryReduceWrapper_& other) : Op(other) {}

  UnaryReduceWrapper_& operator=(const UnaryReduceWrapper_& other) {
    Op::operator=(other);
    return *this;
  }

  // Import base class functionality
  using Op::operator();

 private:
  template <typename T>
  typename std::enable_if<is_lazy_tile<T>::value>::type reduce(
      result_type& result, const T& arg) const {
    typename eval_trait<T>::type eval_arg(arg);
    Op::operator()(result, eval_arg);
  }

  template <typename T>
  typename std::enable_if<!is_lazy_tile<T>::value>::type reduce(
      result_type& result, const T& arg) const {
    Op::operator()(result, arg);
  }

 public:
  // Reduce an argument
  void operator()(result_type& result, const argument_type& arg) const {
    reduce(result, arg);
  }

};  // struct UnaryReduceWrapper

/// Unary reduction wrapper class that handles lazy tile evaluation

/// This class is a shallow wrapper for cases where the argument is not a
/// lazy tile.
/// \tparam Op The base reduction operation type
template <typename Op>
class UnaryReduceWrapper<typename Op::argument_type, Op> : public Op {
 public:
  // typedefs
  typedef UnaryReduceWrapper<typename Op::argument_type, Op>
      UnaryReduceWrapper_;                       ///< This class type
  typedef typename Op::result_type result_type;  ///< The reduction result type
  typedef typename Op::argument_type
      argument_type;  ///< The reduction argument type

  // Constructors
  UnaryReduceWrapper() : Op() {}
  UnaryReduceWrapper(const Op& op) : Op(op) {}
  UnaryReduceWrapper(const UnaryReduceWrapper_& other) : Op(other) {}

  UnaryReduceWrapper_& operator=(const UnaryReduceWrapper_& other) {
    Op::operator=(other);
    return *this;
  }

  // Import base class functionality
  using Op::operator();

};  // struct UnaryReduceWrapper

/// Binary reduction wrapper class that handles lazy tile evaluation

/// This allows reduction functors to handle lazy tile types.
/// \tparam Left The left-hand tile argument type
/// \tparam Right The right-hand tile argument type
/// \tparam Op The base reduction operation type
template <typename Left, typename Right, typename Op>
struct BinaryReduceWrapper : public Op {
 public:
  // typedefs
  typedef BinaryReduceWrapper<Left, Right, Op>
      BinaryReduceWrapper_;                      ///< This class type
  typedef typename Op::result_type result_type;  ///< The reduction result type
  typedef Left first_argument_type;  ///< The reduction left-hand argument type
  typedef Right
      second_argument_type;  ///< The reduction right-hand argument type

  // Constructors
  BinaryReduceWrapper() : Op() {}
  BinaryReduceWrapper(const Op& op) : Op(op) {}
  BinaryReduceWrapper(const BinaryReduceWrapper_& other) : Op(other) {}

  BinaryReduceWrapper_& operator=(const BinaryReduceWrapper_& other) {
    Op::operator=(other);
    return *this;
  }

 private:
  template <typename L, typename R>
  typename std::enable_if<is_lazy_tile<L>::value &&
                          is_lazy_tile<R>::value>::type
  reduce(result_type& result, const L& left, const R& right) const {
    typename eval_trait<L>::type eval_left(left);
    typename eval_trait<R>::type eval_right(right);
    Op::operator()(result, eval_left, eval_right);
  }

  template <typename L, typename R>
  typename std::enable_if<(!is_lazy_tile<L>::value) &&
                          is_lazy_tile<R>::value>::type
  reduce(result_type& result, const L& left, const R& right) const {
    typename eval_trait<R>::type eval_right(right);
    Op::operator()(result, left, eval_right);
  }

  template <typename L, typename R>
  typename std::enable_if<is_lazy_tile<L>::value &&
                          (!is_lazy_tile<R>::value)>::type
  reduce(result_type& result, const L& left, const R& right) const {
    typename eval_trait<L>::type eval_left(left);
    Op::operator()(result, eval_left, right);
  }

  template <typename L, typename R>
  typename std::enable_if<!(is_lazy_tile<L>::value ||
                            is_lazy_tile<R>::value)>::type
  reduce(result_type& result, const L& left, const R& right) const {
    Op::operator()(result, left, right);
  }

 public:
  // Import base class functionality
  using Op::operator();

  // Reduce an argument pair
  void operator()(result_type& result, const first_argument_type& left,
                  const second_argument_type& right) const {
    reduce(result, left, right);
  }

};  // struct BinaryReduceWrapper

/// Binary reduction operation wrapper
template <typename Op>
struct BinaryReduceWrapper<typename Op::first_argument_type,
                           typename Op::second_argument_type, Op> : public Op {
  // typedefs
  typedef BinaryReduceWrapper<typename Op::first_argument_type,
                              typename Op::second_argument_type, Op>
      BinaryReduceWrapper_;                      ///< This class type
  typedef typename Op::result_type result_type;  ///< The reduction result type
  typedef typename Op::first_argument_type
      first_argument_type;  ///< The reduction left-hand argument type
  typedef typename Op::second_argument_type
      second_argument_type;  ///< The reduction right-hand argument type

  // Constructors
  BinaryReduceWrapper() : Op() {}
  BinaryReduceWrapper(const Op& op) : Op(op) {}
  BinaryReduceWrapper(const BinaryReduceWrapper_& other) : Op(other) {}

  BinaryReduceWrapper_& operator=(const BinaryReduceWrapper_& other) {
    Op::operator=(other);
    return *this;
  }

  // Import base class functionality
  using Op::operator();

};  // struct BinaryReduceWrapper

}  // namespace math
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_REDUCE_INTERFACE_H__INCLUDED
