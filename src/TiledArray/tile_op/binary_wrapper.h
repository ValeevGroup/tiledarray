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
 *  binary_interface.h
 *  Oct 6, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_BINARY_WRAPPER_H__INCLUDED
#define TILEDARRAY_TILE_OP_BINARY_WRAPPER_H__INCLUDED

#include <TiledArray/permutation.h>
#include <TiledArray/tile_op/tile_interface.h>
#include <TiledArray/zero_tensor.h>

namespace TiledArray {
namespace detail {

/// Binary tile operation wrapper

/// This wrapper class handles evaluation of lazily evaluated tiles in binary
/// operations and forwards the evaluated arguments to the base operation
/// object.
///
/// The base binary operation class must have the following interface.
/// \code
/// class Operator {
/// public:
///
///   typedef ... left_type;
///   typedef ... right_type;
///   typedef ... result_type;
///
///   static constexpr bool left_is_consumable =
///       LeftConsumable && std::is_same<result_type, Left>::value;
///   static constexpr bool right_is_consumable =
///       RightConsumable && std::is_same<result_type, Right>::value;
///
///   // Constructor
///   Operator();
///
///   // Constructor required for scaling operations only, and may be omitted
///   for other operations Operator(const Scalar);
///
///   // Operation evaluation operators
///   // L and R template parameters types may be Left and Right, respectively,
///   or
///   // TiledArray::ZeroTensor.
///
///   // Evaluate the operation with left and right arguments and permute the
///   result. template <typename L, typename R> result_type operator()(L&& left,
///   R&& right, const Permutation& perm) const;
///
///   // Evaluate the operation with left and right arguments.
///   // If the left_is_consumable or right_is_consumable variables are true,
///   then this
///   // will try to consume the left or right arguments, respectively.
///   template <typename L, typename R>
///   result_type operator()(L&& left, R&& right) const;
///
///   // Evaluate the operation with left and right arguments and try to consume
///   the left-hand
///   // argument. This function may not consume left if it is not consumable
///   (see is_consumable_tile trait). template <typename R> result_type
///   consume_left(left_type& left, R&& right) const;
///
///   // Evaluate the operation with left and right arguments and try to consume
///   the right-hand
///   // argument. This function may not consume right if it is not consumable
///   (see is_consumable_tile trait). template <typename L> result_type
///   consume_right(L&& left, right_type& right) const;
///
/// }; // class Operator
/// \endcode
/// \tparam Op The base binary operation type
template <typename Op>
class BinaryWrapper {
 public:
  typedef BinaryWrapper<Op> BinaryWrapper_;
  typedef typename Op::left_type left_type;      ///< Left-hand argument type
  typedef typename Op::right_type right_type;    ///< Right-hand argument type
  typedef typename Op::result_type result_type;  ///< The result tile type

  /// Boolean value that indicates the left-hand argument can always be consumed
  static constexpr bool left_is_consumable = Op::left_is_consumable;
  /// Boolean value that indicates the right-hand argument can always be
  /// consumed
  static constexpr bool right_is_consumable = Op::right_is_consumable;

  template <typename T>
  static constexpr bool is_lazy_tile_v = is_lazy_tile<std::decay_t<T>>::value;

  template <typename T>
  static constexpr bool is_array_tile_v = is_array_tile<std::decay_t<T>>::value;

  template <typename T>
  static constexpr bool is_nonarray_lazy_tile_v =
      is_lazy_tile_v<T> && !is_array_tile_v<T>;

  template <typename T>
  using eval_t = typename eval_trait<std::decay_t<T>>::type;

 private:
  Op op_;                      ///< Tile operation
  BipartitePermutation perm_;  ///< Permutation applied to the result

 public:
  // Compiler generated functions
  BinaryWrapper(const BinaryWrapper<Op>&) = default;
  BinaryWrapper(BinaryWrapper<Op>&&) = default;
  ~BinaryWrapper() = default;
  BinaryWrapper<Op>& operator=(const BinaryWrapper<Op>&) = default;
  BinaryWrapper<Op>& operator=(BinaryWrapper<Op>&&) = default;

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  BinaryWrapper(const Op& op, const Perm& perm) : op_(op), perm_(perm) {}

  BinaryWrapper(const Op& op) : op_(op), perm_() {}

  /// Evaluate two non-zero tiles and possibly permute

  /// Evaluate the result tile using the appropriate \c Derived class
  /// evaluation kernel.
  /// \param left The left-hand argument
  /// \param right The right-hand argument
  /// \return The result tile from the binary operation applied to the
  /// \c left and \c right arguments.
  template <typename L, typename R,
            std::enable_if_t<
                !(is_lazy_tile_v<L> || is_lazy_tile_v<R>)&&!std::is_same<
                    std::decay_t<L>, ZeroTensor>::value &&
                !std::is_same<std::decay_t<R>, ZeroTensor>::value>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    static_assert(
        std::is_same<std::decay_t<L>, left_type>::value,
        "BinaryWrapper::operator()(L&&,R&&): invalid argument type L");
    static_assert(
        std::is_same<std::decay_t<R>, right_type>::value,
        "BinaryWrapper::operator()(L&&,R&&): invalid argument type R");
    if (perm_) return op_(std::forward<L>(left), std::forward<R>(right), perm_);

    return op_(std::forward<L>(left), std::forward<R>(right));
  }

  /// Evaluate a zero tile to a non-zero tiles and possibly permute

  /// Evaluate the result tile using the appropriate \c Derived class
  /// evaluation kernel.
  /// \param left The left-hand argument
  /// \param right The right-hand argument
  /// \return The result tile from the binary operation applied to the
  /// \c left and \c right arguments.
  template <typename R, std::enable_if_t<!is_lazy_tile_v<R>>* = nullptr>
  auto operator()(const ZeroTensor& left, R&& right) const {
    static_assert(
        std::is_same<std::decay_t<R>, right_type>::value,
        "BinaryWrapper::operator()(zero,R&&): invalid argument type R");
    if (perm_) return op_(left, std::forward<R>(right), perm_);

    return op_(left, std::forward<R>(right));
  }

  /// Evaluate a non-zero tiles to a zero tile and possibly permute

  /// Evaluate the result tile using the appropriate \c Derived class
  /// evaluation kernel.
  /// \param left The left-hand argument
  /// \param right The right-hand argument
  /// \return The result tile from the binary operation applied to the
  /// \c left and \c right arguments.
  template <typename L, std::enable_if_t<!is_lazy_tile_v<L>>* = nullptr>
  auto operator()(L&& left, const ZeroTensor& right) const {
    static_assert(
        std::is_same<std::decay_t<L>, left_type>::value,
        "BinaryWrapper::operator()(L&&,zero): invalid argument type L");
    if (perm_) return op_(std::forward<L>(left), right, perm_);

    return op_(std::forward<L>(left), right);
  }

  // The following operators will evaluate lazy tile and use the base class
  // interface functions to call the correct evaluation kernel.

  /// Evaluate two lazy tiles

  /// This function will evaluate the \c left and \c right , then pass the
  /// evaluated tiles to the appropriate \c BinaryInterfaceBase_::operator()
  /// function.
  /// \tparam L The left-hand, lazy tile type
  /// \tparam R The right-hand, lazy tile type
  /// \param left The left-hand, lazy tile argument
  /// \param right The right-hand, lazy tile argument
  /// \return The result tile from the binary operation applied to the
  /// evaluated \c left and \c right .
  template <
      typename L, typename R,
      std::enable_if_t<is_lazy_tile_v<L> && is_lazy_tile_v<R> &&
                       (left_is_consumable || right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_left = invoke_cast(std::forward<L>(left));
    auto eval_right = invoke_cast(std::forward<R>(right));
    auto continuation = [this](
                            madness::future_to_ref_t<decltype(eval_left)> l,
                            madness::future_to_ref_t<decltype(eval_right)> r) {
      return BinaryWrapper_::operator()(l, r);
    };
    return meta::invoke(continuation, eval_left, eval_right);
  }

  /// Evaluate lazy and non-lazy tiles

  /// This function will evaluate the \c left , then pass the
  /// evaluated tile and \c right to the appropriate
  /// \c BinaryInterfaceBase_::operator() function.
  /// \tparam L The left-hand, lazy tile type
  /// \tparam R The right-hand, non-lazy tile type
  /// \param left The left-hand, lazy tile argument
  /// \param right The right-hand, non-lazy tile argument
  /// \return The result tile from the binary operation applied to the
  /// evaluated \c left and \c right .
  template <
      typename L, typename R,
      std::enable_if_t<is_lazy_tile_v<L> &&
                       (!is_lazy_tile_v<R>)&&(left_is_consumable ||
                                              right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_left = invoke_cast(std::forward<L>(left));
    auto continuation = [this](madness::future_to_ref_t<decltype(eval_left)> l,
                               R&& r) {
      return BinaryWrapper_::operator()(l, std::forward<R>(r));
    };
    return meta::invoke(continuation, eval_left, std::forward<R>(right));
  }

  /// Evaluate non-lazy and lazy tiles

  /// This function will evaluate the \c right , then pass the
  /// evaluated tile and \c left to the appropriate
  /// \c BinaryInterfaceBase_::operator() function.
  /// \tparam L The left-hand, non-lazy tile type
  /// \tparam R The right-hand, lazy tile type
  /// \param left The left-hand, non-lazy tile argument
  /// \param right The right-hand, lazy tile argument
  /// \return The result tile from the binary operation applied to the
  /// evaluated \c left and \c right .
  template <
      typename L, typename R,
      std::enable_if_t<(!is_lazy_tile_v<L>)&&is_lazy_tile_v<R> &&
                       (left_is_consumable || right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_right = invoke_cast(std::forward<R>(right));
    auto continuation =
        [this](L&& l, madness::future_to_ref_t<decltype(eval_right)> r) {
          return BinaryWrapper_::operator()(std::forward<L>(l), r);
        };
    return meta::invoke(continuation, std::forward<L>(left), eval_right);
  }

  /// Evaluate two lazy-array tiles

  /// This function will evaluate the \c left and \c right , then pass the
  /// evaluated tiles to the appropriate \c Derived class evaluation kernel.
  /// \tparam L The left-hand, lazy-array tile type
  /// \tparam R The right-hand, lazy-array tile type
  /// \param left The left-hand, non-lazy tile argument
  /// \param right The right-hand, lazy tile argument
  /// \return The result tile from the binary operation applied to the
  /// evaluated \c left and \c right .
  template <
      typename L, typename R,
      std::enable_if_t<is_array_tile_v<L> && is_array_tile_v<R> &&
                       !(left_is_consumable || right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_left = invoke_cast(std::forward<L>(left));
    auto eval_right = invoke_cast(std::forward<R>(right));

    if (perm_) return meta::invoke(op_, eval_left, eval_right, perm_);

    auto op_left = [=](eval_t<L>& _left, eval_t<R>& _right) {
      return op_.consume_left(_left, _right);
    };
    auto op_right = [=](eval_t<L>& _left, eval_t<R>& _right) {
      return op_.consume_right(_left, _right);
    };
    // Override consumable
    if (is_consumable_tile<eval_t<L>>::value && left.is_consumable())
      return meta::invoke(op_left, eval_left, eval_right);
    if (is_consumable_tile<eval_t<R>>::value && right.is_consumable())
      return meta::invoke(op_right, eval_left, eval_right);

    return meta::invoke(op_, eval_left, eval_right);
  }

  template <
      typename L, typename R,
      std::enable_if_t<is_array_tile_v<L> &&
                       (!is_lazy_tile_v<R>)&&!(left_is_consumable ||
                                               right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_left = invoke_cast(std::forward<L>(left));

    if (perm_) return op_(eval_left, std::forward<R>(right), perm_);

    // Override consumable
    if (is_consumable_tile<eval_t<L>>::value && left.is_consumable())
      return op_.consume_left(eval_left, std::forward<R>(right));

    return op_(eval_left, std::forward<R>(right));
  }

  template <
      typename L, typename R,
      std::enable_if_t<is_array_tile_v<L> && is_nonarray_lazy_tile_v<R> &&
                       !(left_is_consumable || right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_left = invoke_cast(std::forward<L>(left));
    auto eval_right = invoke_cast(std::forward<R>(right));

    if (perm_) return op_(eval_left, eval_right, perm_);

    // Override consumable
    if (is_consumable_tile<eval_t<L>>::value && left.is_consumable())
      return op_.consume_left(eval_left, eval_right);

    return op_(eval_left, eval_right);
  }

  template <
      typename L, typename R,
      std::enable_if_t<(!is_lazy_tile_v<L>)&&is_array_tile_v<R> &&
                       !(left_is_consumable || right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_right = invoke_cast(std::forward<R>(right));

    if (perm_) return op_(std::forward<L>(left), eval_right, perm_);

    // Override consumable
    if (is_consumable_tile<eval_t<R>>::value && right.is_consumable())
      return op_.consume_right(std::forward<L>(left), eval_right);

    return op_(std::forward<L>(left), eval_right);
  }

  template <
      typename L, typename R,
      std::enable_if_t<is_nonarray_lazy_tile_v<L> && is_array_tile_v<R> &&
                       !(left_is_consumable || right_is_consumable)>* = nullptr>
  auto operator()(L&& left, R&& right) const {
    auto eval_left = invoke_cast(std::forward<L>(left));
    auto eval_right = invoke_cast(std::forward<R>(right));

    if (perm_) return op_(eval_left, eval_right, perm_);

    // Override consumable
    if (is_consumable_tile<eval_t<R>>::value && right.is_consumable())
      return op_.consume_right(eval_left, eval_right);

    return op_(eval_left, eval_right);
  }

};  // class BinaryWrapper

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_BINARY_WRAPPER_H__INCLUDED
