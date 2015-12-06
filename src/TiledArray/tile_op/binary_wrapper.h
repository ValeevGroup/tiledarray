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

#ifndef TILEDARRAY_TILE_OP_BINARY_INTERFACE_H__INCLUDED
#define TILEDARRAY_TILE_OP_BINARY_INTERFACE_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>
#include <TiledArray/permutation.h>
#include <TiledArray/zero_tensor.h>

namespace TiledArray {
  namespace detail {

    /// Binary tile operation wrapper

    /// This wrapper class is handles evaluation of lazily evaluated tiles in binary operations and
    /// forwards the evaluated arguments to the base operation object.
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
    ///   // Constructor required for scaling operations only, and may be omitted for other operations
    ///   Operator(const Scalar);
    ///
    ///   // Operation evaluation operators
    ///   // L and R template parameters types may be Left and Right, repsectively, or
    ///   // TiledArray::ZeroTensor.
    ///
    ///   // Evaluate the operation with left and right arguments and permute the result.
    ///   template <typename L, typename R>
    ///   result_type operator()(L&& left, R&& right, const Permutation& perm) const;
    ///
    ///   // Evaluate the operation with left and right arguments.
    ///   // If the left_is_consumable or right_is_consumable variables are true, then this
    ///   // may the left or right arguments, respectively.
    ///   template <typename L, typename R>
    ///   result_type operator()(L&& left, R&& right) const;
    ///
    ///   // Evaluate the operation with left and right arguments and consume the left-hand
    ///   // argument. This function may not consume left if it is not consumable.
    ///   template <typename R>
    ///   result_type consume_left(left_type& left, R&& right) const;
    ///
    ///   // Evaluate the operation with left and right arguments and consume the right-hand
    ///   // argument. This function may not consume right if it is not consumable.
    ///   template <typename L>
    ///   result_type consume_right(L&& left, right_type& right) const;
    ///
    /// }; // class Operator
    /// \endcode
    /// \tparam Op The base binary operation type
    template <typename Op>
    class BinaryWrapper {
    public:
      typedef BinaryWrapper<Op> BinaryWrapper_;
      typedef typename Op::left_type left_type; ///< Left-hand argument type
      typedef typename Op::right_type right_type; ///< Right-hand argument type
      typedef typename Op::result_type result_type; ///< The result tile type


      /// Boolean value that indicates the left-hand argument can always be consumed
      static constexpr bool left_is_consumable = Op::left_is_consumable;
      /// Boolean value that indicates the right-hand argument can always be consumed
      static constexpr bool right_is_consumable = Op::right_is_consumable;

      template <typename T>
      using decay_t = typename std::decay<T>::type;

      template <typename T>
      using is_lazy_tile_t = is_lazy_tile<decay_t<T> >;

      template <typename T>
      using is_array_tile_t = is_array_tile<decay_t<T> >;

      template <typename T>
      using eval_t = typename eval_trait<decay_t<T> >::type;

    private:

      Op op_; ///< Tile operation
      Permutation perm_; ///< Permuation applied to the result

    public:

      // Compiler generated functions
      BinaryWrapper(const BinaryWrapper<Op>&) = default;
      BinaryWrapper(BinaryWrapper<Op>&&) = default;
      ~BinaryWrapper() = default;
      BinaryWrapper<Op>& operator=(const BinaryWrapper<Op>&) = default;
      BinaryWrapper<Op>& operator=(BinaryWrapper<Op>&&) = default;

      BinaryWrapper(const Op& op, const Permutation& perm) :
        op_(op), perm_(perm)
      { }

      BinaryWrapper(const Op& op) :
        op_(op), perm_()
      { }


      /// Evaluate two non-zero tiles and possibly permute

      /// Evaluate the result tile using the appropriate \c Derived class
      /// evaluation kernel.
      /// \param left The left-hand argument
      /// \param right The right-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c left and \c right arguments.
      result_type operator()(left_type&& left, right_type&& right) const {
        if(perm_)
          return op_(left, right, perm_);

        return op_(left, right);
      }

      /// Evaluate a zero tile to a non-zero tiles and possibly permute

      /// Evaluate the result tile using the appropriate \c Derived class
      /// evaluation kernel.
      /// \param left The left-hand argument
      /// \param right The right-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c left and \c right arguments.
      result_type operator()(const ZeroTensor& left, right_type&& right) const {
        if(perm_)
          return op_(left, right, perm_);

        return op_(left, right);
      }

      /// Evaluate a non-zero tiles to a zero tile and possibly permute

      /// Evaluate the result tile using the appropriate \c Derived class
      /// evaluation kernel.
      /// \param left The left-hand argument
      /// \param right The right-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c left and \c right arguments.
      result_type operator()(left_type&& left, const ZeroTensor& right) const {
        if(perm_)
          return op_(left, right, perm_);

        return op_(left, right);
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
      template <typename L, typename R,
          typename std::enable_if<
              is_lazy_tile_t<L>::value &&
              is_lazy_tile_t<R>::value &&
              (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);
        eval_t<R> eval_right(right);
        return BinaryWrapper_::operator()(std::forward<eval_t<L> >(eval_left),
            std::forward<eval_t<R> >(eval_right));
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
      template <typename L, typename R,
          typename std::enable_if<
              is_lazy_tile_t<L>::value &&
              (! is_lazy_tile_t<R>::value) &&
              (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);
        return BinaryWrapper_::operator()(std::forward<eval_t<L> >(eval_left), right);
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
      template <typename L, typename R,
          typename std::enable_if<
              (! is_lazy_tile_t<L>::value) &&
              is_lazy_tile_t<R>::value &&
              (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);
        eval_t<R> eval_right(right);
        return BinaryWrapper_::operator()(std::forward<eval_t<L> >(eval_left),
            std::forward<eval_t<R> >(eval_right));
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
      template <typename L, typename R,
          typename std::enable_if<
              is_array_tile_t<L>::value &&
              is_array_tile_t<R>::value &&
              ! (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);
        eval_t<R> eval_right(right);

        if(perm_)
          return op_(std::forward<eval_t<L> >(eval_left), std::forward<eval_t<R> >(eval_right), perm_);

        // Override consumable
        if(is_consumable_tile<eval_t<L> >::value && left.is_consumable())
          return op_.consume_left(eval_left, std::forward<eval_t<R> >(eval_right));
        if(is_consumable_tile<eval_t<R> >::value && right.is_consumable())
          return op_.consume_right(std::forward<eval_t<L> >(eval_left), eval_right);

        return op_(std::forward<eval_t<L> >(eval_left), std::forward<eval_t<R> >(eval_right));
      }


      template <typename L, typename R,
          typename std::enable_if<
              is_array_tile_t<L>::value &&
              (! is_lazy_tile_t<R>::value) &&
              ! (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);

        if(perm_)
          return op_(std::forward<eval_t<L> >(eval_left), right, perm_);

        // Override consumable
        if(is_consumable_tile<eval_t<L> >::value && left.is_consumable())
          return op_.consume_left(eval_left, right);

        return op_(std::forward<eval_t<L> >(eval_left), right);
      }


      template <typename L, typename R,
          typename std::enable_if<
              is_array_tile_t<L>::value &&
              is_lazy_tile_t<R>::value && (! is_array_tile_t<R>::value) &&
              ! (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);
        eval_t<R> eval_right(right);

        if(perm_)
          return op_(std::forward<eval_t<L> >(eval_left),
              std::forward<eval_t<R> >(eval_right), perm_);

        // Override consumable
        if(is_consumable_tile<eval_t<L> >::value && left.is_consumable())
          return op_.consume_left(eval_left, std::forward<eval_t<R> >(eval_right));

        return op_(std::forward<eval_t<L> >(eval_left),
            std::forward<eval_t<R> >(eval_right));
      }


      template <typename L, typename R,
          typename std::enable_if<
              (! is_lazy_tile_t<L>::value) &&
              is_array_tile_t<R>::value &&
              ! (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<R> eval_right(right);

        if(perm_)
          return op_(left, std::forward<eval_t<R> >(eval_right), perm_);

        // Override consumable
        if(is_consumable_tile<eval_t<R> >::value && right.is_consumable())
          return op_.consume_right(left, eval_right);

        return op_(left, std::forward<eval_t<R> >(eval_right));
      }


      template <typename L, typename R,
          typename std::enable_if<
              is_lazy_tile_t<L>::value && (! is_array_tile_t<L>::value) &&
              is_array_tile_t<R>::value &&
              ! (left_is_consumable || right_is_consumable)
          >::type* = nullptr>
      result_type operator()(L&& left, R&& right) const {
        eval_t<L> eval_left(left);
        eval_t<R> eval_right(right);

        if(perm_)
          return op_(std::forward<eval_t<L> >(eval_left), std::forward<eval_t<R> >(eval_right),
              perm_);

        // Override consumable
        if(is_consumable_tile<eval_t<R> >::value && right.is_consumable())
          return op_.consume_right(std::forward<eval_t<L> >(eval_left), eval_right);

        return op_(std::forward<eval_t<L> >(eval_left), std::forward<eval_t<R> >(eval_right));
      }

    }; // class BinaryWrapper

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_BINARY_INTERFACE_H__INCLUDED
