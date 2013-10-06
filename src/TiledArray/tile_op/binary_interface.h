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

#include <TiledArray/tile_op/type_traits.h>

namespace TiledArray {
  namespace math {

    template <typename Op>
    struct BinaryTileOpPolicy;

    /// Policy class for binary tile operations

    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    /// \tparam Op The binary tile operation template
    template <typename Result, typename Left, typename Right, bool LeftConsumable,
        bool RightConsumable, template <typename, typename, typename, bool, bool> class Op>
    struct BinaryTileOpPolicy<Op<Result, Left, Right, LeftConsumable,RightConsumable> > {
      typedef typename madness::if_c<LeftConsumable, Left&, const Left&>::type
          first_argument_type; ///< The left-hand argument type
      typedef typename madness::if_c<RightConsumable, Right&,
          const Right&>::type second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>&
          zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>&
          zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type
    }; // struct BinaryTileOpPolicy


    /// Binary tile operation interface base

    /// This base class defines binary operations with zero and non-zero tiles,
    /// and maps arguments given to the appropriate evaluation kernel.
    /// \tparam Derived The derived operation class type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable
    template <typename Derived, bool LeftConsumable, bool RightConsumable>
    class BinaryInterfaceBase {
    public:
      typedef typename BinaryTileOpPolicy<Derived>::first_argument_type
          first_argument_type; ///< The left-hand argument type
      typedef typename BinaryTileOpPolicy<Derived>::second_argument_type
          second_argument_type; ///< The right-hand argument type
      typedef typename BinaryTileOpPolicy<Derived>::zero_left_type
          zero_left_type; ///< Zero left-hand tile type
      typedef typename BinaryTileOpPolicy<Derived>::zero_right_type
          zero_right_type; ///< Zero right-hand tile type
      typedef typename BinaryTileOpPolicy<Derived>::result_type
          result_type; ///< The result tile type

    protected:

      /// Derived type accessor

      /// \return A const reference to the derived object
      const Derived& derived() const { return static_cast<const Derived&>(*this); }

    public:

      /// Evaluate two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c first and \c second .
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        if(derived().perm_.dim() > 1u)
          return derived().permute(first, second);

        return derived().template no_permute<LeftConsumable, RightConsumable>(first, second);
      }

      /// Evaluate a zero tile to a non-zero tiles and possibly permute

      /// \param second The right-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c first and \c second .
      result_type operator()(zero_left_type first, second_argument_type second) const {
        if(derived().perm_.dim() > 1)
          return derived().permute(first, second);

        return derived().template no_permute<LeftConsumable, RightConsumable>(first, second);
      }

      /// Evaluate a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c first and \c second .
      result_type operator()(first_argument_type first, zero_right_type second) const {
        if(derived().perm_.dim() > 1)
          return derived().permute(first, second);

        return derived().template no_permute<LeftConsumable, RightConsumable>(first, second);
      }

    }; // class BinaryInterfaceBase

    /// Binary tile operation interface

    /// In addition to the interface defined by \c BinaryInterfaceBase, this
    /// class defines binary operations with lazy tiles. It will evaluate
    /// arguments as necessary and pass them to the \c BinaryInterfaceBase
    /// interface functions.
    /// \tparam Derived The derived operation class type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable
    template <typename Derived, bool LeftConsumable, bool RightConsumable>
    class BinaryInterface : public BinaryInterfaceBase<Derived, LeftConsumable, RightConsumable> {
    public:
      typedef BinaryInterfaceBase<Derived, LeftConsumable, RightConsumable> BinaryInterfaceBase_;
      typedef typename BinaryInterfaceBase_::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename BinaryInterfaceBase_::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef typename BinaryInterfaceBase_::zero_left_type zero_left_type; ///< Zero left-hand tile type
      typedef typename BinaryInterfaceBase_::zero_right_type zero_right_type; ///< Zero right-hand tile type
      typedef typename BinaryInterfaceBase_::result_type result_type; ///< The result tile type

    private:

      // Import the derived accessor function
      using BinaryInterfaceBase_::derived;

    public:

      // Import interface of BinaryInterfaceBase
      using BinaryInterfaceBase_::operator();

      // The following operators will evaluate lazy tile and use the base class
      // interface functions to call the correct evaluation kernel.

      /// Evaluate two lazy tiles

      /// \tparam L The left-hand, lazy tile type
      /// \tparam R The right-hand, lazy tile type
      /// \param first The left-hand, lazy tile argument
      /// \param second The right-hand, lazy tile argument
      /// \return The result tile from the binary operation applied to the
      /// evaluated \c first and \c second .
      template <typename L, typename R>
      typename madness::enable_if_c<is_lazy_tile<L>::value && is_lazy_tile<R>::value,
          result_type>::type
      operator()(const L& first, const R& second) const {
        typename L::eval_type eval_first(first);
        typename R::eval_type eval_second(second);
        return operator()(eval_first, eval_second);
      }

      /// Evaluate lazy and non-lazy tiles

      /// \tparam L The left-hand, lazy tile type
      /// \tparam R The right-hand, non-lazy tile type
      /// \param first The left-hand, lazy tile argument
      /// \param second The right-hand, non-lazy tile argument
      /// \return The result tile from the binary operation applied to the
      /// evaluated \c first and \c second .
      template <typename L, typename R>
      typename madness::enable_if_c<
          is_lazy_tile<L>::value &&
          (! is_lazy_tile<typename std::remove_const<R>::type >::value),
          result_type>::type
      operator()(const L& first, R& second) const {
        typename L::eval_type eval_first(first);
        return operator()(eval_first, second);
      }

      /// Evaluate non-lazy and lazy tiles

      /// \tparam L The left-hand, non-lazy tile type
      /// \tparam R The right-hand, lazy tile type
      /// \param first The left-hand, non-lazy tile argument
      /// \param second The right-hand, lazy tile argument
      /// \return The result tile from the binary operation applied to the
      /// evaluated \c first and \c second .
      template <typename L, typename R>
      typename madness::enable_if_c<
          (! is_lazy_tile<typename std::remove_const<L>::type>::value) &&
          is_lazy_tile<R>::value,
          result_type>::type
      operator()(L& first, const R& second) const {
        typename R::eval_type eval_second(second);
        return operator()(first, eval_second);
      }

    }; // class BinaryInterface


    /// Binary tile operation interface

    /// In addition to the interface defined by \c BinaryInterfaceBase, this
    /// class defines binary operations with lazy tiles. It will evaluate
    /// arguments as necessary and pass them to the \c BinaryInterfaceBase
    /// interface functions. It also handles runtime consumable resources.
    /// \tparam Derived The derived operation class type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable
    template <typename Derived>
    class BinaryInterface<Derived, false, false> :
        public BinaryInterfaceBase<Derived, false, false>
    {
    public:
      typedef BinaryInterfaceBase<Derived, false, false> BinaryInterfaceBase_;
      typedef typename BinaryInterfaceBase_::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename BinaryInterfaceBase_::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef typename BinaryInterfaceBase_::zero_left_type zero_left_type; ///< Zero left-hand tile type
      typedef typename BinaryInterfaceBase_::zero_right_type zero_right_type; ///< Zero right-hand tile type
      typedef typename BinaryInterfaceBase_::result_type result_type; ///< The result tile type

    private:

      // Import the derived accessor function
      using BinaryInterfaceBase_::derived;

    public:

      // Import interface of BinaryInterfaceBase
      using BinaryInterfaceBase_::operator();

      template <typename L, typename R>
      typename madness::enable_if_c<is_array_tile<L>::value && is_array_tile<R>::value,
          result_type>::type
      operator()(const L& first, const R& second) const {
        typename L::eval_type eval_first(first);
        typename R::eval_type eval_second(second);

        if(derived().perm_.dim() > 1u)
          return derived().permute(eval_first, eval_second);

        if(first.is_consumable())
          return derived().template no_permute<true, false>(eval_first, eval_second);
        else if(second.is_consumable())
          return derived().template no_permute<false, true>(eval_first, eval_second);

        return derived().template no_permute<false, false>(eval_first, eval_second);
      }


      template <typename L, typename R>
      typename madness::enable_if_c<
          is_array_tile<L>::value &&
          (! is_lazy_tile<typename std::remove_const<R>::type>::value),
          result_type>::type
      operator()(const L& first, R& second) const {
        typename L::eval_type eval_first(first);

        if(derived().perm_.dim() > 1u)
          return derived().permute(eval_first, second);

        if(first.is_consumable())
          return derived().template no_permute<true, false>(eval_first, second);

        return derived().template no_permute<false, false>(eval_first, second);
      }


      template <typename L, typename R>
      typename madness::enable_if_c<
          (! is_lazy_tile<typename std::remove_const<L>::type>::value) &&
          is_array_tile<R>::value,
          result_type>::type
      operator()(L& first, const R& second) const {
        typename R::eval_type eval_second(second);

        if(derived().perm_.dim() > 1u)
          return derived().permute(first, eval_second);

        if(second.is_consumable())
          return derived().template no_permute<false, true>(first, eval_second);

        return derived().template no_permute<false, false>(first, eval_second);
      }

      template <typename L, typename R>
      typename madness::enable_if_c<
          is_non_array_lazy_tile<L>::value && is_non_array_lazy_tile<R>::value,
          result_type>::type
      operator()(const L& first, const R& second) const {
        typename L::eval_type eval_first(first);
        typename R::eval_type eval_second(second);
        return operator()(eval_first, eval_second);
      }


      template <typename L, typename R>
      typename madness::enable_if_c<
          is_non_array_lazy_tile<L>::value &&
          (! is_non_array_lazy_tile<typename std::remove_const<R>::type>::value),
          result_type>::type
      operator()(const L& first, R& second) const {
        typename L::eval_type eval_first(first);
        return operator()(eval_first, second);
      }


      template <typename L, typename R>
      typename madness::enable_if_c<
          (! is_non_array_lazy_tile<typename std::remove_const<L>::type>::value) &&
          is_non_array_lazy_tile<R>::value,
          result_type>::type
      operator()(L& first, const R& second) const {
        typename R::eval_type eval_second(second);
        return operator()(first, eval_second);
      }

    }; // class BinaryInterface

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_BINARY_INTERFACE_H__INCLUDED
