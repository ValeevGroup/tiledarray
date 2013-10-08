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

      /// Evaluate the result tile using the appropriate \c Derived class
      /// evaluation kernel.
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

      /// Evaluate the result tile using the appropriate \c Derived class
      /// evaluation kernel.
      /// \param second The right-hand argument
      /// \return The result tile from the binary operation applied to the
      /// \c first and \c second .
      result_type operator()(zero_left_type first, second_argument_type second) const {
        if(derived().perm_.dim() > 1)
          return derived().permute(first, second);

        return derived().template no_permute<LeftConsumable, RightConsumable>(first, second);
      }

      /// Evaluate a non-zero tiles to a zero tile and possibly permute

      /// Evaluate the result tile using the appropriate \c Derived class
      /// evaluation kernel.
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
    ///
    /// To use this interface class, a derived class needs to have the following
    /// form:
    /// \code
    /// template <typename Result, typename Left, typename Right, bool LeftConsumable,
    ///     bool RightConsumable>
    /// class Operation : public BinaryInterface<Operation<Result, Left, Right,
    ///     LeftConsumable, RightConsumable>, LeftConsumable, RightConsumable>
    /// {
    /// public:
    ///   typedef Operation<Result, Left, Right, LeftConsumable, RightConsumable> Operation_;
    ///   typedef BinaryInterface<Operation_, LeftConsumable, RightConsumable> BinaryInterface_;
    ///   typedef typename BinaryInterface_::first_argument_type first_argument_type;
    ///   typedef typename BinaryInterface_::second_argument_type second_argument_type;
    ///   typedef typename BinaryInterface_::zero_left_type zero_left_type;
    ///   typedef typename BinaryInterface_::zero_right_type zero_right_type;
    ///   typedef typename BinaryInterface_::result_type result_type;
    ///
    /// private:
    ///
    ///   // Make friends with base classes
    ///   friend class BinaryInterface<Operation_, LeftConsumable, RightConsumable>;
    ///   friend class BinaryInterfaceBase<Operation_, LeftConsumable, RightConsumable>;
    ///
    ///   // Permuting tile evaluation function
    ///
    ///   result_type permute(const Left& first, const Right& second) const {
    ///     // ...
    ///   }
    ///
    ///   result_type permute(zero_left_type, const Right& second) const {
    ///     // ...
    ///   }
    ///
    ///   result_type permute(const Left& first, zero_right_type) const {
    ///     // ...
    ///   }
    ///
    ///   // Non-permuting tile evaluation functions
    ///   // The compiler will select the correct functions based on the
    ///   // type ane consumability of the arguments.
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::disable_if_c<(LC && std::is_same<Result, Left>::value) ||
    ///       (RC && std::is_same<Result, Right>::value), result_type>::type
    ///   no_permute(const Left& first, const Right& second) {
    ///     // ...
    ///   }
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::enable_if_c<LC && std::is_same<Result, Left>::value, result_type>::type
    ///   no_permute(Left& first, const Right& second) {
    ///      // ...
    ///   }
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::enable_if_c<(RC && std::is_same<Result, Right>::value) &&
    ///       (!(LC && std::is_same<Result, Left>::value)), result_type>::type
    ///   no_permute(const Left& first, Right& second) {
    ///     // ...
    ///   }
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::disable_if_c<RC, result_type>::type
    ///   no_permute(zero_left_type, const Right& second) {
    ///     // ...
    ///   }
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::enable_if_c<RC, result_type>::type
    ///   no_permute(zero_left_type, Right& second) {
    ///     // ...
    ///   }
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::disable_if_c<LC, result_type>::type
    ///   no_permute(const Left& first, zero_right_type) {
    ///     // ...
    ///   }
    ///
    ///   template <bool LC, bool RC>
    ///   typename madness::enable_if_c<LC, result_type>::type
    ///   no_permute(Left& first, zero_right_type) {
    ///     // ...
    ///   }
    ///
    /// public:
    ///   // Implement default constructor, copy constructor and assignment operator
    ///
    ///   // Import interface from base class
    ///   using BinaryInterface_::operator();
    ///
    /// }; // class Operation
    /// \endcode
    /// \tparam Derived The derived operation class type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable
    template <typename Derived, bool LeftConsumable, bool RightConsumable>
    class BinaryInterface : public BinaryInterfaceBase<Derived, LeftConsumable, RightConsumable> {
    public:
      typedef BinaryInterfaceBase<Derived, LeftConsumable, RightConsumable> BinaryInterfaceBase_; ///< This class type
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

      /// This function will evaluate the \c first and \c second , then pass the
      /// evaluated tiles to the appropriate \c BinaryInterfaceBase_::operator()
      /// function.
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

      /// This function will evaluate the \c first , then pass the
      /// evaluated tile and \c second to the appropriate
      /// \c BinaryInterfaceBase_::operator() function.
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

      /// This function will evaluate the \c second , then pass the
      /// evaluated tile and \c first to the appropriate
      /// \c BinaryInterfaceBase_::operator() function.
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
    /// interface functions. This specialization is necessary to handle runtime
    /// consumable resources, when the tiles are not marked as consumable at
    /// compile time.
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
      typedef BinaryInterfaceBase<Derived, false, false> BinaryInterfaceBase_; ///< This class type
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

      /// Evaluate two lazy-array tiles

      /// This function will evaluate the \c first and \c second , then pass the
      /// evaluated tiles to the appropriate \c Derived class evaluation kernel.
      /// \tparam L The left-hand, lazy-array tile type
      /// \tparam R The right-hand, lazy-array tile type
      /// \param first The left-hand, non-lazy tile argument
      /// \param second The right-hand, lazy tile argument
      /// \return The result tile from the binary operation applied to the
      /// evaluated \c first and \c second .
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
