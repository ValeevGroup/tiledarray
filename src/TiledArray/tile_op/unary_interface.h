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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  unary_interface.h
 *  Oct 7, 2013
 *
 */

#ifndef TILEDARRAY_UNARY_INTERFACE_H__INCLUDED
#define TILEDARRAY_UNARY_INTERFACE_H__INCLUDED

#include <TiledArray/tile_op/type_traits.h>

namespace TiledArray {
  namespace math {

    template <typename Op>
    struct UnaryTileOpPolicy;

    /// Policy class for binary tile operations

    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    /// \tparam Op The binary tile operation template
    template <typename Result, typename Arg, bool Consumable,
        template <typename, typename, bool> class Op>
    struct UnaryTileOpPolicy<Op<Result, Arg, Consumable> > {
      typedef typename madness::if_c<Consumable, Arg, const Arg>::type &
          argument_type; ///< The argument type
      typedef Result result_type; ///< The result tile type
    }; // struct UnaryTileOpPolicy


    /// Unary tile operation interface base

    /// This base class defines unary operations with zero or non-zero tiles,
    /// and maps arguments given to the appropriate evaluation kernel.
    /// \tparam Derived The derived operation class type
    /// \tparam Consumable A flag that is \c true when the argument is consumable
    template <typename Derived, bool Consumable>
    class UnaryInterface {
    public:
      typedef typename UnaryTileOpPolicy<Derived>::argument_type
          argument_type; ///< The argument type
      typedef typename UnaryTileOpPolicy<Derived>::result_type
          result_type; ///< The result tile type

    protected:

      /// Derived type accessor

      /// \return A const reference to the derived object
      const Derived& derived() const { return static_cast<const Derived&>(*this); }

    public:

      /// \param arg The argument
      /// \return A scaled and permuted \c arg
      result_type operator()(argument_type arg) const {
        if(derived().perm_.dim() > 1)
          return derived().permute(arg);

        return derived().template no_permute<Consumable>(arg);
      }

      template <typename A>
      typename madness::enable_if<is_lazy_tile<A>, result_type>::type
      operator()(const A& arg) const {
        typename A::eval_type eval_arg(arg);
        return operator()(eval_arg);
      }

      template <typename A>
      result_type operator()(A& arg, const bool) const {
        return operator()(arg);
      }

    }; // class UnaryInterfaceBase

    /// Unary tile operation interface base

    /// This base class defines unary operations with zero or non-zero tiles,
    /// and maps arguments given to the appropriate evaluation kernel.
    /// \tparam Derived The derived operation class type
    /// \tparam Consumable A flag that is \c true when the argument is consumable
    template <typename Derived>
    class UnaryInterface<Derived, false> {
    public:
      typedef typename UnaryTileOpPolicy<Derived>::argument_type
          argument_type; ///< The argument type
      typedef typename UnaryTileOpPolicy<Derived>::result_type
          result_type; ///< The result tile type

    protected:

      /// Derived type accessor

      /// \return A const reference to the derived object
      const Derived& derived() const { return static_cast<const Derived&>(*this); }

    public:

      /// \param arg The argument
      /// \return A scaled and permuted \c arg
      result_type operator()(argument_type arg) const {
        if(derived().perm_.dim() > 1)
          return derived().permute(arg);

        return derived().template no_permute<false>(arg);
      }

      template <typename A>
      typename madness::enable_if<is_non_array_lazy_tile<A>, result_type>::type
      operator()(const A& arg) const {
        typename A::eval_type eval_arg(arg);
        return operator()(eval_arg);
      }

      template <typename A>
      typename madness::enable_if<is_array_tile<A>, result_type>::type
      operator()(const A& arg) const {
        typename A::eval_type eval_arg(arg);
        return operator()(arg, arg.is_consumable());
      }

      template <typename A>
      typename madness::disable_if<is_lazy_tile<A>, result_type>::type
      operator()(const A& arg, const bool consume) const {
        operator()(arg);
      }

      template <typename A>
      typename madness::disable_if<is_lazy_tile<typename std::remove_const<A>::type>,
          result_type>::type
      operator()(A& arg, const bool consume) const {
        if(derived().perm_.dim() > 1)
          return derived().permute(arg);

        if(consume)
          return derived().template no_permute<true>(arg);

        return derived().template no_permute<false>(arg);
      }

      template <typename A>
      typename madness::enable_if<is_lazy_tile<A>, result_type>::type
      operator()(const A& arg, const bool consume) const {
        typename A::eval_type eval_arg(arg);
        return operator()(eval_arg, consume);
      }

    }; // class UnaryInterfaceBase

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_UNARY_INTERFACE_H__INCLUDED
