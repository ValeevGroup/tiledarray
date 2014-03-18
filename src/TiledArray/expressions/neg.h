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
 */

#ifndef TILEDARRAY_EXPRESSIONS_TSR_NEG_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_TSR_NEG_H__INCLUDED

#include <TiledArray/expressions/unary.h>
#include <TiledArray/dist_eval/unary_eval.h>
#include <TiledArray/tile_op/neg.h>

namespace TiledArray {
  namespace expressions {

    /// Addition expression

    /// \tparam Arg The argument expression type
    template <typename Arg>
    class Neg : public Unary<Neg<Arg> > {
    public:
      typedef Unary<Neg<Arg> > Unary_; ///< Unary base class type
      typedef typename Unary_::Base_ Base_; ///< Base expression type
      typedef Arg argument_type; ///< The argument expression type
      typedef typename argument_type::eval_type value_type; ///< The result tile type
      typedef TiledArray::math::Neg<value_type, typename argument_type::eval_type,
          argument_type::consumable> op_type; ///< The tile operation type
      typedef typename argument_type::policy policy; ///< The result policy type
      typedef DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

      /// Expression constructor

      /// \param arg The argument expression
      Neg(const argument_type& arg) : Unary_(arg) { }

      /// Copy constructor

      /// \param other The expression to be copied
      Neg(const Neg<Arg>& other) : Unary_(other) { }

      /// Non-permuting shape factory function

      /// \param arg_shape The shape of the argument
      /// \return The result shape
      static typename dist_eval_type::shape_type
      make_shape(const typename argument_type::shape_type& arg_shape) const {
        return arg_shape;
      }

      /// Permuting shape factory function

      /// \param arg_shape The shape of the argument
      /// \param perm The permutation to be applied to the argument
      /// \return The result shape
      static typename dist_eval_type::shape_type
      make_shape(const typename argument_type::shape_type& arg_shape, const Permutation& perm) const {
        return arg_shape.perm(perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      static op_type make_tile_op() const { return op_type(); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      static op_type make_tile_op(const Permutation& perm) const { return op_type(perm); }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      const char* print_tag() const { return "[-1]"; }

    }; // class Add

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_TSR_NEG_H__INCLUDED
