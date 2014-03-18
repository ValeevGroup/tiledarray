/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  scal.h
 *  Mar 16, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_H__INCLUDED

#include <TiledArray/expressions/neg.h>
#include <TiledArray/tile_op/scale.h>

namespace TiledArray {
  namespace expressions {

    /// Scaled expression expression

    /// \tparam Arg The argument expression type
    template <typename Arg>
    class Scal : public Unary<Scal<Arg> > {
    public:
      typedef Unary<Scal<Arg> > Unary_; ///< Binary base class type
      typedef typename Unary_::Base_ Base_; ///< Base expression type
      typedef Arg argument_type; ///< The argument expression type
      typedef TiledArray::math::Scal<typename argument_type::eval_type,
          typename argument_type::eval_type, argument_type::consumable> op_type; ///< The tile operation type
      typedef typename op_type::scalar_type scalar_type; ///< The scalar type
      typedef typename op_type::result_type value_type; ///< The result tile type
      typedef typename argument_type::policy policy; ///< The result policy type
      typedef DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

    private:
      scalar_type factor_; ///< The scaling factor

      // Not allowed
      Scal<Arg>& operator=(const Scal<Arg>&);

    public:

      /// Scaled expression constructor

      /// \param arg The argument expression
      /// \param factor The scalar type
      Scal(const argument_type& arg, const scalar_type factor) :
        Unary_(arg), factor_(factor)
      { }

      /// Rescale expression constructor

      /// \param other The expression to be copied
      Scal(const Scal<Arg>& other, const scalar_type factor) :
        Unary_(other), factor_(other.factor_ * factor)
      { }

      /// Scale negated expression constructor

      /// \param other The expression to be copied
      Scal(const Neg<Arg>& other, const scalar_type factor) :
        Unary_(other.arg()), factor_(-factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      Scal(const Scal<Arg>& other) : Unary_(other), factor_(other.factor_) { }

      /// Non-permuting shape factory function

      /// \param left The shape of the left-hand argument
      /// \param right The shape of the right-hand argument
      /// \return The result shape
      static typename dist_eval_type::shape_type
      make_shape(const typename argument_type::shape_type& arg_shape) {
        return arg_shape.scale(factor_);
      }

      /// Permuting shape factory function

      /// \param left The shape of the left-hand argument
      /// \param right The shape of the right-hand argument
      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      static typename dist_eval_type::shape_type
      make_shape(const typename argument_type::shape_type& arg_shape, const Permutation& perm) {
        return arg_shape.scale(factor_, perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      static op_type make_tile_op() const { return op_type(factor_); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      static op_type make_tile_op(const Permutation& perm) const { return op_type(factor_, perm); }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string print_tag() const {
        std::stringstream ss;
        ss << "[" << factor_ << "]";
        return ss.str();
      }

    }; // class Scal

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_H__INCLUDED
