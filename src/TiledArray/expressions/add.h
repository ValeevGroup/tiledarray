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

#ifndef TILEDARRAY_EXPRESSIONS_ADD_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_ADD_H__INCLUDED

#include <TiledArray/expressions/binary.h>
#include <TiledArray/dist_eval/binary_eval.h>
#include <TiledArray/tile_op/add.h>
#include <TiledArray/tile_op/scal_add.h>

namespace TiledArray {
  namespace expressions {

    /// Addition expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class Add : public Binary<Add<Left, Right> > {
    public:
      typedef Binary<Add<Left, Right> > Binary_; ///< Binary base class type
      typedef typename Binary_::Base_ Base_; ///< Base expression type
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef TiledArray::math::Add<typename left_type::eval_type,
          typename left_type::eval_type, typename right_type::eval_type,
          left_type::consumable, right_type::consumable> op_type; ///< The tile operation type
      typedef typename op_type::result_type value_type; ///< The result tile type
      typedef typename BinaryExprPolicy<Left, Right>::policy policy; ///< The result policy type
      typedef DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

      /// Expression constructor

      /// \param left The left-hand expression
      /// \param right The right-hand expression
      Add(const left_type& left, const right_type& right) : Binary_(left, right) { }

      /// Copy constructor

      /// \param other The expression to be copied
      Add(const Add<Left, Right>& other) : Binary_(other) { }

      /// Non-permuting shape factory function

      /// \param left The shape of the left-hand argument
      /// \param right The shape of the right-hand argument
      /// \return The result shape
      static typename dist_eval_type::shape_type
      make_shape(const typename left_type::shape_type& left_shape,
          const typename right_type::shape_type& right_shape) {
        return left_shape.add(right_shape);
      }

      /// Permuting shape factory function

      /// \param left The shape of the left-hand argument
      /// \param right The shape of the right-hand argument
      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      static typename dist_eval_type::shape_type
      make_shape(const typename left_type::shape_type& left_shape,
          const typename right_type::shape_type& right_shape, const Permutation& perm) {
        return left_shape.add(right_shape, perm);
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
      const char* print_tag() const { return "[+]"; }

    }; // class Add


    /// Scaled addition expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class ScalAdd : public Binary<ScalAdd<Left, Right> > {
    public:
      typedef Binary<ScalAdd<Left, Right> > Binary_; ///< Binary base class type
      typedef typename Binary_::Base_ Base_; ///< Base expression type
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef TiledArray::math::ScalAdd<typename left_type::eval_type,
          typename left_type::eval_type, typename right_type::eval_type,
          left_type::consumable, right_type::consumable> op_type; ///< The tile operation type
      typedef typename op_type::scalar_type scalar_type; ///< The scaling factor type
      typedef typename op_type::result_type value_type; ///< The result tile type
      typedef typename BinaryExprPolicy<Left, Right>::policy policy; ///< The result policy type
      typedef BinaryEvalImpl<typename left_type::dist_eval_type,
          typename right_type::dist_eval_type, op_type, policy> binary_impl_type; ///< The distributed evaluator impl type
      typedef DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

    private:

      scalar_type factor_; ///< The scaling factor

    public:

      /// Expression constructor

      /// \param arg The non-scaled expression
      /// \param factor The scaling factor
      ScalAdd(const Add<Left, Right>& arg, const scalar_type factor) :
        Binary_(arg), factor_(factor)
      { }

      /// Expression constructor

      /// \param arg The scaled expression
      /// \param factor The scaling factor
      ScalAdd(const ScalAdd<Left, Right>& arg, const scalar_type factor) :
        Binary_(arg), factor_(factor * arg.factor_)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalAdd(const ScalAdd<Left, Right>& other) :
        Binary_(other), factor_(other.factor_)
      { }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

      /// Non-permuting shape factory function

      /// \param left The shape of the left-hand argument
      /// \param right The shape of the right-hand argument
      /// \return The result shape
      typename dist_eval_type::shape_type
      make_shape(const typename left_type::shape_type& left_shape,
          const typename right_type::shape_type& right_shape) {
        return left_shape.add(right_shape, factor_);
      }

      /// Permuting shape factory function

      /// \param left The shape of the left-hand argument
      /// \param right The shape of the right-hand argument
      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      typename dist_eval_type::shape_type
      make_shape(const typename left_type::shape_type& left_shape,
          const typename right_type::shape_type& right_shape, const Permutation& perm) {
        return left_shape.add(right_shape, factor_, perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const { return op_type(factor_); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const {
        return op_type(perm, factor_);
      }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string print_tag() const {
        std::stringstream ss;
        ss << "[+] [" << factor_ << "]";
        return ss.str();
      }

    }; // class ScalAdd

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_ADD_H__INCLUDED
