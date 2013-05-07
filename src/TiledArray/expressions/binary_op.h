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

#ifndef TILEDARRAY_EXPRESSIONS_BINARY_OP_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BINARY_OP_H__INCLUDED

#include <TiledArray/expressions/Binary_base.h>

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class BinaryOp : public BinaryBase<Derived> {
    private:
      typedef BinaryBase<Direvid> base; ///< Base class type

    public:
      typedef typename Derived::left_exp_type left_exp_type; ///< Left-hand argument expressions type
      typedef typename Derived::right_exp_type right_exp_type; ///< Right-hand argument expression type
      typedef typename Derived::expression_type expression_type; ///< Non-scaled expression type
      typedef typename Derived::scaled_expression_type scaled_expression_type; ///< Non-scaled expression type
      typedef typename base::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename base::shape_type shape_type; ///< The expression shape type
      typedef detail::BinaryEvalImpl<typename ExpLeft::eval_type,
          typename ExpRight::eval_type> eval_type; ///< Distributed evaluation type
      typedef math::ScalAdd<typename ExpLeft::value_type, typename ExpLeft::value_type,
          typename ExpRight::value_type, ExpLeft::is_consumable,
          ExpRight::is_consumable> op_type; ///< Tile operation type

      static const bool is_consumable = true; ///< Result tiles are consumable by other expressions

      /// Expression constructor

      /// Construct a scaled tensor addition expression from a tensor addition
      /// expression.
      /// \param exp The base expression
      /// \param factor The scaling factor applied to tiles in this operation
      ScalTsrAdd(const TsrAdd<ExpLeft, ExpRight>& exp, numeric_type factor) :
        base(exp, factor)
      { }

      /// Expression constructor

      /// Construct a scaled tensor addition expression from a tensor addition
      /// expression.
      /// \param exp The base expression
      /// \param factor The scaling factor applied to tiles in this operation
      ScalTsrAdd(const ScalTsrAdd<ExpLeft, ExpRight>& exp, numeric_type factor) :
        base(exp, factor)
      { }

      // Import base class function into this class namespace
      using base::derived;
      using base::vars;
      using base::permute;
      using base::left;
      using base::right;
      using base::factor;

      template <typename A>
      void eval_to(Tsr<A>& tsr) const {
      }

      eval_type eval(madness::World& world, const VariableList& vars,
          const std::shared_ptr<pmap_interface>& pmap)
      {
        // Get the argument variable lists
        const VariableList* left_vars = & left().vars();
        const VariableList* right_vars = & right().vars();

        // Determine the optimal variable list for arguments and the permutation
        // for this expression
        Permutation perm = permute(left_vars, right_vars);

        // Evaluate child expressions
        typename ExpLeft::eval_type left_eval = left().eval(world, *left_vars, pmap.clone());
        typename ExpRight::eval_type right_eval = right().eval(world, *right_vars, pmap.clone());

        // Evaluate the expression arguments
        typename ExpLeft::eval_type left_eval = left_.eval(*left_vars, pmap->clone());
        typename ExpRight::eval_type right_eval = right_.eval(*right_vars, pmap->clone());

        // Construct the shape
        shape_type shape = (!(left_eval.is_dense() || right_eval.is_dense()) ?
            left_eval.shape() & right_eval.shape() :
            shape_type(0));

        // Construct the tile evaluations operation
        op_type op(factor(), perm);

        // Construct the evaluation object for this operation
        return eval_type(left_eval, right_eval, world, perm, left_eval.trange(),
            shape, pmap, op, false);
      }

    }; // class ScalTsrAdd

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BINARY_OP_H__INCLUDED
