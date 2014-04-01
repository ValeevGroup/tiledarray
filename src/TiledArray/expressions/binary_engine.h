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
 *  binary_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_BINARY_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BINARY_ENGINE_H__INCLUDED

#include <TiledArray/expressions/expr_engine.h>
#include <TiledArray/dist_eval/binary_eval.h>

namespace TiledArray {
  namespace expressions {

    // Forward declarations
    template <typename> class BinaryExpr;

    template <typename, typename>
    struct BinaryExprPolicyHelper { }; // Different policy types is an error.

    template <typename Policy>
    struct BinaryExprPolicyHelper<Policy, Policy> {
      typedef Policy policy;
    };

    template <typename Left, typename Right>
    struct BinaryExprPolicy :
        public BinaryExprPolicyHelper<typename Left::policy, typename Right::policy>
    { };


    template <typename Derived>
    class BinaryEngine : ExprEngine<Derived> {
    public:
      // Class hierarchy typedefs
      typedef BinaryEngine<Derived> BinaryEngine_; ///< This class type
      typedef ExprEngine<Derived> ExprEngine_; ///< Base class type

      // Argument typedefs
      typedef typename Derived::left_type left_type; ///< The left-hand expression type
      typedef typename Derived::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename Derived::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      // Meta data typedefs
      typedef typename Derived::size_type size_type; ///< This expression's distributed evaluator type
      typedef typename Derived::trange_type trange_type; ///< This expression's distributed evaluator type
      typedef typename Derived::shape_type shape_type; ///< This expression's distributed evaluator type
      typedef typename Derived::pmap_interface pmap_interface; ///< This expression's distributed evaluator type

      static const bool consumable = true;
      static const unsigned int leaves = left_type::leaves + right_type::leaves;

    protected:

      // Import base class variables to this scope
      using ExprEngine_::vars_;

      left_type left_; ///< The left-hand argument
      right_type right_; ///< The right-hand argument

    public:

      template <typename D>
      BinaryEngine(const BinaryExpr<D>& expr) :
        ExprEngine_(), left_(expr.left()), right_(expr.right())
      { }

      // Pull base class functions into this class.
      using ExprEngine_::derived;

      /// Set the variable list for this expression

      /// This function will set the variable list for this expression and its
      /// children such that the number of permutations is minimized. The final
      /// variable list may not be set to target, which indicates that the
      /// result of this expression will be permuted to match \c target_vars.
      /// \param target_vars The target variable list for this expression
      void vars(const VariableList& target_vars) {
        TA_ASSERT(ExprEngine_::permute_tiles());

        // Determine the equality of the variable lists
        bool left_target = true, right_target = true, left_right = true;
        for(unsigned int i = 0u; i < target_vars.dim(); ++i) {
          left_target = left_target && left_.vars()[i] == target_vars[i];
          right_target = right_target && right_.vars()[i] == target_vars[i];
          left_right = left_right && left_.vars()[i] == right_.vars()[i];
        }

        if(left_right) {
          vars_ = left_.vars();
        } else {
          // Determine which argument will be permuted
          const bool perm_left = (right_target || ((! (left_target || right_target))
              && (left_type::leaves <= right_type::leaves)));

          if(perm_left) {
            vars_ = right_.vars();
            left_.vars(right_.target());
          } else {
            vars_ = left_.vars();
            right_.vars(left_.vars());
          }
        }
      }

      /// Initialize the variable list of this expression

      /// \param target_vars The target variable list for this expression
      void init_vars(const VariableList& target_vars) {
        left_.init_vars(target_vars);
        right_.init_vars(target_vars);
        vars(target_vars);
      }


      /// Initialize the variable list of this expression
      void init_vars() {
        if(left_type::leaves <= right_type::leaves) {
          left_.init_vars();
          vars_ = left_.vars();
          right_.vars(vars_);
        } else {
          right_.init_vars();
          vars_ = right_.vars();
          left_.vars(vars_);
        }
      }

      /// Initialize result tensor structure

      /// This function will initialize the permutation, tiled range, and shape
      /// for the left-hand, right-hand, and result tensor.
      /// \param target_vars The target variable list for the result tensor
      void init_struct(const VariableList& target_vars) {
        left_.init_struct(ExprEngine_::vars());
        right_.init_struct(ExprEngine_::vars());
        TA_ASSERT(left_.trange() == right_.trange());
        ExprEngine_::init_struct(target_vars);
      }

      /// Initialize result tensor distribution

      /// This function will initialize the world and process map for the result
      /// tensor.
      /// \param world The world were the result will be distributed
      /// \param pmap The process map for the result tensor tiles
      void init_distribution(madness::World* world,
          const std::shared_ptr<pmap_interface>& pmap)
      {
        left_.init_distribution(world, pmap);
        right_.init_distribution(world, left_.pmap());
        ExprEngine_::init_distribution(world, left_.pmap());
      }

      /// Non-permuting tiled range factory function

      /// \return The result tiled range
      trange_type make_trange() const { return left_.trange(); }

      /// Permuting tiled range factory function

      /// \param perm The permutation to be applied to the tiled range
      /// \return The result shape
      trange_type make_trange(const Permutation& perm) const {
        return perm ^ left_.trange();
      }

      /// Construct the distributed evaluator for this expression

      /// \return The distributed evaluator that will evaluate this expression
      dist_eval_type make_dist_eval() const {
        typedef TiledArray::detail::BinaryEvalImpl<typename left_type::dist_eval_type,
            typename right_type::dist_eval_type, typename Derived::op_type,
            typename dist_eval_type::policy> binary_impl_type;

        // Construct left and right distributed evaluators
        const typename left_type::dist_eval_type left = left_.make_dist_eval();
        const typename right_type::dist_eval_type right = right_.make_dist_eval();

        // Construct the distributed evaluator type
        std::shared_ptr<typename dist_eval_type::impl_type> pimpl(
            new binary_impl_type(left, right, *ExprEngine_::world(),
                ExprEngine_::trange(), ExprEngine_::shape(), ExprEngine_::pmap(),
                ExprEngine_::perm(), ExprEngine_::make_op()));

        return dist_eval_type(pimpl);
      }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream os, const VariableList& target_vars) const {
        ExprEngine_::print(os, target_vars);
        left_.print(os, vars_);
        right_.print(os, vars_);
      }
    }; // class BinaryEngine

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BINARY_ENGINE_H__INCLUDED
