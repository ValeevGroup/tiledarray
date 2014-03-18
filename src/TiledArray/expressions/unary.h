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

#ifndef TILEDARRAY_EXPRESSIONS_UNARY_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_UNARY_H__INCLUDED

#include <TiledArray/expressions/base.h>
#include <TiledArray/dist_eval/unary_eval.h>

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class Unary : public Base<Derived> {
    private:
      typedef Base<Derived> Base_;

    public:
      typedef typename Derived::argument_type argument_type; ///< The expression type
      typedef typename Derived::dist_eval_type dist_eval_type; ///< This expression's distributed evaluator type

      static const bool consumable = true;
      static const unsigned int leaves = argument_type::leaves;

    protected:
      argument_type arg_; ///< The argument expression

    public:
      /// Constructor

      /// \param arg The argument expression
      Unary(const argument_type& arg) const : Base_(arg.vars()), arg_(arg) { }


      /// Copy constructor

      /// \param other The expression to be copied
      Unary(const Unary<Derived>& other) const : Base_(other), arg_(other.arg_) { }

      // Pull base class functions into this class.
      using Base_::vars;

      /// Argument expression accessor

      /// \return A const reference to the argument expression object
      const argument_type& arg() const { return arg_; }

      /// Set the variable list for this expression

      /// This function will set the variable list for this expression and its
      /// children such that the number of permutations is minimized.
      /// \param target_vars The target variable list for this expression
      void vars(const VariableList& target_vars) {
        Base_::vars_ = target_vars;
        if(arg_.vars() != target_vars)
          arg_.vars(target_vars);
      }

      /// Initialize the variable list of this expression

      /// \param target_vars The target variable list for this expression
      void init_vars(const VariableList& target_vars) {
        arg_.init_vars(target_vars);
        vars(target_vars);
      }

      /// Initialize the variable list of this expression
      void init_vars() {
        arg_.init_vars();
        Base_::vars_ = arg_.vars();
      }

      /// Construct the distributed evaluator for this expression

      /// \param world The world where this expression will be evaluated
      /// \param target_vars The variable list for the output data layout
      /// \param pmap The process map for the output
      /// \return The distributed evaluator that will evaluate this expression
      dist_eval_type make_dist_eval(madness::World& world, const VariableList& target_vars,
          const std::shared_ptr<typename dist_eval_type::pmap_interface>& pmap) const
      {
        typedef UnaryEvalImpl<typename argument_type::dist_eval_type,
            typename Derived::op_type, typename dist_eval_type::policy> unary_impl_type;

        // Verify input
        TA_ASSERT(pmap->procs() == world.size());


        // Construct left and right distributed evaluators
        typename argument_type::dist_eval_type arg =
            arg_.make_dist_eval(world, Base_::vars_, pmap);

        // If the pmap provided was NULL, the use the pmap from the argument.
        if(! pmap)
          pmap = arg.pmap();

        // Construct the distributed evaluator type
        std::shared_ptr<typename dist_eval_type::impl_type> pimpl;
        if(Base_::vars_ != target_vars) {
          // Determine the permutation that will be applied to the result, if any.
          Permutation perm = target_vars.permutation(Base_::vars_);

          pimpl.reset(new unary_impl_type(arg, world, perm ^ arg.trange(),
              derived().make_shape(arg.shape(), perm), pmap, perm,
              derived().make_tile_op(perm)));
        } else {
          pimpl.reset(new unary_impl_type(arg, world, arg.trange(),
              derived().make_shape(arg.shape()), pmap, perm,
              derived().make_tile_op()));
        }

        return dist_eval_type(pimpl);
      }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream os, const VariableList& target_vars) const {
        if(target_vars != Base_::vars_) {
          const Permutation perm = target_vars.permutation(Base_::vars());
          os << "[P" << perm << "] " << derived().print_tag() << " " << Base_::vars_ << "\n";
        } else {
          os << derived().print_tag() << Base_::vars_ << "\n";
        }

        arg_.print(os, Base_::vars_);
      }

    }; // class Unary

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_UNARY_H__INCLUDED
