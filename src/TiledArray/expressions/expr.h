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
 *  expr.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED

#include <TiledArray/expressions/expr_engine.h>
#include <TiledArray/reduce_task.h>

namespace TiledArray {
  namespace expressions {

    /// Base class for expression evaluation

    /// \tparam Derived The derived class type
    template <typename Derived>
    class Expr {
    private:

      // Not allowed
      Expr<Derived>& operator=(const Expr<Derived>&);

    public:

      typedef Derived derived_type; ///< The derived object type
      typedef typename Derived::expr_engine expr_engine; ///< Expression data object

      /// Cast this object to it's derived type
      derived_type& derived() { return *static_cast<derived_type*>(this); }

      /// Cast this object to it's derived type
      const derived_type& derived() const { return *static_cast<const derived_type*>(this); }

      /// Evaluate this object and assign it to \c tsr

      /// This expression is evaluated in parallel in distributed environments,
      /// where the content of \c tsr will be replace by the results of the
      /// evaluated tensor expression.
      /// \tparam A The array type
      /// \param tsr The tensor to be assigned
      template <typename A>
      void eval_to(TsrExpr<A>& tsr) {

        // Get the target world
        madness::World& world = (tsr.array().is_initialized() ?
            tsr.array().get_world() :
            madness::World::get_default());

        // Get the output process map
        std::shared_ptr<typename TsrExpr<A>::array_type::pmap_interface> pmap;
        if(tsr.array().is_initialized())
          pmap = tsr.array().get_pmap();


        // Construct the expression engine
        expr_engine engine(derived());
        engine.init(world, pmap, tsr.vars());

        // Create the distributed evaluator from this expression
        typename expr_engine::dist_eval_type dist_eval = engine.make_dist_eval();

        // Create the result array
        typename TsrExpr<A>::array_type result(dist_eval.get_world(), dist_eval.trange(),
            dist_eval.shape(), dist_eval.pmap());

        // Move the data from disteval into the result array
        typename expr_engine::dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap().begin();
        const typename expr_engine::dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap().end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            result.set(*it, dist_eval.move(*it));

        // Wait for child expressions of dist_eval
        dist_eval.wait();

        // Swap the new array with the result array object.
        tsr.array().swap(result);
      }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream& os, const VariableList& target_vars) const {
        // Construct the expression engine
        expr_engine engine(derived());
        engine.init_vars(target_vars);
        engine.init_struct(target_vars);
        engine.print(os, target_vars);
      }

    private:

      struct ExpressionReduceTag { };

    public:

      template <typename Op>
      madness::Future<typename Op::result_type>
      reduce(const Op& op, madness::World& world = madness::World::get_default()) {
        // Typedefs
        typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag> key_type;

        // Construct the expression engine
        expr_engine engine(derived());
        engine.init(world, std::shared_ptr<typename expr_engine::pmap_interface>(),
            VariableList());

        // Create the distributed evaluator from this expression
        typename expr_engine::dist_eval_type dist_eval =
            derived().make_dist_eval();

        // Create a local reduction task
        TiledArray::detail::ReduceTask<Op> reduce_task(world, op);

        // Move the data from dist_eval into the local reduction task
        typename expr_engine::dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap().begin();
        const typename expr_engine::dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap().end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            reduce_task.add(dist_eval.move(*it));

        // All reduce the result of the expression
        return world.gop.all_reduce(key_type(dist_eval.id()), reduce_task.submit(), op);
      }

    }; // class Expr

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED
