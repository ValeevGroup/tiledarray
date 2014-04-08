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
#include <TiledArray/tile_op/unary_reduction.h>
#include <TiledArray/tile_op/binary_reduction.h>
#include <TiledArray/tile_op/reduce_wrapper.h>

namespace TiledArray {

  // Forward declaration
  template <typename, unsigned int, typename, typename> class Array;

  namespace expressions {

    // Forward declaration
    template <typename> struct ExprTrait;


    /// Base class for expression evaluation

    /// \tparam Derived The derived class type
    template <typename Derived>
    class Expr {
    private:

      Expr<Derived>& operator=(const Expr<Derived>&);

    public:

      typedef Expr<Derived> Expr_; ///< This class type
      typedef Derived derived_type; ///< The derived object type
      typedef typename ExprTrait<Derived>::engine_type engine_type; ///< Expression engine type

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
      void eval_to(TsrExpr<A>& tsr) const {

        // Get the target world
        madness::World& world = (tsr.array().is_initialized() ?
            tsr.array().get_world() :
            madness::World::get_default());

        // Get the output process map
        std::shared_ptr<typename TsrExpr<A>::array_type::pmap_interface> pmap;
        if(tsr.array().is_initialized())
          pmap = tsr.array().get_pmap();


        // Construct the expression engine
        engine_type engine(derived());
        VariableList target_vars(tsr.vars());
        engine.init(world, pmap, target_vars);

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();

        // Create the result array
        typename TsrExpr<A>::array_type result(dist_eval.get_world(), dist_eval.trange(),
            dist_eval.shape(), dist_eval.pmap());

        // Move the data from disteval into the result array
        typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap()->begin();
        const typename engine_type::dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap()->end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            result.set(*it, dist_eval.move(*it));

        // Wait for child expressions of dist_eval
        dist_eval.wait();

        // Swap the new array with the result array object.
        tsr.array().swap(result);
      }


      template <typename T, unsigned int DIM, typename Tile, typename Policy>
      operator Array<T, DIM, Tile, Policy>() {
        typedef Array<T, DIM, Tile, Policy> array_type;

        // Construct the expression engine
        engine_type engine(derived());
        engine.init(madness::World::get_default(),
            std::shared_ptr<typename array_type::pmap_interface>(), VariableList());

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();

        // Create the result array
        array_type result(dist_eval.get_world(), dist_eval.trange(),
            dist_eval.shape(), dist_eval.pmap());

        // Move the data from disteval into the result array
        typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap()->begin();
        const typename engine_type::dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap()->end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            result.set(*it, dist_eval.move(*it));

        // Wait for child expressions of dist_eval
        dist_eval.wait();

        return result;
      }

      /// Expression print

      /// \param os The output stream
      /// \param target_vars The target variable list for this expression
      void print(ExprOStream& os, const VariableList& target_vars) const {
        // Construct the expression engine
        engine_type engine(derived());
        engine.init_vars(target_vars);
        engine.init_struct(target_vars);
        engine.print(os, target_vars);
      }

    private:

      struct ExpressionReduceTag { };

    public:

      template <typename Op>
      madness::Future<typename Op::result_type>
      reduce(const Op& op, madness::World& world = madness::World::get_default()) const {
        // Typedefs
        typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag> key_type;

        // Construct the expression engine
        engine_type engine(derived());
        engine.init(world, std::shared_ptr<typename engine_type::pmap_interface>(),
            VariableList());

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();

        // Create a local reduction task
        TiledArray::detail::ReduceTask<Op> reduce_task(world, op);

        // Move the data from dist_eval into the local reduction task
        typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap()->begin();
        const typename engine_type::dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap()->end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            reduce_task.add(dist_eval.move(*it));

        // All reduce the result of the expression
        return world.gop.all_reduce(key_type(dist_eval.id()), reduce_task.submit(), op);
      }

      template <typename D, typename Op>
      madness::Future<typename Op::result_type>
      reduce(const Expr<D>& right_expr, const Op& op,
          madness::World& world = madness::World::get_default()) const
      {
        // Typedefs
        typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag> key_type;

        // Evaluate this expression
        engine_type left_engine(derived());
        left_engine.init(world, std::shared_ptr<typename engine_type::pmap_interface>(),
            VariableList());

        // Create the distributed evaluator for this expression
        typename engine_type::dist_eval_type left_dist_eval =
            left_engine.make_dist_eval();

        // Evaluate the right-hand expression
        typename D::engine_type right_engine(right_expr.derived());
        right_engine.init(world, left_engine.pmap(), left_engine.vars());

        // Create the distributed evaluator for the right-hand expression
        typename engine_type::dist_eval_type right_dist_eval =
            right_engine.make_dist_eval();

        // Create a local reduction task
        TiledArray::detail::ReducePairTask<Op> local_reduce_task(world, op);

        // Move the data from dist_eval into the local reduction task
        typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
            left_dist_eval.pmap()->begin();
        const typename engine_type::dist_eval_type::pmap_interface::const_iterator end =
            left_dist_eval.pmap()->end();
        for(; it != end; ++it) {
          if(!left_dist_eval.is_zero(*it)) {
            madness::Future<typename engine_type::value_type> left_tile =
                left_dist_eval.move(*it);

            if(!right_dist_eval.is_zero(*it))
              local_reduce_task.add(left_tile, right_dist_eval.move(*it));
          } else {
            if(!right_dist_eval.is_zero(*it))
              right_dist_eval.move(*it);
          }
        }

        return world.gop.all_reduce(key_type(left_dist_eval.id()),
            local_reduce_task.submit(), op);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      sum(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::SumReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      product(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::ProductReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      squared_norm(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::SquaredNormReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

    private:

      template <typename T>
      static T sqrt(const T t) { return std::sqrt(t); }

    public:

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      norm(madness::World& world = madness::World::get_default()) const {
        return world.taskq.add(Expr_::template sqrt<
            typename engine_type::value_type::eval_type::numeric_type>,
            squared_norm(world));
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      min(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::MinReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      max(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::MaxReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      abs_min(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::AbsMinReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      abs_max(madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            TiledArray::math::AbsMaxReduction<typename engine_type::value_type::eval_type> >
            reduction_type;
        return reduce(reduction_type(), world);
      }

      template <typename D>
      madness::Future<typename ExprTrait<Derived>::scalar_type>
      dot(const Expr<D>& right_expr, madness::World& world = madness::World::get_default()) const {
        typedef TiledArray::math::BinaryReduceWrapper<typename engine_type::value_type,
            typename D::engine_type::value_type,
            TiledArray::math::DotReduction<typename engine_type::value_type::eval_type,
            typename D::engine_type::value_type::eval_type> > reduction_type;

        return reduce(right_expr, reduction_type(), world);
      }

    }; // class Expr

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED
