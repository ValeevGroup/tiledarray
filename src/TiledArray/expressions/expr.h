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
#include <TiledArray/tile_op/shift.h>

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename> struct ExprTrait;
    template <typename> class TsrExpr;
    template <typename> class BlkTsrExpr;


    /// Base class for expression evaluation

    /// \tparam Derived The derived class type
    template <typename Derived>
    class Expr {
    public:

      typedef Expr<Derived> Expr_; ///< This class type
      typedef Derived derived_type; ///< The derived object type
      typedef typename ExprTrait<Derived>::engine_type engine_type; ///< Expression engine type

    private:

      Expr<Derived>& operator=(const Expr<Derived>&);

      /// Task function used to evaluate lazy tiles

      /// \tparam R The result type
      /// \tparam T The lazy tile type
      /// \param tile The lazy tile
      /// \return The evaluated tile
      template <typename R, typename T>
      static typename TiledArray::eval_trait<T>::type eval_tile(const T& tile) {
        return tile;
      }


      /// Task function used to mutate result tiles tiles

      /// \tparam R The result type
      /// \tparam T The lazy tile type
      /// \tparam Op Tile operation type
      /// \param tile The lazy tile
      /// \return The evaluated tile
      /// \param op The tile mutating operation
      template <typename R, typename T, typename Op>
      static R eval_tile(T& tile, const std::shared_ptr<Op>& op) {
        return (*op)(tile);
      }

      /// Set an array tile with a lazy tile

      /// Spawn a task to evaluate a lazy tile and set the \a array tile at
      /// \c index with the result.
      /// \tparam A The array type
      /// \tparam I The index type
      /// \tparam T The lazy tile type
      /// \param array The result array
      /// \param index The tile index
      /// \param tile The lazy tile
      template <typename A, typename I, typename T>
      typename std::enable_if<TiledArray::detail::is_lazy_tile<T>::value>::type
      set_tile(A& array, const I index, const Future<T>& tile) const {
        array.set(index, array.get_world().taskq.add(
              & Expr_::template eval_tile<typename A::value_type, T>, tile));
      }

      /// Set the \c array tile at \c index with \c tile

      /// \tparam A The array type
      /// \tparam I The index type
      /// \tparam T The lazy tile type
      /// \param array The result array
      /// \param index The tile index
      /// \param tile The tile
      template <typename A, typename I, typename T>
      typename std::enable_if<! TiledArray::detail::is_lazy_tile<T>::value>::type
      set_tile(A& array, const I index, const Future<T>& tile) const {
        array.set(index, tile);
      }

      /// Set an array tile with a lazy tile

      /// Spawn a task to evaluate a lazy tile and set the \a array tile at
      /// \c index with the result.
      /// \tparam A The array type
      /// \tparam I The index type
      /// \tparam T The lazy tile type
      /// \tparam Op Tile operation type
      /// \param array The result array
      /// \param index The tile index
      /// \param tile The lazy tile
      /// \param op The tile mutating operation
      template <typename A, typename I, typename T, typename Op>
      void set_tile(A& array, const I index, const Future<T>& tile,
          const std::shared_ptr<Op>& op) const
      {
        array.set(index, array.get_world().taskq.add(
              & Expr_::template eval_tile<typename A::value_type, T, Op>, tile, op));
      }

    public:

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
        static_assert(! TiledArray::detail::is_lazy_tile<typename A::value_type>::value,
            "Assignment to an Array of lazy tiles is not supported.");

        // Get the target world.
        World& world = (tsr.array().is_initialized() ?
            tsr.array().get_world() :
            World::get_default());

        // Get the output process map.
        std::shared_ptr<typename TsrExpr<A>::array_type::pmap_interface> pmap;
        if(tsr.array().is_initialized())
          pmap = tsr.array().get_pmap();

        // Get result variable list.
        VariableList target_vars(tsr.vars());

        // Construct the expression engine
        engine_type engine(derived());
        engine.init(world, pmap, target_vars);

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
        dist_eval.eval();

        // Create the result array
        A result(dist_eval.get_world(), dist_eval.trange(),
            dist_eval.shape(), dist_eval.pmap());

        // Move the data from dist_eval into the result array. There is no
        // communication in this step.
        for(const auto index : *dist_eval.pmap()) {
          if(! dist_eval.is_zero(index))
            set_tile(result, index, dist_eval.get(index));
        }

        // Wait for child expressions of dist_eval
        dist_eval.wait();

        // Swap the new array with the result array object.
        result.swap(tsr.array());
      }


      /// Evaluate this object and assign it to \c tsr

      /// This expression is evaluated in parallel in distributed environments,
      /// where the content of \c tsr will be replace by the results of the
      /// evaluated tensor expression.
      /// \tparam A The array type
      /// \param tsr The tensor to be assigned
      template <typename A>
      void eval_to(BlkTsrExpr<A>& tsr) const {
        typedef TiledArray::math::Shift<typename A::value_type,
            typename EngineTrait<engine_type>::eval_type,
            EngineTrait<engine_type>::consumable> shift_op_type;
        static_assert(! TiledArray::detail::is_lazy_tile<typename A::value_type>::value,
            "Assignment to an Array of lazy tiles is not supported.");

#ifndef NDEBUG
        // Check that the array has been initialized.
        if(! tsr.array().is_initialized()) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "Assignment to an uninitialized Array sub-block is not supported.");
          }

          TA_EXCEPTION("Assignment to an uninitialized Array sub-block is not supported.");
        }

        // Note: Unfortunately we cannot check that the array tiles have been
        // set even though this is a requirement.
#endif // NDEBUG

        // Get the target world.
        World& world = tsr.array().get_world();

        // Get the output process map.
        std::shared_ptr<typename TsrExpr<A>::array_type::pmap_interface> pmap;

        // Get result variable list.
        VariableList target_vars(tsr.vars());

        // Construct the expression engine
        engine_type engine(derived());
        engine.init(world, pmap, target_vars);

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
        dist_eval.eval();

        // Create the result array
        A result(world, tsr.array().trange(),
            tsr.array().get_shape().update_block(tsr.lower_bound(), tsr.upper_bound(),
            dist_eval.shape()), tsr.array().get_pmap());

        // NOTE: The tiles from the original array and the sub-block are copied
        // in two separate steps because the two tensors have different data
        // distribution.

        // Copy tiles from the original array to the result array that are not
        // included in the sub-block assignment. There is no communication in
        // this step.
        const BlockRange blk_range(tsr.array().trange().tiles(),
            tsr.lower_bound(), tsr.upper_bound());
        for(const auto index : *tsr.array().get_pmap()) {
          if(! tsr.array().is_zero(index)) {
            if(! blk_range.includes(tsr.array().trange().tiles().idx(index)))
              result.set(index, tsr.array().find(index));
          }
        }

        // Move the data from dist_eval into the sub-block of result array.
        // This step may involve communication when the tiles are moved from the
        // sub-block distribution to the array distribution.
        {
          const std::vector<long> shift =
              tsr.array().trange().make_tile_range(tsr.lower_bound()).lobound();

          std::shared_ptr<shift_op_type> shift_op =
              std::make_shared<shift_op_type>(shift);

          for(const auto index : *dist_eval.pmap()) {
            if(! dist_eval.is_zero(index))
              set_tile(result, blk_range.ordinal(index), dist_eval.get(index), shift_op);
          }
        }

        // Wait for child expressions of dist_eval
        dist_eval.wait();

        // Swap the new array with the result array object.
        result.swap(tsr.array());
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

      template <typename T>
      using reduce_t = typename TiledArray::detail::scalar_type<typename ExprTrait<T>::engine_type::value_type>::type;

      template <typename Op>
      Future<typename Op::result_type>
      reduce(const Op& op, World& world = World::get_default()) const {
        // Typedefs
        typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag> key_type;
        typedef TiledArray::math::UnaryReduceWrapper<typename engine_type::value_type,
            Op> reduction_op_type;

        // Construct the expression engine
        engine_type engine(derived());
        engine.init(world, std::shared_ptr<typename engine_type::pmap_interface>(),
            VariableList());

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
        dist_eval.eval();

        // Create a local reduction task
        reduction_op_type wrapped_op(op);
        TiledArray::detail::ReduceTask<reduction_op_type> reduce_task(world, wrapped_op);

        // Move the data from dist_eval into the local reduction task
        typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
            dist_eval.pmap()->begin();
        const typename engine_type::dist_eval_type::pmap_interface::const_iterator end =
            dist_eval.pmap()->end();
        for(; it != end; ++it)
          if(! dist_eval.is_zero(*it))
            reduce_task.add(dist_eval.get(*it));

        // All reduce the result of the expression
        return world.gop.all_reduce(key_type(dist_eval.id()), reduce_task.submit(), op);
      }

      template <typename D, typename Op>
      Future<typename Op::result_type>
      reduce(const Expr<D>& right_expr, const Op& op,
          World& world = World::get_default()) const
      {
        // Typedefs
        typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag> key_type;
        typedef TiledArray::math::BinaryReduceWrapper<typename engine_type::value_type,
            typename D::engine_type::value_type, Op> reduction_op_type;

        // Evaluate this expression
        engine_type left_engine(derived());
        left_engine.init(world, std::shared_ptr<typename engine_type::pmap_interface>(),
            VariableList());

        // Create the distributed evaluator for this expression
        typename engine_type::dist_eval_type left_dist_eval =
            left_engine.make_dist_eval();
        left_dist_eval.eval();

        // Evaluate the right-hand expression
        typename D::engine_type right_engine(right_expr.derived());
        right_engine.init(world, left_engine.pmap(), left_engine.vars());

        // Create the distributed evaluator for the right-hand expression
        typename D::engine_type::dist_eval_type right_dist_eval =
            right_engine.make_dist_eval();
        right_dist_eval.eval();

#ifndef NDEBUG
        if(left_dist_eval.trange() != right_dist_eval.trange()) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The TiledRanges of the left- and right-hand arguments the binary reduction are not equal:" \
                << "\n    left  = " << left_dist_eval.trange() \
                << "\n    right = " << right_dist_eval.trange() );
          }

          TA_EXCEPTION("The TiledRange objects of a binary expression are not equal.");
        }
#endif // NDEBUG

        // Create a local reduction task
        reduction_op_type wrapped_op(op);
        TiledArray::detail::ReducePairTask<reduction_op_type>
            local_reduce_task(world, wrapped_op);

        // Move the data from dist_eval into the local reduction task
        typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
            left_dist_eval.pmap()->begin();
        const typename engine_type::dist_eval_type::pmap_interface::const_iterator end =
            left_dist_eval.pmap()->end();
        for(; it != end; ++it) {
          const typename engine_type::size_type index = *it;
          const bool left_not_zero = !left_dist_eval.is_zero(index);
          const bool right_not_zero = !right_dist_eval.is_zero(index);

          if(left_not_zero && right_not_zero) {
            local_reduce_task.add(left_dist_eval.get(index), right_dist_eval.get(index));
          } else {
            if(left_not_zero) left_dist_eval.get(index);
            if(right_not_zero) right_dist_eval.get(index);
          }
        }

        return world.gop.all_reduce(key_type(left_dist_eval.id()),
            local_reduce_task.submit(), op);
      }

      Future<reduce_t<Derived> >
      trace(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::TraceReduction<value_type>(), world);
      }

      Future<reduce_t<Derived> >
      sum(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::SumReduction<value_type>(), world);
      }

      Future<reduce_t<Derived> >
      product(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::ProductReduction<value_type>(), world);
      }

      Future<reduce_t<Derived> >
      squared_norm(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::SquaredNormReduction<value_type>(), world);
      }

    private:

      template <typename T>
      static T sqrt(const T t) { return std::sqrt(t); }

    public:

      Future<reduce_t<Derived> >
      norm(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::scalar_type scalar_type;
        return world.taskq.add(Expr_::template sqrt<scalar_type>, squared_norm(world));
      }

      Future<reduce_t<Derived> >
      min(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::MinReduction<value_type>(), world);
      }

      Future<reduce_t<Derived> >
      max(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::MaxReduction<value_type>(), world);
      }

      Future<reduce_t<Derived> >
      abs_min(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::AbsMinReduction<value_type>(), world);
      }

      Future<reduce_t<Derived> >
      abs_max(World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::AbsMaxReduction<value_type>(), world);
      }

      template <typename D>
      Future<reduce_t<Derived> >
      dot(const Expr<D>& right_expr, World& world = World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type left_value_type;
        typedef typename EngineTrait<typename D::engine_type>::eval_type right_value_type;
        return reduce(right_expr, TiledArray::math::DotReduction<left_value_type,
            right_value_type>(), world);
      }

    }; // class Expr

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED
