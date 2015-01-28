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
    public:

      typedef Expr<Derived> Expr_; ///< This class type
      typedef Derived derived_type; ///< The derived object type
      typedef typename ExprTrait<Derived>::engine_type engine_type; ///< Expression engine type

    private:

      Expr<Derived>& operator=(const Expr<Derived>&);

      /// Tile assignment functor

      /// This functor will set tiles of a distributed evaluator to an array,
      /// given a process map iterator. If the input tile is a lazy tile, then
      /// it is evaluated on the stack or a task is spawned to evaluate it when
      /// the tile is ready.
      /// \tparam A An \c Array type
      /// \tparam DE A distributed evaluator type
      template <typename A, typename DE>
      struct EvalTiles {
      private:
        A& array_; ///< The array object that will hold the result tiles
        DE& dist_eval_; ///< The distributed evaluator that holds the input tiles

        /// Task function used to evaluate lazy tiles

        /// \tparam Tile The lazy tile type
        /// \param The lazy tile
        /// \return The evaluated tile
        template <typename Tile>
        static typename Tile::eval_type eval_tile(const Tile& tile) {
          return tile;
        }

        /// Set an array tile from a lazy tile

        /// \tparam Tile The lazy tile type
        /// \param index The tile index
        /// \param tile The lazy tile
        template <typename Tile>
        typename madness::enable_if<TiledArray::math::is_lazy_tile<Tile> >::type
        set_tile(typename A::size_type index, const madness::Future<Tile>& tile) const {
          if(tile.probe()) {
            array_.set(index, tile.get());
          } else {
            array_.set(index, array_.get_world().taskq.add(
                EvalTiles::template eval_tile<Tile>, tile));
          }
        }

        /// Set an array tile

        /// \tparam Tile The tile type
        /// \param index The tile index
        /// \param tile The tile
        template <typename Tile>
        typename madness::disable_if<TiledArray::math::is_lazy_tile<Tile> >::type
        set_tile(typename A::size_type index, const madness::Future<Tile>& tile) const {
          array_.set(index, tile);
        }

      public:

        /// Constructor

        /// \param array The array object that will hold the result
        /// \param dist_eval The distributed evaluator that holds the input tiles
        EvalTiles(A& array, DE& dist_eval) :
          array_(array), dist_eval_(dist_eval)
        { }

        /// Copy constructor

        /// \param other The functor to be copied
        EvalTiles(const EvalTiles& other) :
          array_(other.array_), dist_eval_(other.dist_eval_)
        { }

        /// Set tile operator

        /// \param it An index iterator from a process map
        /// \return true
        bool operator()(const typename DE::pmap_interface::const_iterator& it) const {
          if(! dist_eval_.is_zero(*it)) {
            madness::Future<typename DE::value_type> tile = dist_eval_.get(*it);
            set_tile(*it, tile);
          }
          return true;
        }
      }; // struct EvalTiles

      /// Array factor function

      /// Construct an array that will hold the result of this expression
      /// \tparam A The output array type
      /// \param world The world that will hold the result
      /// \param pmap The process map for the result
      /// \param vars The target variable list
      template <typename A>
      A make_array(madness::World& world, const std::shared_ptr<typename A::pmap_interface>& pmap,
          const VariableList& target_vars) const
      {
        typedef madness::Range<typename engine_type::pmap_interface::const_iterator> range_type;

        // Construct the expression engine
        engine_type engine(derived());
        engine.init(world, pmap, target_vars);

        // Create the distributed evaluator from this expression
        typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
        dist_eval.eval();

        // Create the result array
        A result(dist_eval.get_world(), dist_eval.trange(),
            dist_eval.shape(), dist_eval.pmap());

        // Move the data from disteval into the result array
        int chuck_size = std::max<int>(1u,
            dist_eval.pmap()->local_size() / (madness::ThreadPool::size() + 1));
        world.taskq.for_each(range_type(dist_eval.pmap()->begin(), dist_eval.pmap()->end(), chuck_size),
            EvalTiles<A, typename engine_type::dist_eval_type>(result, dist_eval)).get();

        // Wait for child expressions of dist_eval
        dist_eval.wait();

        return result;
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

        // Get the target world.
        madness::World& world = (tsr.array().is_initialized() ?
            tsr.array().get_world() :
            madness::World::get_default());

        // Get the output process map.
        std::shared_ptr<typename TsrExpr<A>::array_type::pmap_interface> pmap;
        if(tsr.array().is_initialized())
          pmap = tsr.array().get_pmap();

        // Get result variable list.
        VariableList target_vars(tsr.vars());

        // Swap the new array with the result array object.
        make_array<A>(world, pmap, target_vars).swap(tsr.array());
      }

      /// Array conversion operator

      /// \tparam T The array element type
      /// \tparam DIM The array dimension
      /// \tparam Tile The array tile type
      /// \tparam Policy The array policy type
      /// \return A array object that holds the result of this expression
//      template <typename T, unsigned int DIM, typename Tile, typename Policy>
//      explicit operator Array<T, DIM, Tile, Policy>() {
//        return make_array<Array<T, DIM, Tile, Policy> >(madness::World::get_default(),
//            std::shared_ptr<typename Array<T, DIM, Tile, Policy>::pmap_interface>(),
//            VariableList());
//      }

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
      madness::Future<typename Op::result_type>
      reduce(const Expr<D>& right_expr, const Op& op,
          madness::World& world = madness::World::get_default()) const
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

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      trace(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::TraceReduction<value_type>(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      sum(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::SumReduction<value_type>(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      product(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::ProductReduction<value_type>(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      squared_norm(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::SquaredNormReduction<value_type>(), world);
      }

    private:

      template <typename T>
      static T sqrt(const T t) { return std::sqrt(t); }

    public:

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      norm(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::scalar_type scalar_type;
        return world.taskq.add(Expr_::template sqrt<scalar_type>, squared_norm(world));
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      min(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::MinReduction<value_type>(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      max(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::MaxReduction<value_type>(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      abs_min(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::AbsMinReduction<value_type>(), world);
      }

      madness::Future<typename ExprTrait<Derived>::scalar_type>
      abs_max(madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type value_type;
        return reduce(TiledArray::math::AbsMaxReduction<value_type>(), world);
      }

      template <typename D>
      madness::Future<typename ExprTrait<Derived>::scalar_type>
      dot(const Expr<D>& right_expr, madness::World& world = madness::World::get_default()) const {
        typedef typename EngineTrait<engine_type>::eval_type left_value_type;
        typedef typename EngineTrait<typename D::engine_type>::eval_type right_value_type;
        return reduce(right_expr, TiledArray::math::DotReduction<left_value_type,
            right_value_type>(), world);
      }

    }; // class Expr

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED
