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

#include "TiledArray/expressions/fwd.h"


#include "../reduce_task.h"
#include "../tile_interface/cast.h"
#include "../tile_interface/scale.h"
#include "../tile_op/binary_reduction.h"
#include "../tile_op/reduce_wrapper.h"
#include "../tile_op/shift.h"
#include "../tile_op/unary_reduction.h"
#include "../tile_op/unary_wrapper.h"
#include "TiledArray/config.h"
#include "TiledArray/tile.h"
#include "TiledArray/tile_interface/trace.h"
#include "expr_engine.h"
#ifdef TILEDARRAY_HAS_CUDA
#include <TiledArray/cuda/cuda_task_fn.h>
#include <TiledArray/external/cuda.h>
#endif

#include <TiledArray/tensor/type_traits.h>

namespace TiledArray::expressions {

template <typename Engine>
struct EngineParamOverride {
  EngineParamOverride() : world(nullptr), pmap(), shape(nullptr) {}

  typedef
      typename EngineTrait<Engine>::policy policy;  ///< The result policy type
  typedef typename EngineTrait<Engine>::shape_type
      shape_type;  ///< Tensor shape type
  typedef typename EngineTrait<Engine>::pmap_interface
      pmap_interface;  ///< Process map interface type

  World* world;
  std::shared_ptr<pmap_interface> pmap;
  const shape_type* shape;
};

/// \brief type trait checks if T has array() member
/// Useful to determine if an Expr is a TsrExpr or a related type
template <class E>
class has_array {
  /// true case
  template <class U>
  static auto __test(U* p) -> decltype(p->array(), std::true_type());
  /// false case
  template <class>
  static std::false_type __test(...);

 public:
  static constexpr const bool value =
      std::is_same<std::true_type, decltype(__test<E>(0))>::value;
};

/// Base class for expression evaluation

/// \tparam Derived The derived class type
template <typename Derived>
class Expr {
 public:
  template <typename Derived_ = Derived>
  using engine_t = typename ExprTrait<Derived_>::engine_type;
  template <typename Derived_ = Derived>
  using eval_type_t = typename engine_t<Derived_>::eval_type;

  typedef Expr<Derived> Expr_;                 ///< This class type
  typedef Derived derived_type;                ///< The derived object type
  typedef engine_t<derived_type> engine_type;  ///< Expression engine type

 private:
  template <typename D>
  friend class ExprEngine;

  typedef EngineParamOverride<engine_type>
      override_type;  ///< Expression engine parameters
  std::shared_ptr<override_type> override_ptr_;

 public:
  /// \param shape the shape to use for the result
  /// \internal \c shape is taken by const reference, but converted to a
  /// pointer; passing by const ref ensures lifetime management for temporary
  /// shapes
  Expr<Derived>& set_shape(typename override_type::shape_type const& shape) {
    if (override_ptr_ != nullptr) {
      override_ptr_->shape = &shape;
    } else {
      override_ptr_ = std::make_shared<override_type>();
      override_ptr_->shape = &shape;
    }
    return derived();
  }
  /// \param world the World object to use for the result
  Expr<Derived>& set_world(World& world) {
    if (override_ptr_ != nullptr) {
      override_ptr_->world = &world;
    } else {
      override_ptr_ = std::make_shared<override_type>();
      override_ptr_->world = &world;
    }
    return derived();
  }
  /// \param pmap the Pmap object to use for the result
  Expr<Derived>& set_pmap(
      const std::shared_ptr<typename override_type::pmap_interface> pmap) {
    if (override_ptr_) {
      override_ptr_->pmap = pmap;
    } else {
      override_ptr_ = std::make_shared<override_type>();
      override_ptr_->pmap = pmap;
    }
    return derived();
  }

 private:
  /// Task function used to evaluate a lazy tile and apply an op

  /// \tparam R The result type
  /// \tparam T A lazy tile type
  /// \tparam Op Tile operation type
  /// \param tile A forwarding reference to a lazy tile
  /// \param cast A const lvalue reference to the object that will cast the lazy
  /// tile to its result \param op A smart pointer to the Op object \return The
  /// evaluated tile
  template <typename R, typename T, typename C, typename Op>
  static auto eval_tile(T&& tile, const C& cast,
                        const std::shared_ptr<Op>& op) {
    auto&& cast_tile = cast(std::forward<T>(tile));
    return (*op)(cast_tile);
  }

  /// Task function used to mutate result tiles

  /// \tparam R The result type
  /// \tparam T The lazy tile type
  /// \tparam Op Tile operation type
  /// \param tile A forwarding reference to a lazy tile
  /// \return The evaluated tile
  /// \param op The tile mutating operation
  template <typename T, typename Op>
  static auto eval_tile(T&& tile, const std::shared_ptr<Op>& op) {
    return (*op)(std::forward<T>(tile));
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
  template <
      typename A, typename I, typename T,
      typename std::enable_if<!std::is_same<typename A::value_type, T>::value &&
                              is_lazy_tile<T>::value
#ifdef TILEDARRAY_HAS_CUDA
                              && !::TiledArray::detail::is_cuda_tile_v<T>
#endif
                              >::type* = nullptr>
  void set_tile(A& array, const I& index, const Future<T>& tile) const {
    array.set(index, array.world().taskq.add(
                         TiledArray::Cast<typename A::value_type, T>(), tile));
  }

#ifdef TILEDARRAY_HAS_CUDA
  /// Set an array tile with a lazy tile

  /// Spawn a task to evaluate a lazy tile and set the \a array tile at
  /// \c index with the result.
  /// \tparam A The array type
  /// \tparam I The index type
  /// \tparam T The lazy tile type
  /// \param array The result array
  /// \param index The tile index
  /// \param tile The lazy tile
  template <typename A, typename I, typename T,
            typename std::enable_if<
                !std::is_same<typename A::value_type, T>::value &&
                is_lazy_tile<T>::value &&
                ::TiledArray::detail::is_cuda_tile_v<T>>::type* = nullptr>
  void set_tile(A& array, const I& index, const Future<T>& tile) const {
    array.set(index, madness::add_cuda_task(
                         array.world(),
                         TiledArray::Cast<typename A::value_type, T>(), tile));
  }
#endif

  /// Set the \c array tile at \c index with \c tile

  /// \tparam A The array type
  /// \tparam I The index type
  /// \tparam T The lazy tile type
  /// \param array The result array
  /// \param index The tile index
  /// \param tile The tile
  template <typename A, typename I, typename T,
            typename std::enable_if<std::is_same<typename A::value_type,
                                                 T>::value>::type* = nullptr>
  void set_tile(A& array, const I& index, const Future<T>& tile) const {
    array.set(index, tile);
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
  template <
      typename A, typename I, typename T, typename Op,
      typename std::enable_if<!std::is_same<typename A::value_type, T>::value
#ifdef TILEDARRAY_HAS_CUDA
                              && !::TiledArray::detail::is_cuda_tile_v<T>
#endif
                              >::type* = nullptr>
  void set_tile(A& array, const I index, const Future<T>& tile,
                const std::shared_ptr<Op>& op) const {
    auto eval_tile_fn =
        &Expr_::template eval_tile<typename A::value_type, const T&,
                                   TiledArray::Cast<typename A::value_type, T>,
                                   Op>;
    array.set(index, array.world().taskq.add(
                         eval_tile_fn, tile,
                         TiledArray::Cast<typename A::value_type, T>(), op));
  }

#ifdef TILEDARRAY_HAS_CUDA
  /// Set an array tile with a lazy tile

  /// Spawn a task to evaluate a lazy tile and set the \a array tile at
  /// \c index with the result.
  /// \tparam A The array type
  /// \tparam I The index type
  /// \tparam T The lazy tile type
  /// \param array The result array
  /// \param index The tile index
  /// \param tile The lazy tile
  template <typename A, typename I, typename T, typename Op,
            typename std::enable_if<
                !std::is_same<typename A::value_type, T>::value &&
                ::TiledArray::detail::is_cuda_tile_v<T>>::type* = nullptr>
  void set_tile(A& array, const I index, const Future<T>& tile,
                const std::shared_ptr<Op>& op) const {
    auto eval_tile_fn =
        &Expr_::template eval_tile<typename A::value_type, const T&,
                                   TiledArray::Cast<typename A::value_type, T>,
                                   Op>;
    array.set(index, madness::add_cuda_task(
                         array.world(), eval_tile_fn, tile,
                         TiledArray::Cast<typename A::value_type, T>(), op));
  }
#endif

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
  template <
      typename A, typename I, typename T, typename Op,
      typename std::enable_if<std::is_same<typename A::value_type, T>::value
#ifdef TILEDARRAY_HAS_CUDA
                              && !::TiledArray::detail::is_cuda_tile_v<T>
#endif
                              >::type* = nullptr>
  void set_tile(A& array, const I index, const Future<T>& tile,
                const std::shared_ptr<Op>& op) const {
    auto eval_tile_fn_ptr = &Expr_::template eval_tile<const T&, Op>;
    using fn_ptr_type = decltype(eval_tile_fn_ptr);
    static_assert(madness::detail::function_traits<fn_ptr_type(
                      const T&, const std::shared_ptr<Op>&)>::value,
                  "ouch");
    array.set(index, array.world().taskq.add(eval_tile_fn_ptr, tile, op));
  }

#ifdef TILEDARRAY_HAS_CUDA

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
  template <typename A, typename I, typename T, typename Op,
            typename std::enable_if<
                std::is_same<typename A::value_type, T>::value&& ::TiledArray::
                    detail::is_cuda_tile_v<T>>::type* = nullptr>
  void set_tile(A& array, const I index, const Future<T>& tile,
                const std::shared_ptr<Op>& op) const {
    auto eval_tile_fn_ptr = &Expr_::template eval_tile<const T&, Op>;
    using fn_ptr_type = decltype(eval_tile_fn_ptr);
    static_assert(madness::detail::function_traits<fn_ptr_type(
                      const T&, const std::shared_ptr<Op>&)>::value,
                  "ouch");
    array.set(index, madness::add_cuda_task(array.world(), eval_tile_fn_ptr,
                                            tile, op));
  }
#endif

 public:
  // Compiler generated functions
  Expr() = default;
  Expr(const Expr_&) = default;
  Expr(Expr_&&) = default;
  ~Expr() = default;
  Expr_& operator=(const Expr_&) = delete;
  Expr_& operator=(Expr_&&) = delete;

  /// Cast this object to its derived type
  derived_type& derived() { return *static_cast<derived_type*>(this); }

  /// Cast this object to its derived type
  const derived_type& derived() const {
    return *static_cast<const derived_type*>(this);
  }

  /// Evaluate this object and assign it to \c tsr

  /// This expression is evaluated in parallel in distributed environments,
  /// where the content of \c tsr will be replaced by the results of the
  /// evaluated tensor expression.
  /// \tparam A The array type
  /// \tparam Alias Tile alias flag
  /// \param tsr The tensor to be assigned
  template <typename A, bool Alias>
  void eval_to(TsrExpr<A, Alias>& tsr) const {
    static_assert(!is_lazy_tile<typename A::value_type>::value,
                  "Assignment to an array of lazy tiles is not supported.");

    // Get the target world
    // 1. result's world is assigned, use it
    // 2. if this expression's world was assigned by set_world(), use it
    // 3. otherwise revert to the TA default for the MADNESS world
    const auto has_set_world = override_ptr_ && override_ptr_->world;
    World& world = (tsr.array().is_initialized()
                        ? tsr.array().world()
                        : (has_set_world ? *override_ptr_->world
                                         : TiledArray::get_default_world()));

    // Get the output process map.
    // If result's pmap is assigned use it as the initial guess
    // it will be assigned in engine.init
    std::shared_ptr<typename TsrExpr<A, Alias>::array_type::pmap_interface>
        pmap;
    if (tsr.array().is_initialized()) pmap = tsr.array().pmap();

    // Get result index list.
    BipartiteIndexList target_indices(tsr.annotation());

    // Construct the expression engine
    engine_type engine(derived());
    engine.init(world, pmap, target_indices);

    // Create the distributed evaluator from this expression
    typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
    dist_eval.eval();

    // Create the result array
    A result(dist_eval.world(), dist_eval.trange(), dist_eval.shape(),
             dist_eval.pmap());

    // Move the data from dist_eval into the result array. There is no
    // communication in this step.
    for (const auto index : *dist_eval.pmap()) {
      if (dist_eval.is_zero(index)) continue;
      auto tile_contents = dist_eval.get(index);
      set_tile(result, index, tile_contents);
    }

    // Wait for child expressions of dist_eval
    dist_eval.wait();
    // Swap the new array with the result array object.
    result.swap(tsr.array());
  }

  /// Evaluate this object and assign it to \c tsr

  /// This expression is evaluated in parallel in distributed environments,
  /// where the content of \c tsr will be replaced by the results of the
  /// evaluated tensor expression.
  /// \tparam A The array type
  /// \tparam Alias Tile alias flag
  /// \param tsr The tensor to be assigned
  template <typename A, bool Alias>
  void eval_to(BlkTsrExpr<A, Alias>& tsr) const {
    typedef TiledArray::detail::Shift<
        typename std::decay<A>::type::value_type,
        typename EngineTrait<engine_type>::eval_type,
        EngineTrait<engine_type>::consumable>
        shift_op_type;
    typedef TiledArray::detail::UnaryWrapper<shift_op_type> op_type;
    static_assert(!is_lazy_tile<typename A::value_type>::value,
                  "Assignment to an array of lazy tiles is not supported.");

#ifndef NDEBUG
    // Check that the array has been initialized.
    if (!tsr.array().is_initialized()) {
      if (TiledArray::get_default_world().rank() == 0) {
        TA_USER_ERROR_MESSAGE(
            "Assignment to an uninitialized array sub-block is not supported.");
      }

      TA_EXCEPTION(
          "Assignment to an uninitialized array sub-block is not supported.");
    }

    // Note: Unfortunately we cannot check that the array tiles have been
    // set even though this is a requirement.
#endif  // NDEBUG

    // Get the target world.
    World& world = tsr.array().world();

    // Get the output process map.
    std::shared_ptr<typename BlkTsrExpr<A, Alias>::array_type::pmap_interface>
        pmap;

    // Get result index list.
    BipartiteIndexList target_indices(tsr.annotation());

    // Construct the expression engine
    engine_type engine(derived());
    engine.init(world, pmap, target_indices);

    // Create the distributed evaluator from this expression
    typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
    dist_eval.eval();

    // Create the result array
    A result(world, tsr.array().trange(),
             tsr.array().shape().update_block(
                 tsr.lower_bound(), tsr.upper_bound(), dist_eval.shape()),
             tsr.array().pmap());

    // NOTE: The tiles from the original array and the sub-block are copied
    // in two separate steps because the two tensors have different data
    // distribution.

    // Copy tiles from the original array to the result array that are not
    // included in the sub-block assignment. There is no communication in
    // this step.
    const BlockRange blk_range(tsr.array().trange().tiles_range(),
                               tsr.lower_bound(), tsr.upper_bound());
    for (const auto index : *tsr.array().pmap()) {
      if (!tsr.array().is_zero(index)) {
        if (!blk_range.includes(tsr.array().trange().tiles_range().idx(index)))
          result.set(index, tsr.array().find(index));
      }
    }

    // Move the data from dist_eval into the sub-block of result array.
    // This step may involve communication when the tiles are moved from the
    // sub-block distribution to the array distribution.
    {
      // N.B. must deep copy
      const container::svector<long> shift =
          tsr.array().trange().make_tile_range(tsr.lower_bound()).lobound();

      std::shared_ptr<op_type> shift_op =
          std::make_shared<op_type>(shift_op_type(shift));

      for (const auto index : *dist_eval.pmap()) {
        if (!dist_eval.is_zero(index))
          set_tile(result, blk_range.ordinal(index), dist_eval.get(index),
                   shift_op);
      }
    }

    // Wait for child expressions of dist_eval
    dist_eval.wait();
    // Swap the new array with the result array object.
    result.swap(tsr.array());
  }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream& os, const BipartiteIndexList& target_indices) const {
    // Construct the expression engine
    engine_type engine(derived());
    engine.init_indices(target_indices);
    engine.init_struct(target_indices);
    engine.print(os, target_indices);
  }

 private:
  struct ExpressionReduceTag {};

  template <typename D, typename Enabler = void>
  struct default_world_helper {
    default_world_helper(const D&) {}
    World& get() const { return TiledArray::get_default_world(); }
  };
  template <typename D>
  struct default_world_helper<
      D, typename std::enable_if<has_array<D>::value>::type> {
    default_world_helper(const D& d) : derived_(d) {}
    World& get() const { return derived_.array().world(); }
    const D& derived_;
  };
  World& default_world() const {
    return default_world_helper<Derived>(this->derived()).get();
  }

 public:
  template <typename Op>
  Future<typename Op::result_type> reduce(const Op& op, World& world) const {
    // Typedefs
    typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag>
        key_type;
    typedef TiledArray::math::UnaryReduceWrapper<
        typename engine_type::value_type, Op>
        reduction_op_type;

    // Construct the expression engine
    engine_type engine(derived());
    engine.init(world, std::shared_ptr<typename engine_type::pmap_interface>(),
                BipartiteIndexList());

    // Create the distributed evaluator from this expression
    typename engine_type::dist_eval_type dist_eval = engine.make_dist_eval();
    dist_eval.eval();

    // Create a local reduction task
    reduction_op_type wrapped_op(op);
    TiledArray::detail::ReduceTask<reduction_op_type> reduce_task(world,
                                                                  wrapped_op);

    // Move the data from dist_eval into the local reduction task
    typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
        dist_eval.pmap()->begin();
    const typename engine_type::dist_eval_type::pmap_interface::const_iterator
        end = dist_eval.pmap()->end();
    for (; it != end; ++it)
      if (!dist_eval.is_zero(*it)) reduce_task.add(dist_eval.get(*it));

    // All reduce the result of the expression
    auto result = world.gop.all_reduce(key_type(dist_eval.id()),
                                       reduce_task.submit(), op);
    dist_eval.wait();
    return result;
  }

  template <typename Op>
  Future<typename Op::result_type> reduce(const Op& op) const {
    return reduce(op, default_world());
  }

  template <typename D, typename Op>
  Future<typename Op::result_type> reduce(const Expr<D>& right_expr,
                                          const Op& op, World& world) const {
    static_assert(
        is_aliased<D>::value,
        "no_alias() expressions are not allowed on the right-hand side of "
        "the assignment operator.");

    // Typedefs
    typedef madness::TaggedKey<madness::uniqueidT, ExpressionReduceTag>
        key_type;
    typedef TiledArray::math::BinaryReduceWrapper<
        typename engine_type::value_type, typename D::engine_type::value_type,
        Op>
        reduction_op_type;

    // Evaluate this expression
    engine_type left_engine(derived());
    left_engine.init(world,
                     std::shared_ptr<typename engine_type::pmap_interface>(),
                     BipartiteIndexList());

    // Create the distributed evaluator for this expression
    typename engine_type::dist_eval_type left_dist_eval =
        left_engine.make_dist_eval();
    left_dist_eval.eval();

    // Evaluate the right-hand expression
    typename D::engine_type right_engine(right_expr.derived());
    right_engine.init(world, left_engine.pmap(), left_engine.indices());

    // Create the distributed evaluator for the right-hand expression
    typename D::engine_type::dist_eval_type right_dist_eval =
        right_engine.make_dist_eval();
    right_dist_eval.eval();

#ifndef NDEBUG
    if (left_dist_eval.trange() != right_dist_eval.trange()) {
      if (TiledArray::get_default_world().rank() == 0) {
        TA_USER_ERROR_MESSAGE(
            "The TiledRanges of the left- and right-hand arguments the binary "
            "reduction are not equal:"
            << "\n    left  = " << left_dist_eval.trange()
            << "\n    right = " << right_dist_eval.trange());
      }

      TA_EXCEPTION(
          "The TiledRange objects of a binary expression are not equal.");
    }
#endif  // NDEBUG

    // Create a local reduction task
    reduction_op_type wrapped_op(op);
    TiledArray::detail::ReducePairTask<reduction_op_type> local_reduce_task(
        world, wrapped_op);

    // Move the data from dist_eval into the local reduction task
    typename engine_type::dist_eval_type::pmap_interface::const_iterator it =
        left_dist_eval.pmap()->begin();
    const typename engine_type::dist_eval_type::pmap_interface::const_iterator
        end = left_dist_eval.pmap()->end();
    for (; it != end; ++it) {
      const auto index = *it;
      const bool left_not_zero = !left_dist_eval.is_zero(index);
      const bool right_not_zero = !right_dist_eval.is_zero(index);

      if (left_not_zero && right_not_zero) {
        local_reduce_task.add(left_dist_eval.get(index),
                              right_dist_eval.get(index));
      } else {
        if (left_not_zero) left_dist_eval.get(index);
        if (right_not_zero) right_dist_eval.get(index);
      }
    }

    auto result = world.gop.all_reduce(key_type(left_dist_eval.id()),
                                       local_reduce_task.submit(), op);
    left_dist_eval.wait();
    right_dist_eval.wait();
    return result;
  }

  template <typename D, typename Op>
  Future<typename Op::result_type> reduce(const Expr<D>& right_expr,
                                          const Op& op) const {
    return reduce(right_expr, op, default_world());
  }

  template <
      typename TileType = typename EngineTrait<engine_type>::eval_type,
      typename = TiledArray::detail::enable_if_trace_is_defined_t<TileType>>
  Future<result_of_trace_t<TileType>> trace(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::TraceReduction<value_type>(), world);
  }

  template <
      typename TileType = typename EngineTrait<engine_type>::eval_type,
      typename = TiledArray::detail::enable_if_trace_is_defined_t<TileType>>
  Future<result_of_trace_t<TileType>> trace() const {
    return trace(default_world());
  }

  Future<typename TiledArray::SumReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  sum(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::SumReduction<value_type>(), world);
  }

  Future<typename TiledArray::SumReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  sum() const {
    return sum(default_world());
  }

  Future<typename TiledArray::ProductReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  product(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::ProductReduction<value_type>(), world);
  }

  Future<typename TiledArray::ProductReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  product() const {
    return product(default_world());
  }

  Future<typename TiledArray::SquaredNormReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  squared_norm(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::SquaredNormReduction<value_type>(), world);
  }

  Future<typename TiledArray::SquaredNormReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  squared_norm() const {
    return squared_norm(default_world());
  }

 private:
  template <typename T>
  static T sqrt(const T t) {
    return std::sqrt(t);
  }

 public:
  Future<typename TiledArray::SquaredNormReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  norm(World& world) const {
    return world.taskq.add(
        Expr_::template sqrt<typename TiledArray::SquaredNormReduction<
            typename EngineTrait<engine_type>::eval_type>::result_type>,
        squared_norm(world));
  }
  Future<typename TiledArray::SquaredNormReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  norm() const {
    return norm(default_world());
  }

  template <typename Derived_ = Derived>
  std::enable_if_t<
      TiledArray::detail::is_strictly_ordered<
          TiledArray::detail::numeric_t<typename EngineTrait<
              typename ExprTrait<Derived_>::engine_type>::eval_type>>::value,
      Future<typename TiledArray::MinReduction<typename EngineTrait<
          typename ExprTrait<Derived_>::engine_type>::eval_type>::result_type>>
  min(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::MinReduction<value_type>(), world);
  }

  template <typename Derived_ = Derived>
  std::enable_if_t<
      TiledArray::detail::is_strictly_ordered<
          TiledArray::detail::numeric_t<typename EngineTrait<
              typename ExprTrait<Derived_>::engine_type>::eval_type>>::value,
      Future<typename TiledArray::MinReduction<typename EngineTrait<
          typename ExprTrait<Derived_>::engine_type>::eval_type>::result_type>>
  min() const {
    return min(default_world());
  }

  template <typename Derived_ = Derived>
  std::enable_if_t<
      TiledArray::detail::is_strictly_ordered<
          TiledArray::detail::numeric_t<typename EngineTrait<
              typename ExprTrait<Derived_>::engine_type>::eval_type>>::value,
      Future<typename TiledArray::MaxReduction<typename EngineTrait<
          typename ExprTrait<Derived_>::engine_type>::eval_type>::result_type>>
  max(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::MaxReduction<value_type>(), world);
  }

  template <typename Derived_ = Derived>
  std::enable_if_t<
      TiledArray::detail::is_strictly_ordered<
          TiledArray::detail::numeric_t<typename EngineTrait<
              typename ExprTrait<Derived_>::engine_type>::eval_type>>::value,
      Future<typename TiledArray::MaxReduction<typename EngineTrait<
          typename ExprTrait<Derived_>::engine_type>::eval_type>::result_type>>
  max() const {
    return max(default_world());
  }

  Future<typename TiledArray::AbsMinReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  abs_min(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::AbsMinReduction<value_type>(), world);
  }

  Future<typename TiledArray::AbsMinReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  abs_min() const {
    return abs_min(default_world());
  }

  Future<typename TiledArray::AbsMaxReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  abs_max(World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type value_type;
    return reduce(TiledArray::AbsMaxReduction<value_type>(), world);
  }

  Future<typename TiledArray::AbsMaxReduction<
      typename EngineTrait<engine_type>::eval_type>::result_type>
  abs_max() const {
    return abs_max(default_world());
  }

  template <typename D>
  Future<typename TiledArray::DotReduction<
      typename EngineTrait<engine_type>::eval_type,
      typename EngineTrait<typename D::engine_type>::eval_type>::result_type>
  dot(const Expr<D>& right_expr, World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type left_value_type;
    typedef typename EngineTrait<typename D::engine_type>::eval_type
        right_value_type;
    return reduce(right_expr,
                  TiledArray::DotReduction<left_value_type, right_value_type>(),
                  world);
  }

  template <typename D>
  Future<typename TiledArray::DotReduction<
      typename EngineTrait<engine_type>::eval_type,
      typename EngineTrait<typename D::engine_type>::eval_type>::result_type>
  dot(const Expr<D>& right_expr) const {
    return dot(right_expr, default_world());
  }

  template <typename D>
  Future<typename TiledArray::InnerProductReduction<
      typename EngineTrait<engine_type>::eval_type,
      typename EngineTrait<typename D::engine_type>::eval_type>::result_type>
  inner_product(const Expr<D>& right_expr, World& world) const {
    typedef typename EngineTrait<engine_type>::eval_type left_value_type;
    typedef typename EngineTrait<typename D::engine_type>::eval_type
        right_value_type;
    return reduce(
        right_expr,
        TiledArray::InnerProductReduction<left_value_type, right_value_type>(),
        world);
  }

  template <typename D>
  Future<typename TiledArray::InnerProductReduction<
      typename EngineTrait<engine_type>::eval_type,
      typename EngineTrait<typename D::engine_type>::eval_type>::result_type>
  inner_product(const Expr<D>& right_expr) const {
    return inner_product(right_expr, default_world());
  }

};  // class Expr

}

#endif  // TILEDARRAY_EXPRESSIONS_EXPR_H__INCLUDED
