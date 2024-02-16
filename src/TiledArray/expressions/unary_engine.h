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
 *  unary_engine.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_UNARY_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_UNARY_ENGINE_H__INCLUDED

#include <TiledArray/dist_eval/unary_eval.h>
#include <TiledArray/expressions/expr_engine.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename>
class UnaryExpr;

template <typename Derived>
class UnaryEngine : ExprEngine<Derived> {
 public:
  // Class hierarchy typedefs
  typedef UnaryEngine<Derived> UnaryEngine_;  ///< This class type
  typedef ExprEngine<Derived> ExprEngine_;    ///< Base class type

  // Argument typedefs
  typedef typename EngineTrait<Derived>::argument_type
      argument_type;  ///< The argument expression engine type

  // Operational typedefs
  typedef typename EngineTrait<Derived>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<Derived>::op_type
      op_type;  ///< The tile operation type
  typedef
      typename EngineTrait<Derived>::policy policy;  ///< The result policy type
  typedef typename EngineTrait<Derived>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<Derived>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<Derived>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<Derived>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<Derived>::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = true;
  static constexpr unsigned int leaves = argument_type::leaves;

 protected:
  // Import base class variables to this scope
  using ExprEngine_::implicit_permute_inner_;
  using ExprEngine_::implicit_permute_outer_;
  using ExprEngine_::indices_;
  using ExprEngine_::perm_;
  using ExprEngine_::pmap_;
  using ExprEngine_::shape_;
  using ExprEngine_::trange_;
  using ExprEngine_::world_;

  argument_type arg_;  ///< The argument

 private:
  // Not allowed
  UnaryEngine(const UnaryEngine_&);
  UnaryEngine_& operator=(const UnaryEngine_&);

 public:
  template <typename D>
  UnaryEngine(const UnaryExpr<D>& expr) : ExprEngine_(expr), arg_(expr.arg()) {}

  // Pull base class functions into this class.
  using ExprEngine_::derived;
  using ExprEngine_::indices;

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized.
  /// \param target_indices The target index list for this expression
  void perm_indices(const BipartiteIndexList& target_indices) {
    TA_ASSERT(!this->implicit_permute());

    indices_ = target_indices;
    if (arg_.indices() != target_indices) arg_.perm_indices(target_indices);
  }

  /// Initialize the index list of this expression

  /// \param target_indices The target index list for this expression
  void init_indices(const BipartiteIndexList& target_indices) {
    arg_.init_indices(target_indices);
    perm_indices(target_indices);
  }

  /// Initialize the index list of this expression
  void init_indices() {
    arg_.init_indices();
    indices_ = arg_.indices();
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the left-hand, right-hand, and result tensor.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    arg_.init_struct(ExprEngine_::indices());
    ExprEngine_::init_struct(target_indices);
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world,
                         const std::shared_ptr<const pmap_interface>& pmap) {
    arg_.init_distribution(world, pmap);
    ExprEngine_::init_distribution(world, arg_.pmap());
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range
  trange_type make_trange() const { return arg_.trange(); }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the tiled range
  /// \return The result shape
  trange_type make_trange(const Permutation& perm) const {
    return perm ^ arg_.trange();
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    typedef TiledArray::detail::UnaryEvalImpl<
        typename argument_type::dist_eval_type, typename Derived::op_type,
        typename dist_eval_type::policy>
        impl_type;

    // Construct left and right distributed evaluators
    const typename argument_type::dist_eval_type arg = arg_.make_dist_eval();

    // Construct the distributed evaluator type
    std::shared_ptr<impl_type> pimpl = std::make_shared<impl_type>(
        arg, *world_, trange_, shape_, pmap_, perm_, ExprEngine_::make_op());

    return dist_eval_type(pimpl);
  }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_indices) const {
    ExprEngine_::print(os, target_indices);
    os.inc();
    arg_.print(os, indices_);
    os.dec();
  }

};  // class UnaryEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_UNARY_ENGINE_H__INCLUDED
