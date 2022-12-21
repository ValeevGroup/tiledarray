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
 *  leaf_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_LEAF_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_LEAF_ENGINE_H__INCLUDED

#include <TiledArray/dist_eval/array_eval.h>
#include <TiledArray/expressions/expr_engine.h>

namespace TiledArray {
namespace expressions {

/// Leaf expression engine

/// \tparam Derived The derived class type
template <typename Derived>
class LeafEngine : public ExprEngine<Derived> {
 public:
  // Class hierarchy typedefs
  typedef LeafEngine<Derived> LeafEngine_;  ///< This class type
  typedef ExprEngine<Derived> ExprEngine_;  ///< Base class type

  // Argument typedefs
  typedef typename EngineTrait<Derived>::array_type
      array_type;  ///< The left-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<Derived>::value_type
      value_type;  ///< Tensor value type
  typedef
      typename EngineTrait<Derived>::op_type op_type;  ///< Tile operation type
  typedef
      typename EngineTrait<Derived>::policy policy;  ///< The result policy type
  typedef typename EngineTrait<Derived>::dist_eval_type
      dist_eval_type;  ///< This expression's distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<Derived>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<Derived>::trange_type
      trange_type;  ///< Tiled range type type
  typedef typename EngineTrait<Derived>::shape_type
      shape_type;  ///< Tensor shape type
  typedef typename EngineTrait<Derived>::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = EngineTrait<Derived>::consumable;
  static constexpr unsigned int leaves = EngineTrait<Derived>::leaves;

 protected:
  // Import base class variables to this scope
  using ExprEngine_::indices_;
  using ExprEngine_::perm_;
  using ExprEngine_::permute_tiles_;
  using ExprEngine_::pmap_;
  using ExprEngine_::shape_;
  using ExprEngine_::trange_;
  using ExprEngine_::world_;

  array_type array_;  ///< The array object

 public:
  /// Engine constructor

  /// \param expr The argument expression
  template <typename D>
  LeafEngine(const Expr<D>& expr)
      : ExprEngine_(expr), array_(expr.derived().array_value()) {
    indices_ = BipartiteIndexList(expr.derived().annotation());
  }

  // Import base class variables to this scope
  using ExprEngine_::derived;

  /// Set the index list for this expression

  /// This function is a noop since the index list is fixed.
  void perm_indices(const BipartiteIndexList&) {}

  /// Initialize the index list of this expression

  /// This function only checks for valid index lists.
  /// \param target_indices The target index list for this expression
  void init_indices(const BipartiteIndexList& target_indices) {
#ifndef NDEBUG
    if (!target_indices.is_permutation(indices_)) {
      if (TiledArray::get_default_world().rank() == 0) {
        TA_USER_ERROR_MESSAGE(
            "The array index list is not compatible with the expected "
            "output:"
            << "\n    expected = " << target_indices
            << "\n    array    = " << indices_);
      }

      TA_EXCEPTION(
          "Target index list is not a permutation of the given array index "
          "list "
          "list.");
    }
#endif  // NDEBUG
  }

  /// Initialize the index list of this expression

  /// This function is a noop since the index list is fixed.
  void init_indices() {}

  void init_distribution(World* world,
                         const std::shared_ptr<const pmap_interface>& pmap) {
    ExprEngine_::init_distribution(world, (pmap ? pmap : array_.pmap()));
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range
  trange_type make_trange() const { return array_.trange(); }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  trange_type make_trange(const Permutation& perm) const {
    return perm * array_.trange();
  }

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() { return array_.shape(); }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) {
    return array_.shape().perm(perm);
  }

  /// Construct the distributed evaluator for array
  dist_eval_type make_dist_eval() const {
    // Define the distributed evaluator implementation type
    typedef TiledArray::detail::ArrayEvalImpl<array_type, op_type, policy>
        impl_type;

    /// Create the pimpl for the distributed evaluator
    std::shared_ptr<impl_type> pimpl =
        std::make_shared<impl_type>(array_, *world_, trange_, shape_, pmap_,
                                    outer(perm_), ExprEngine_::make_op());

    return dist_eval_type(pimpl);
  }

};  // class LeafEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_LEAF_ENGINE_H__INCLUDED
