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
 *  scal_engine.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_ENGINE_H__INCLUDED

#include <TiledArray/expressions/unary_engine.h>
#include <TiledArray/tile_op/scal.h>

namespace TiledArray {
  namespace expressions {

    // Forward declarations
    template <typename> class ScalExpr;

    /// Scaling expression engine

    /// \tparam Arg The argument expression engine type
    template <typename Arg>
    class ScalEngine : public UnaryEngine<ScalEngine<Arg> > {
    public:
      // Class hierarchy typedefs
      typedef ScalEngine<Arg> ScalEngine_; ///< This class type
      typedef UnaryEngine<ScalEngine_ > UnaryEngine_; ///< Unary expression engine base type
      typedef typename UnaryEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base type

      // Argument typedefs
      typedef Arg argument_type; ///< The left-hand expression type

      // Operational typedefs
      typedef typename argument_type::eval_type value_type; ///< The result tile type
      typedef TiledArray::math::Scal<value_type, typename argument_type::value_type::eval_type,
          argument_type::consumable> op_type; ///< The tile operation type
      typedef typename op_type::scalar_type scalar_type; ///< The scaling factor type
      typedef typename argument_type::policy policy; ///< The result policy type
      typedef TiledArray::detail::DistEval<value_type, policy> dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename policy::size_type size_type; ///< Size type
      typedef typename policy::trange_type trange_type; ///< Tiled range type
      typedef typename policy::shape_type shape_type; ///< Shape type
      typedef typename policy::pmap_interface pmap_interface; ///< Process map interface type

    private:

      scalar_type factor_; ///< Scaling factor

    public:

      /// Constructor

      /// \param A The argument expression type
      /// \param expr The parent expression
      template <typename A>
      ScalEngine(const ScalExpr<A>& expr) : UnaryEngine_(expr), factor_(expr.factor()) { }

      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() const {
        return UnaryEngine_::arg().shape().scale(factor_);
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type make_shape(const Permutation& perm) const {
        return UnaryEngine_::left().shape().add(factor_, perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const { return op_type(factor_); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const { return op_type(perm, factor_); }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string make_tag() const {
        std::stringstream ss;
        ss << "[" << factor_ << "]";
        return ss.str();
      }

    }; // class ScalEngine


  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_ENGINE_H__INCLUDED
