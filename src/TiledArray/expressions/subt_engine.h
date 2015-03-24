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
 *  subt_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_SUBT_ENGINE_H__INCLUDED
#define TILEDARRAY_SUBT_ENGINE_H__INCLUDED

#include <TiledArray/expressions/binary_engine.h>
#include <TiledArray/tile_op/subt.h>
#include <TiledArray/tile_op/scal_subt.h>

namespace TiledArray {
  namespace expressions {

    // Forward declarations
    template <typename, typename> class SubtExpr;
    template <typename, typename> class ScalSubtExpr;
    template <typename, typename> class SubtEngine;
    template <typename, typename> class ScalSubtEngine;

    template <typename Left, typename Right>
    struct EngineTrait<SubtEngine<Left, Right> > :
      public BinaryEngineTrait<Left, Right, TiledArray::math::Subt>
    { };

    template <typename Left, typename Right>
    struct EngineTrait<ScalSubtEngine<Left, Right> > :
      public BinaryEngineTrait<Left, Right, TiledArray::math::ScalSubt>
    { };


    /// Subtraction expression engine

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class SubtEngine : public BinaryEngine<SubtEngine<Left, Right> > {
    public:
      // Class hierarchy typedefs
      typedef SubtEngine<Left, Right> SubtEngine_; ///< This class type
      typedef BinaryEngine<SubtEngine<Left, Right> > BinaryEngine_; ///< Binary base class type
      typedef typename BinaryEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base type

      // Argument typedefs
      typedef typename EngineTrait<SubtEngine_>::left_type left_type; ///< The left-hand expression type
      typedef typename EngineTrait<SubtEngine_>::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<SubtEngine_>::value_type value_type; ///< The result tile type
      typedef typename EngineTrait<SubtEngine_>::op_type op_type; ///< The tile operation type
      typedef typename EngineTrait<SubtEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<SubtEngine_>::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<SubtEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<SubtEngine_>::trange_type trange_type; ///< Tiled range type
      typedef typename EngineTrait<SubtEngine_>::shape_type shape_type; ///< Shape type
      typedef typename EngineTrait<SubtEngine_>::pmap_interface pmap_interface; ///< Process map interface type

      /// Constructor

      /// \tparam L The left-hand argument expression type
      /// \tparam R The right-hand argument expression type
      /// \param expr The parent expression
      template <typename L, typename R>
      SubtEngine(const SubtExpr<L, R>& expr) : BinaryEngine_(expr) { }

      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() const {
        return BinaryEngine_::left_.shape().subt(BinaryEngine_::right_.shape());
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type make_shape(const Permutation& perm) const {
        return BinaryEngine_::left_.shape().subt(BinaryEngine_::right_.shape(), perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      static op_type make_tile_op() { return op_type(); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      static op_type make_tile_op(const Permutation& perm) { return op_type(perm); }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      const char* make_tag() const { return "[-] "; }

    }; // class SubtEngine


    /// Subtraction expression engine

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class ScalSubtEngine : public BinaryEngine<ScalSubtEngine<Left, Right> > {
    public:
      // Class hierarchy typedefs
      typedef ScalSubtEngine<Left, Right> ScalSubtEngine_; ///< This class type
      typedef BinaryEngine<ScalSubtEngine<Left, Right> > BinaryEngine_; ///< Binary expression engine base type
      typedef typename BinaryEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base type

      // Argument typedefs
      typedef typename EngineTrait<ScalSubtEngine_>::left_type left_type; ///< The left-hand expression type
      typedef typename EngineTrait<ScalSubtEngine_>::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<ScalSubtEngine_>::value_type value_type; ///< The result tile type
      typedef typename EngineTrait<ScalSubtEngine_>::scalar_type scalar_type; ///< Tile scalar type
      typedef typename EngineTrait<ScalSubtEngine_>::op_type op_type; ///< The tile operation type
      typedef typename EngineTrait<ScalSubtEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<ScalSubtEngine_>::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<ScalSubtEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<ScalSubtEngine_>::trange_type trange_type; ///< Tiled range type
      typedef typename EngineTrait<ScalSubtEngine_>::shape_type shape_type; ///< Shape type
      typedef typename EngineTrait<ScalSubtEngine_>::pmap_interface pmap_interface; ///< Process map interface type

    private:

      scalar_type factor_; ///< Scaling factor

    public:

      /// Constructor

      /// \tparam L The left-hand argument expression type
      /// \tparam R The right-hand argument expression type
      /// \param expr The parent expression
      template <typename L, typename R>
      ScalSubtEngine(const ScalSubtExpr<L, R>& expr) :
        BinaryEngine_(expr), factor_(expr.factor())
      { }

      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() const {
        return BinaryEngine_::left_.shape().subt(BinaryEngine_::right_.shape(),
            factor_);
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type make_shape(const Permutation& perm) const {
        return BinaryEngine_::left_.shape().subt(BinaryEngine_::right_.shape(),
            factor_, perm);
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
        ss << "[-] [" << factor_ << "] ";
        return ss.str();
      }

    }; // class ScalSubtEngine

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_SUBT_ENGINE_H__INCLUDED
