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
 *  add_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_ADD_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_ADD_ENGINE_H__INCLUDED

#include <TiledArray/expressions/binary_engine.h>
#include <TiledArray/tile_op/add.h>
#include <TiledArray/tile_op/scal_add.h>

namespace TiledArray {
  namespace expressions {

    // Forward declarations
    template <typename, typename> class AddExpr;
    template <typename, typename, typename> class ScalAddExpr;
    template <typename, typename> class AddEngine;
    template <typename, typename, typename> class ScalAddEngine;

    template <typename Left, typename Right>
    struct EngineTrait<AddEngine<Left, Right> > :
      public BinaryEngineTrait<Left, Right, void, TiledArray::math::Add>
    { };

    template <typename Left, typename Right, typename Scalar>
    struct EngineTrait<ScalAddEngine<Left, Right, Scalar> > :
      public BinaryEngineTrait<Left, Right, Scalar, TiledArray::math::ScalAdd>
    { };

    /// Addition expression engine

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class AddEngine : public BinaryEngine<AddEngine<Left, Right> > {
    public:
      // Class hierarchy typedefs
      typedef AddEngine<Left, Right> AddEngine_; ///< This class type
      typedef BinaryEngine<AddEngine_> BinaryEngine_; ///< Binary expression engine base type
      typedef typename BinaryEngine_::ExprEngine_ ExprEngine_; ///< Expression engine base type

      // Argument typedefs
      typedef typename EngineTrait<AddEngine_>::left_type left_type; ///< The left-hand expression type
      typedef typename EngineTrait<AddEngine_>::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<AddEngine_>::value_type value_type; ///< The result tile type
      typedef typename EngineTrait<AddEngine_>::op_type op_type; ///< The tile operation type
      typedef typename EngineTrait<AddEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<AddEngine_>::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<AddEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<AddEngine_>::trange_type trange_type; ///< Tiled range type
      typedef typename EngineTrait<AddEngine_>::shape_type shape_type; ///< Shape type
      typedef typename EngineTrait<AddEngine_>::pmap_interface pmap_interface; ///< Process map interface type

      /// Constructor

      /// \tparam L The left-hand argument expression type
      /// \tparam R The right-hand argument expression type
      /// \param expr The parent expression
      template <typename L, typename R>
      AddEngine(const AddExpr<L, R>& expr) : BinaryEngine_(expr) { }

      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() const {
        return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape());
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type make_shape(const Permutation& perm) const {
        return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape(), perm);
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
      const char* make_tag() const { return "[+] "; }

    }; // class AddEngine


    /// Addition expression engine

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar The scaling factor type
    template <typename Left, typename Right, typename Scalar>
    class ScalAddEngine : public BinaryEngine<ScalAddEngine<Left, Right, Scalar> > {
    public:
      // Class hierarchy typedefs
      typedef ScalAddEngine<Left, Right, Scalar> ScalAddEngine_; ///< This class type
      typedef BinaryEngine<ScalAddEngine_> BinaryEngine_; ///< Binary expression engine base type
      typedef ExprEngine<ScalAddEngine_> ExprEngine_; ///< Expression engine base type

      // Argument typedefs
      typedef typename EngineTrait<ScalAddEngine_>::left_type left_type; ///< The left-hand expression type
      typedef typename EngineTrait<ScalAddEngine_>::right_type right_type; ///< The right-hand expression type

      // Operational typedefs
      typedef typename EngineTrait<ScalAddEngine_>::value_type value_type; ///< The result tile type
      typedef typename EngineTrait<ScalAddEngine_>::scalar_type scalar_type; ///< Tile scalar type
      typedef typename EngineTrait<ScalAddEngine_>::op_type op_type; ///< The tile operation type
      typedef typename EngineTrait<ScalAddEngine_>::policy policy; ///< The result policy type
      typedef typename EngineTrait<ScalAddEngine_>::dist_eval_type dist_eval_type; ///< The distributed evaluator type

      // Meta data typedefs
      typedef typename EngineTrait<ScalAddEngine_>::size_type size_type; ///< Size type
      typedef typename EngineTrait<ScalAddEngine_>::trange_type trange_type; ///< Tiled range type
      typedef typename EngineTrait<ScalAddEngine_>::shape_type shape_type; ///< Shape type
      typedef typename EngineTrait<ScalAddEngine_>::pmap_interface pmap_interface; ///< Process map interface type

    private:

      scalar_type factor_; ///< Scaling factor

    public:

      /// Constructor

      /// \tparam L The left-hand argument expression type
      /// \tparam R The right-hand argument expression type
      /// \tparam S The expression scalar type
      /// \param expr The parent expression
      template <typename L, typename R, typename S>
      ScalAddEngine(const ScalAddExpr<L, R, S>& expr) :
        BinaryEngine_(expr), factor_(expr.factor())
      { }

      /// Non-permuting shape factory function

      /// \return The result shape
      shape_type make_shape() const {
        return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape(),
            factor_);
      }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      shape_type make_shape(const Permutation& perm) const {
        return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape(),
            factor_, perm);
      }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const { return op_type(factor_); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const { return op_type(perm, factor_); }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() { return factor_; }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string make_tag() const {
        std::stringstream ss;
        ss << "[+] [" << factor_ << "] ";
        return ss.str();
      }

    }; // class ScalAddEngine

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_ADD_ENGINE_H__INCLUDED
