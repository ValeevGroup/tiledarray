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
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_TSR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_TSR_H__INCLUDED

#include <TiledArray/expressions/neg_tsr.h>
#include <TiledArray/tile_op/scal.h>

namespace TiledArray {
  namespace expressions {


    template <typename> class ScalTsr;

    /// Expression wrapper for scaled array objects

    /// \tparam T The array element type
    /// \tparam DIM The array dimension
    /// \tparam Tile The array tile type
    /// \tparam Policy The array policy type
    template <typename T, unsigned int DIM, typename Tile, typename Policy>
    class ScalTsr<Array<T, DIM, Tile, Policy> > :
        public TsrBase<ScalTsr<Array<T, DIM, Tile, Policy> > >
    {
    private:
      typedef ScalTsr<Array<T, DIM, Tile, Policy> > ScalTsr_;
      typedef TsrBase<ScalTsr_> TsrBase_;
      typedef typename TsrBase_::Base_ Base_;

    public:
      typedef Array<T, DIM, Tile, Policy> array_type; ///< The array type
      typedef TiledArray::math::Neg<typename array_type::eval_type,
          typename array_type::eval_type, false> op_type; ///< The tile operation
      typedef TiledArray::detail::LazyArrayTile<typename array_type::value_type,
          op_type> value_type;  ///< Tile type
      typedef typename op_type::scalar_type ///< The scalar type
      typedef Policy policy; ///< Policy type
      typedef DistEval<value_type, policy> dist_eval_type; ///< Distributed evaluator

    private:

      array_type array_; ///< The array that this expression
      scalar_type factor_; ///< The scaling factor

      // Not allowed
      ScalTsr_& operator=(ScalTsr_&);

    public:

      /// Construct a scaled tensor expression from a tensor expression

      /// \param other The tensor expression
      /// \param factor The scaling factor
      ScalTsr(const Tsr<array_type>& tsr, const scalar_type factor) :
        TsrBase_(tsr.vars()), array_(tsr.array()), factor_(factor)
      { }

      /// Construct a scaled tensor expression from a const tensor expression

      /// \param other The const tensor expression
      /// \param factor The scaling factor
      ScalTsr(const Tsr<const array_type>& tsr, const scalar_type factor) :
        TsrBase_(tsr.vars()), array_(tsr.array()), factor_(factor)
      { }

      /// Construct a scaled tensor expression from a negated tensor expression

      /// \param other The expression to be copied
      ScalTsr(const NegTsr<array_type>& other, const scalar_type factor) :
        TsrBase_(other.vars()), array_(other.array()), factor_(-factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalTsr(const ScalTsr_& other, const scalar_type factor) :
        TsrBase_(other), array_(other.array_), factor_(other.factor_ * factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalTsr(const ScalTsr_& other) :
        TsrBase_(other), array_(other.array_), factor_(other.factor_)
      { }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Non-permuting shape factory function

      /// \return The result shape
      typename dist_eval_type::shape_type
      make_shape() { return array_.get_shape().scale(factor_); }

      /// Permuting shape factory function

      /// \param perm The permutation to be applied to the array
      /// \return The result shape
      typename dist_eval::shape_type
      make_shape(const Permutation& perm) { return array_.get_shape().scale(factor_, perm); }

      /// Non-permuting tile operation factory function

      /// \return The tile operation
      op_type make_tile_op() const { return op_type(factor_); }

      /// Permuting tile operation factory function

      /// \param perm The permutation to be applied to tiles
      /// \return The tile operation
      op_type make_tile_op(const Permutation& perm) const { return op_type(perm, factor_); }

      /// Expression identification tag

      /// \return An expression tag used to identify this expression
      std::string print_tag() const {
        std::stringstream ss;
        ss << "[" << factor_ << "]";
        return ss.str();
      }

    }; // class ScalTsr

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_TSR_H__INCLUDED
