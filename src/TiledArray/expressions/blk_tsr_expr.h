/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  blk_tsr_expr.h
 *  May 20, 2015
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_BLK_TSR_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BLK_TSR_EXPR_H__INCLUDED

#include <TiledArray/expressions/unary_expr.h>
#include "blk_tsr_engine.h"

namespace TiledArray {
  namespace expressions {

    // Forward declaration
    template <typename> class TsrExpr;

    template <typename A>
    struct ExprTrait<BlkTsrExpr<A> > {
      typedef A array_type; ///< The \c Array type
      typedef BlkTsrEngine<A> engine_type; ///< Expression engine type
      typedef typename TiledArray::detail::scalar_type<A>::type scalar_type;
                                                          ///< Tile scalar type
    };


    template <typename A>
    struct ExprTrait<ScalBlkTsrExpr<A> > {
      typedef A array_type; ///< The \c Array type
      typedef ScalBlkTsrEngine<A> engine_type; ///< Expression engine type
      typedef typename TiledArray::detail::scalar_type<A>::type scalar_type;
                                                          ///< Tile scalar type
    };

    /// Block expression

    /// \tparam A The array type
    template <typename A>
    class BlkTsrExpr : public Expr<BlkTsrExpr<A> > {
    public:
      typedef BlkTsrExpr<A> BlkTsrExpr_; ///< This class type
      typedef Expr<BlkTsrExpr_> Expr_; ///< Unary base class type
      typedef typename ExprTrait<BlkTsrExpr_>::array_type array_type;
                                                            ///< The array type
      typedef typename ExprTrait<BlkTsrExpr_>::engine_type engine_type;
                                                    ///< Expression engine type

    private:

      const array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable list
      std::vector<std::size_t> lower_bound_; ///< Lower bound of the tile block
      std::vector<std::size_t> upper_bound_; ///< Upper bound of the tile block

    public:

      // Compiler generated functions
      BlkTsrExpr() = delete;
      BlkTsrExpr(const BlkTsrExpr_&) = default;
      BlkTsrExpr(BlkTsrExpr_&&) = default;
      ~BlkTsrExpr() = default;
      BlkTsrExpr_& operator=(const BlkTsrExpr_&) = delete;
      BlkTsrExpr_& operator=(BlkTsrExpr_&&) = delete;

      /// Block expression constructor

      /// \param tsr The argument expression
      /// \param lower_bound The lower bound of the tile block
      /// \param upper_bound The upper bound of the tile block
      template <typename Index>
      BlkTsrExpr(const TsrExpr<A>& tsr, const Index& lower_bound,
          const Index& upper_bound) :
        Expr_(), array_(tsr.array()), vars_(tsr.vars()),
        lower_bound_(std::begin(lower_bound), std::end(lower_bound)),
        upper_bound_(std::begin(upper_bound), std::end(upper_bound))
      {
#ifndef NDEBUG
        const unsigned int rank = array_.trange().tiles().rank();
        // Check the dimension of the lower block bound
        if(TiledArray::detail::size(lower_bound) != rank) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The size lower bound of the block is not equal to rank of " \
                "the array: " \
                << "\n    array rank  = " << array_.trange().tiles().rank() \
                << "\n    lower bound = " << lower_bound );

            TA_EXCEPTION("The size lower bound of the block is not equal to " \
                "rank of the array.");
          }
        }

        // Check the dimension of the upper block bound
        if(TiledArray::detail::size(upper_bound) != rank) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The size upper bound of the block is not equal to rank of " \
                "the array: " \
                << "\n    array rank  = " << rank \
                << "\n    upper bound = " << upper_bound );

            TA_EXCEPTION("The size upper bound of the block is not equal to " \
                "rank of the array.");
          }
        }

        const bool lower_bound_check =
            std::equal(std::begin(lower_bound), std::end(lower_bound),
                    tsr.array().trange().tiles().lobound_data(),
                    [] (std::size_t l, std::size_t r) { return l >= r; });
        const bool upper_bound_check =
            std::equal(std::begin(upper_bound), std::end(upper_bound),
                    tsr.array().trange().tiles().upbound_data(),
                    [] (std::size_t l, std::size_t r) { return l <= r; });
        if(! (lower_bound_check && upper_bound_check)) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The block range is not a sub-block of the array range: " \
                << "\n    array range = " << array_.trange().tiles() \
                << "\n    block range = [ " << lower_bound  << " , " << upper_bound << " )");
          }

          TA_EXCEPTION("The block range is not a sub-block of the array range.");
        }

        const bool lower_upper_bound_check =
            std::equal(std::begin(lower_bound), std::end(lower_bound),
                    std::begin(upper_bound),
                    [] (std::size_t l, std::size_t r) { return l < r; });
        if(! lower_upper_bound_check) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The block lower bound is not less than the upper bound: " \
                << "\n    lower bound = " << lower_bound \
                << "\n    upper bound = " << upper_bound);
          }

          TA_EXCEPTION("The block lower bound is not less than the upper bound.");
        }

#endif // NDEBUG
      }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Tensor variable string accessor

      /// \return A const reference to the variable string for this tensor
      const std::string& vars() const { return vars_; }

      /// Lower bound accessor

      /// \return The block lower bound
      const std::vector<std::size_t>& lower_bound() const { return lower_bound_; }

      /// Upper bound accessor

      /// \return The block upper bound
      const std::vector<std::size_t>& upper_bound() const { return upper_bound_; }

    }; // class BlkTsrExpr


    /// Block expression

    /// \tparam A The array type
    template <typename A>
    class ScalBlkTsrExpr : public Expr<ScalBlkTsrExpr<A> > {
    public:
      typedef ScalBlkTsrExpr<A> ScalBlkTsrExpr_; ///< This class type
      typedef Expr<ScalBlkTsrExpr_> Expr_; ///< Unary base class type
      typedef typename ExprTrait<ScalBlkTsrExpr_>::array_type array_type;
                                                            ///< The array type
      typedef typename ExprTrait<ScalBlkTsrExpr_>::engine_type engine_type;
                                                    ///< Expression engine type
      typedef typename ExprTrait<ScalBlkTsrExpr_>::scalar_type scalar_type;
                                                                ///< Scalar type

    private:

      const array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable list
      scalar_type factor_; ///< The scaling factor
      std::vector<std::size_t> lower_bound_; ///< Lower bound of the tile block
      std::vector<std::size_t> upper_bound_; ///< Upper bound of the tile block

    public:

      // Compiler generated functions
      ScalBlkTsrExpr() = delete;
      ScalBlkTsrExpr(const ScalBlkTsrExpr_&) = default;
      ScalBlkTsrExpr(ScalBlkTsrExpr_&&) = default;
      ~ScalBlkTsrExpr() = default;
      ScalBlkTsrExpr_& operator=(const ScalBlkTsrExpr_&) = delete;
      ScalBlkTsrExpr_& operator=(ScalBlkTsrExpr_&&) = delete;


      /// Expression constructor

      /// \param tsr The block tensor expression
      /// \param factor The scaling factor
      ScalBlkTsrExpr(const BlkTsrExpr<A>& tsr, const scalar_type factor) :
        Expr_(), array_(tsr.array()), vars_(tsr.vars()), factor_(factor),
        lower_bound_(tsr.lower_bound()), upper_bound_(tsr.upper_bound())
      { }

      /// Expression constructor

      /// \param tsr The scaled block tensor expression
      /// \param factor The scaling factor
      ScalBlkTsrExpr(const ScalBlkTsrExpr_& tsr, const scalar_type factor) :
        Expr_(), array_(tsr.array()), factor_(factor * tsr.factor_),
        lower_bound_(tsr.upper_bound()), upper_bound_(tsr.upper_bound())
      { }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Tensor variable string accessor

      /// \return A const reference to the variable string for this tensor
      const std::string& vars() const { return vars_; }

      /// Lower bound accessor

      /// \return The block lower bound
      const std::vector<std::size_t>& lower_bound() const
      { return lower_bound_; }

      /// Upper bound accessor

      /// \return The block upper bound
      const std::vector<std::size_t>& upper_bound() const
      { return upper_bound_; }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalBlkTsrExpr



    /// Scaled-block expression factor

    /// \tparam A The array type
    /// \tparam Scalar A scalar type
    /// \param expr The block expression object
    /// \param factor The scaling factor
    /// \return A scaled-block expression object
    template <typename A, typename Scalar>
    inline typename std::enable_if<TiledArray::detail::is_numeric<Scalar>::value,
        ScalBlkTsrExpr<A> >::type
    operator*(const BlkTsrExpr<A>& expr, const Scalar& factor) {
      return ScalBlkTsrExpr<A>(expr, factor);
    }

    /// Scaled-block expression factor

    /// \tparam A The array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename A, typename Scalar>
    inline typename std::enable_if<TiledArray::detail::is_numeric<Scalar>::value,
        ScalBlkTsrExpr<A> >::type
    operator*(const Scalar& factor, const BlkTsrExpr<A>& expr) {
      return ScalBlkTsrExpr<A>(expr, factor);
    }

    /// Scaled-block expression factor

    /// \tparam A The array type
    /// \tparam Scalar A scalar type
    /// \param expr The block expression object
    /// \param factor The scaling factor
    /// \return A scaled-block expression object
    template <typename A, typename Scalar>
    inline typename std::enable_if<TiledArray::detail::is_numeric<Scalar>::value,
        ScalBlkTsrExpr<A> >::type
    operator*(const ScalBlkTsrExpr<A>& expr, const Scalar& factor) {
      return ScalBlkTsrExpr<A>(expr, factor);
    }

    /// Scaled-block expression factor

    /// \tparam A The array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename A, typename Scalar>
    inline typename std::enable_if<TiledArray::detail::is_numeric<Scalar>::value,
        ScalBlkTsrExpr<A> >::type
    operator*(const Scalar& factor, const ScalBlkTsrExpr<A>& expr) {
      return ScalBlkTsrExpr<A>(expr, factor);
    }

    /// Negated block expression factor

    /// \tparam A The array type
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename A>
    inline ScalBlkTsrExpr<A> operator-(const BlkTsrExpr<A>& expr) {
      return ScalBlkTsrExpr<A>(expr, -1);
    }

    /// Negated scaled-block expression factor

    /// \tparam A The array type
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename A, typename Scalar>
    inline ScalBlkTsrExpr<A> operator-(const ScalBlkTsrExpr<A>& expr) {
      return ScalBlkTsrExpr<A>(expr, -1);
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BLK_TSR_EXPR_H__INCLUDED
