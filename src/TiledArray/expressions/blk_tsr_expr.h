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
    template <typename> class BlkTsrExpr;
    template <typename, typename> class ScalBlkTsrExpr;

    template <typename Array>
    struct ExprTrait<BlkTsrExpr<Array> > {
      typedef Array array_type; ///< The \c Array type
      typedef Array& reference; ///< \c Array reference type
      typedef BlkTsrEngine<Array> engine_type; ///< Expression engine type
      typedef typename TiledArray::detail::scalar_type<Array>::type scalar_type;
                                                          ///< Tile scalar type
    };

    template <typename Array>
    struct ExprTrait<BlkTsrExpr<const Array> > {
      typedef Array array_type; ///< The \c Array type
      typedef const Array& reference; ///< \c Array reference type
      typedef BlkTsrEngine<Array> engine_type; ///< Expression engine type
      typedef typename TiledArray::detail::scalar_type<Array>::type scalar_type;
                                                          ///< Tile scalar type
    };

    template <typename Array, typename Scalar>
    struct ExprTrait<ScalBlkTsrExpr<Array, Scalar> > {
      typedef Array array_type; ///< The \c Array type
      typedef const Array& reference; ///< \c Array reference type
      typedef ScalBlkTsrEngine<Array, Scalar> engine_type; ///< Expression engine type
      typedef Scalar scalar_type; ///< Tile scalar type
    };

    template <typename Array, typename Scalar>
    struct ExprTrait<ScalBlkTsrExpr<const Array, Scalar> > {
      typedef Array array_type; ///< The \c Array type
      typedef const Array& reference; ///< \c Array reference type
      typedef ScalBlkTsrEngine<Array, Scalar> engine_type; ///< Expression engine type
      typedef Scalar scalar_type; ///< Tile scalar type
    };

    /// Block expression

    /// \tparam Derived The derived class type
    template <typename Derived>
    class BlkTsrExprBase : public Expr<Derived> {
    public:
      typedef BlkTsrExprBase<Derived> BlkTsrExprBase_; ///< This class type
      typedef Expr<Derived> Expr_; ///< Unary base class type
      typedef typename ExprTrait<Derived>::array_type array_type;
                                                            ///< The array type
      typedef typename ExprTrait<Derived>::reference reference;
                                                  ///< The array reference type

    protected:

      reference array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable list
      std::vector<std::size_t> lower_bound_; ///< Lower bound of the tile block
      std::vector<std::size_t> upper_bound_; ///< Upper bound of the tile block

      void check_valid() const {
        const unsigned int rank = array_.trange().tiles().rank();
        // Check the dimension of the lower block bound
        if(TiledArray::detail::size(lower_bound_) != rank) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The size lower bound of the block is not equal to rank of " \
                "the array: " \
                << "\n    array rank  = " << array_.trange().tiles().rank() \
                << "\n    lower bound = " << lower_bound_ );

            TA_EXCEPTION("The size lower bound of the block is not equal to " \
                "rank of the array.");
          }
        }

        // Check the dimension of the upper block bound
        if(TiledArray::detail::size(upper_bound_) != rank) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The size upper bound of the block is not equal to rank of " \
                "the array: " \
                << "\n    array rank  = " << rank \
                << "\n    upper bound = " << upper_bound_ );

            TA_EXCEPTION("The size upper bound of the block is not equal to " \
                "rank of the array.");
          }
        }

        const bool lower_bound_check =
            std::equal(std::begin(lower_bound_), std::end(lower_bound_),
                    array_.trange().tiles().lobound_data(),
                    [] (std::size_t l, std::size_t r) { return l >= r; });
        const bool upper_bound_check =
            std::equal(std::begin(upper_bound_), std::end(upper_bound_),
                    array_.trange().tiles().upbound_data(),
                    [] (std::size_t l, std::size_t r) { return l <= r; });
        if(! (lower_bound_check && upper_bound_check)) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The block range is not a sub-block of the array range: " \
                << "\n    array range = " << array_.trange().tiles() \
                << "\n    block range = [ " << lower_bound_  << " , " << upper_bound_ << " )");
          }

          TA_EXCEPTION("The block range is not a sub-block of the array range.");
        }

        const bool lower_upper_bound_check =
            std::equal(std::begin(lower_bound_), std::end(lower_bound_),
                    std::begin(upper_bound_),
                    [] (std::size_t l, std::size_t r) { return l < r; });
        if(! lower_upper_bound_check) {
          if(World::get_default().rank() == 0) {
            TA_USER_ERROR_MESSAGE( \
                "The block lower bound is not less than the upper bound: " \
                << "\n    lower bound = " << lower_bound_ \
                << "\n    upper bound = " << upper_bound_);
          }

          TA_EXCEPTION("The block lower bound is not less than the upper bound.");
        }
      }

    public:

      // Compiler generated functions
      BlkTsrExprBase() = delete;
      BlkTsrExprBase(const BlkTsrExprBase_&) = default;
      BlkTsrExprBase(BlkTsrExprBase_&&) = default;
      ~BlkTsrExprBase() = default;
      BlkTsrExprBase_& operator=(const BlkTsrExprBase_&) = delete;
      BlkTsrExprBase_& operator=(BlkTsrExprBase_&&) = delete;

      /// Block expression constructor

      /// \param tsr The argument expression
      /// \param lower_bound The lower bound of the tile block
      /// \param upper_bound The upper bound of the tile block
      template <typename A, typename Index>
      BlkTsrExprBase(const TsrExpr<A>& tsr, const Index& lower_bound,
          const Index& upper_bound) :
        Expr_(), array_(tsr.array()), vars_(tsr.vars()),
        lower_bound_(std::begin(lower_bound), std::end(lower_bound)),
        upper_bound_(std::begin(upper_bound), std::end(upper_bound))
      { }

      template <typename D>
      BlkTsrExprBase(const BlkTsrExprBase<D>& tsr) :
        Expr_(), array_(tsr.array()), vars_(tsr.vars()),
        lower_bound_(tsr.lower_bound()), upper_bound_(tsr.upper_bound())
      { }

      /// Array accessor

      /// \return a const reference to this array
      reference array() const { return array_; }

      /// Tensor variable string accessor

      /// \return A const reference to the variable string for this tensor
      const std::string& vars() const { return vars_; }

      /// Lower bound accessor

      /// \return The block lower bound
      const std::vector<std::size_t>& lower_bound() const { return lower_bound_; }

      /// Upper bound accessor

      /// \return The block upper bound
      const std::vector<std::size_t>& upper_bound() const { return upper_bound_; }

    }; // class BlkTsrExprBase

    /// Block expression

    /// \tparam A The array type
    template <typename Array>
    class BlkTsrExpr : public BlkTsrExprBase<BlkTsrExpr<Array> > {
    public:
      typedef BlkTsrExpr<Array> BlkTsrExpr_; ///< This class type
      typedef BlkTsrExprBase<BlkTsrExpr<Array> > BlkTsrExprBase_;
                                                 ///< Block expresion base type
      typedef typename ExprTrait<BlkTsrExpr_>::engine_type engine_type;
                                                    ///< Expression engine type

      // Compiler generated functions
      BlkTsrExpr() = delete;
      BlkTsrExpr(const BlkTsrExpr_&) = default;
      BlkTsrExpr(BlkTsrExpr_&&) = default;
      ~BlkTsrExpr() = default;

      /// Block expression constructor

      /// \param tsr The argument expression
      /// \param lower_bound The lower bound of the tile block
      /// \param upper_bound The upper bound of the tile block
      template <typename Index>
      BlkTsrExpr(const TsrExpr<Array>& tsr, const Index& lower_bound,
          const Index& upper_bound) :
        BlkTsrExprBase_(tsr, lower_bound, upper_bound)
      {
#ifndef NDEBUG
        BlkTsrExprBase_::check_valid();
#endif // NDEBUG
      }

      /// Expression assignment operator

      /// \param other The expression that will be assigned to this array
      BlkTsrExpr_& operator=(const BlkTsrExpr_& other) {
        other.eval_to(*this);
        return *this;
      }

      /// Expression assignment operator

      /// \param other The expression that will be assigned to this array
      BlkTsrExpr_& operator=(BlkTsrExpr_&& other) {
        other.eval_to(*this);
        return *this;
      }

      /// Expression assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be assigned to this array
      template <typename D>
      BlkTsrExpr_& operator=(const Expr<D>& other) {
        other.derived().eval_to(*this);
        return *this;
      }

      /// Expression plus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be added to this array
      template <typename D>
      BlkTsrExpr_& operator+=(const Expr<D>& other) {
        return operator=(AddExpr<BlkTsrExpr_, D>(*this, other.derived()));
      }

      /// Expression minus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be subtracted from this array
      template <typename D>
      BlkTsrExpr_& operator-=(const Expr<D>& other) {
        return operator=(SubtExpr<BlkTsrExpr_, D>(*this, other.derived()));
      }

      /// Expression multiply-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will scale this array
      template <typename D>
      BlkTsrExpr_& operator*=(const Expr<D>& other) {
        return operator=(MultExpr<BlkTsrExpr_, D>(*this, other.derived()));
      }

    }; // class BlkTsrExpr

    /// Block expression

    /// \tparam Array The array type
    template <typename Array>
    class BlkTsrExpr<const Array> : public BlkTsrExprBase<BlkTsrExpr<const Array> > {
    public:
      typedef BlkTsrExpr<const Array> BlkTsrExpr_; ///< This class type
      typedef BlkTsrExprBase<BlkTsrExpr<const Array> > BlkTsrExprBase_;
                                                 ///< Block expresion base type
      typedef typename ExprTrait<BlkTsrExpr_>::engine_type engine_type;
                                                    ///< Expression engine type

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
      BlkTsrExpr(const TsrExpr<const Array>& tsr, const Index& lower_bound,
          const Index& upper_bound) :
        BlkTsrExprBase_(tsr, lower_bound, upper_bound)
      {
#ifndef NDEBUG
        BlkTsrExprBase_::check_valid();
#endif // NDEBUG
      }

      /// Block expression constructor

      /// \param tsr The argument expression
      /// \param lower_bound The lower bound of the tile block
      /// \param upper_bound The upper bound of the tile block
      template <typename Index>
      BlkTsrExpr(const TsrExpr<Array>& tsr, const Index& lower_bound,
          const Index& upper_bound) :
        BlkTsrExprBase_(tsr, lower_bound, upper_bound)
      {
#ifndef NDEBUG
        BlkTsrExprBase_::check_valid();
#endif // NDEBUG
      }
    }; // class BlkTsrExpr<const Array>

    /// Block expression

    /// \tparam Array The array type
    template <typename Array, typename Scalar>
    class ScalBlkTsrExpr : public BlkTsrExprBase<ScalBlkTsrExpr<Array, Scalar> > {
    public:
      typedef ScalBlkTsrExpr<Array, Scalar> ScalBlkTsrExpr_; ///< This class type
      typedef BlkTsrExprBase<ScalBlkTsrExpr_> BlkTsrExprBase_;
                                                 ///< Block expresion base type
      typedef typename ExprTrait<ScalBlkTsrExpr_>::engine_type engine_type;
                                                    ///< Expression engine type
      typedef typename ExprTrait<ScalBlkTsrExpr_>::scalar_type scalar_type;
                                                               ///< Scalar type

    private:

      scalar_type factor_; ///< The scaling factor

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
      ScalBlkTsrExpr(const BlkTsrExpr<Array>& tsr, const scalar_type factor) :
        BlkTsrExprBase_(tsr), factor_(factor)
      { }

      /// Expression constructor

      /// \param tsr The scaled block tensor expression
      /// \param factor The scaling factor
      ScalBlkTsrExpr(const ScalBlkTsrExpr_& tsr, const scalar_type factor) :
        BlkTsrExprBase_(tsr), factor_(factor * tsr.factor_)
      { }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalBlkTsrExpr


    using TiledArray::detail::mult_t;

    /// Scaled-block expression factor

    /// \tparam Array The array type
    /// \tparam Scalar Array scalar type
    /// \param expr The block expression object
    /// \param factor The scaling factor
    /// \return Array scaled-block expression object
    template <typename Array, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalBlkTsrExpr<Array, Scalar>
    operator*(const BlkTsrExpr<Array>& expr, const Scalar& factor) {
      return ScalBlkTsrExpr<Array, Scalar>(expr, factor);
    }

    /// Scaled-block expression factor

    /// \tparam Array The array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename Array, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalBlkTsrExpr<Array, Scalar>
    operator*(const Scalar& factor, const BlkTsrExpr<Array>& expr) {
      return ScalBlkTsrExpr<Array, Scalar>(expr, factor);
    }

    /// Scaled-block expression factor

    /// \tparam Array The array type
    /// \tparam Scalar1 A scalar factor type
    /// \tparam Scalar2 A scalar factor type
    /// \param expr The block expression object
    /// \param factor The scaling factor
    /// \return A scaled-block expression object
    template <typename Array, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar2>::value
        >::type* = nullptr>
    inline ScalBlkTsrExpr<Array, mult_t<Scalar1, Scalar2> >
    operator*(const ScalBlkTsrExpr<Array, Scalar1>& expr, const Scalar2& factor) {
      return ScalBlkTsrExpr<Array, mult_t<Scalar1, Scalar2> >(expr, factor);
    }

    /// Scaled-block expression factor

    /// \tparam Array The array type
    /// \tparam Scalar1 A scalar factor type
    /// \tparam Scalar2 A scalar factor type
    /// \param factor The scaling factor
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename Array, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar1>::value
        >::type * = nullptr>
    inline ScalBlkTsrExpr<Array, mult_t<Scalar2, Scalar1> >
    operator*(const Scalar1& factor, const ScalBlkTsrExpr<Array, Scalar2>& expr) {
      return ScalBlkTsrExpr<Array, mult_t<Scalar2, Scalar1> >(expr, factor);
    }

    /// Negated block expression factor

    /// \tparam Array The array type
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename Array>
    inline ScalBlkTsrExpr<Array, typename ExprTrait<BlkTsrExpr<Array> >::scalar_type>
    operator-(const BlkTsrExpr<Array>& expr) {
      return ScalBlkTsrExpr<Array,
          typename ExprTrait<BlkTsrExpr<Array> >::scalar_type>(expr, -1);
    }

    /// Negated scaled-block expression factor

    /// \tparam Array The array type
    /// \tparam Scalar A scalar factor type
    /// \param expr The block expression object
    /// \return A scaled-block expression object
    template <typename Array, typename Scalar>
    inline ScalBlkTsrExpr<Array, Scalar>
    operator-(const ScalBlkTsrExpr<Array, Scalar>& expr) {
      return ScalBlkTsrExpr<Array, Scalar>(expr, -1);
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BLK_TSR_EXPR_H__INCLUDED
