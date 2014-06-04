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

#ifndef TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED
#define TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED

#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace math {

    /// Square function object

    /// \tparam T argument and result type
    template <typename T>
    struct Square {
      typedef typename TiledArray::detail::param<T>::type argument_type; ///< The argument type
      typedef T result_type; ///< The result type

      /// Square \c t

      /// \param t The value to be squared
      /// \return t * t
      result_type operator()(argument_type t) const { return t * t; }

    }; // class Square

    /// Square and add function object

    /// \tparam T argument and result type
    template <typename First, typename Second>
    struct SquareAddAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Square \c t

      /// \param t The value to be squared
      /// \return first +=  second * second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first +=  second * second;
      }

    }; // class SquareAddAssign


    /// Square and add function object

    /// \tparam T argument and result type
    template <typename First, typename Second, typename Result>
    struct MultAddAssign {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The argument type
      typedef void result_type; ///< The result type

      /// Square \c t

      /// \param t The value to be squared
      /// \return first +=  second * second
      result_type operator()(Result& result, first_argument_type first, second_argument_type second) const {
        result += first * second;
      }

    }; // class first

    /// Generalization of \c std::plus, but \c First plus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct Plus {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type

      /// Compute the sum of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first + \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return first + second;
      }
    }; // class Plus

    /// Plus assign, but \c First plus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second>
    struct PlusAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute the sum of \c first and \c second

      /// \param[in,out] first The left-hand argument, and the sum of \c first
      /// and \c second
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first += second;
      }
    }; // class PlusAssign

    /// Generalization of \c std::plus with scaling, but \c First plus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct ScalPlus {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalPlus(scalar_type factor) : factor_(factor) { }

      /// Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled sum of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c (first+second)*factor
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return result_type(first + second) * factor_;
      }
    }; // class ScalPlus

    /// Generalization of \c std::plus with scaling and assign, but \c First
    /// plus \c Second yielding \c First

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second>
    struct ScalPlusAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type
      typedef typename detail::scalar_type<First>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalPlusAssign(scalar_type factor) : factor_(factor) { }

      /// Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled sum of \c first and \c second and store the result in \c first

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        (first += second) *= factor_;
      }
    }; // class ScalPlus

    /// Generalization of \c std::minus, but \c First minus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second, typename Result>
    struct Minus {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type

      /// Compute the difference of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first-second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return first - second;
      }
    }; // class Minus

    /// Minus assign, but \c First minus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second>
    struct MinusAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute the difference of \c first and \c second

      /// \param[in,out] first The left-hand argument and result
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first -= second;
      }
    }; // class MinusAssign

    /// Generalization of \c std::minus with scaling, but \c First minus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct ScalMinus {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      ScalMinus(scalar_type factor) : factor_(factor) { }

      /// Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled difference of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c (first-second)*factor
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return result_type(first - second) * factor_;
      }
    }; // class ScalMinus

    /// Generalization of \c std::minus with scaling, but \c First minus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second>
    struct ScalMinusAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type
      typedef typename detail::scalar_type<First>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      ScalMinusAssign(scalar_type factor) : factor_(factor) { }

      /// Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled difference of \c first and \c second

      /// Asssigns <tt> first = (first - second) * factor </tt>
      /// \param first The left-hand argument
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        (first -= second) *= factor_;
      }
    }; // class ScalMinusAssign

    /// Generalization of \c std::multiplies, but \c First times \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct Multiplies {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type

      /// Compute the product of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first*second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return first * second;
      }
    }; // class Multiples

    /// Multiply assign, but \c First times \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second>
    struct MultipliesAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute the product of \c first and \c second

      /// \param[in,out] first The left-hand argument and result
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first *= second;
      }
    }; // class MultipliesAssign

    /// Generalization of \c std::multiples with scaling, but \c First times \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct ScalMultiplies {
      typedef typename TiledArray::detail::param<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalMultiplies(scalar_type factor) : factor_(factor) { }

      /// Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled product of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first * \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return result_type(first * second) * factor_;
      }
    }; // class ScalMultiplies

    /// Generalization of \c std::multiples with scaling, but \c First times \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    template <typename First, typename Second>
    struct ScalMultipliesAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type
      typedef typename detail::scalar_type<First>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalMultipliesAssign(scalar_type factor) : factor_(factor) { }

      /// Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled product of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first *= second * factor_;
      }
    }; // class ScalMultiplies

    /// Generalization of \c std::negate, but \c Arg yielding \c Result

    /// \tparam Arg Argument type
    /// \tparam Result Result type
    template <typename Arg, typename Result>
    struct Negate {
      typedef typename TiledArray::detail::param<Arg>::type argument_type; ///< The argument type
      typedef Result result_type; ///< The result type

      /// Compute the product of \c first and \c second

      /// \param arg The  argument
      /// \return \c -arg
      result_type operator()(argument_type arg) const {
        return -arg;
      }
    }; // class Negate

    /// Negate and assign a value, where \c Arg yielding \c Arg

    /// \tparam Arg Argument type
    template <typename Arg>
    struct NegateAssign {
      typedef Arg& argument_type; ///< The argument type
      typedef Arg result_type; ///< The result type

      /// Compute and assign the negative of \c arg

      /// \param arg The argument
      void operator()(argument_type arg) const {
        arg = -arg;
      }
    }; // class NegateAssign

    /// Generalization of \c std::negate with scaling, but \c Arg yielding \c Result

    /// \tparam Arg Argument type
    /// \tparam Result Result type
    template <typename Arg, typename Result>
    struct ScalNegate {
      typedef typename TiledArray::detail::param<Arg>::type argument_type; ///< The argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors
      ScalNegate(scalar_type factor) : factor_(-factor) { }

      // Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Compute the scaled product of \c first and \c second

      /// \param arg The argument
      /// \return \c -arg*factor
      result_type operator()(argument_type arg) const {
        // Note: factor is negated in the constructor
        return arg * factor_;
      }
    }; // class ScalNegate

    /// Scaling operations

    /// \tparam Arg The argument type
    template <typename Arg>
    struct Scale {
      typedef typename TiledArray::detail::param<Arg>::type argument_type; ///< The argument type
      typedef Arg result_type;
      typedef typename TiledArray::detail::scalar_type<Arg>::type scalar_type;

    private:
      scalar_type factor_; ///< The scaling factor

    public:
      // Constructors
      Scale(const scalar_type factor) : factor_(factor) { }

      // Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Scale operation

      /// \param arg The argument to be scaled
      /// \return \c arg*factor
      result_type operator()(argument_type arg) const {
        return arg * factor_;
      }

    }; // struct Scale

    /// Scaling assign operations

    /// \tparam Arg The argument type
    template <typename Arg>
    struct ScaleAssign {
      typedef Arg& argument_type;
      typedef void result_type;
      typedef typename TiledArray::detail::scalar_type<Arg>::type scalar_type;

    private:
      scalar_type factor_; ///< The scaling factor

    public:
      // Constructors
      ScaleAssign(const scalar_type factor) : factor_(factor) { }

      // Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Scale operation

      /// Multiply-assign \c arg by factor.
      /// \param arg The argument to be scaled
      result_type operator()(argument_type arg) const {
        arg *= factor_;
      }

    }; // struct ScaleAssign

    /// Add constant operations

    /// \tparam Arg The argument type
    template <typename Arg>
    struct PlusAssignConst {
      typedef Arg& argument_type;
      typedef void result_type;
      typedef typename TiledArray::detail::scalar_type<Arg>::type scalar_type;

    private:
      scalar_type factor_; ///< The scaling factor

    public:
      // Constructors
      PlusAssignConst(const scalar_type factor) : factor_(factor) { }

      // Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Scale operation

      /// Add factor to \c arg
      /// \param arg The argument to be scaled
      result_type operator()(argument_type arg) const {
        arg += factor_;
      }

    }; // struct PlusAssignConst

    /// Add constant operations

    /// \tparam Arg The argument type
    template <typename Arg>
    struct PlusConst {
      typedef typename TiledArray::detail::param<Arg>::type argument_type; ///< The argument type
      typedef Arg result_type;
      typedef typename TiledArray::detail::scalar_type<Arg>::type scalar_type;

    private:
      scalar_type factor_; ///< The scaling factor

    public:
      // Constructors
      PlusConst(const scalar_type factor) : factor_(factor) { }

      // Scaling factor accessor
      scalar_type factor() const { return factor_; }

      /// Scale operation

      /// Add factor to \c arg
      /// \param arg The argument to be scaled
      result_type operator()(argument_type arg) const {
        return arg + factor_;
      }

    }; // struct PlusAssignConst



    /// Minimum assign

    /// \tparam Arg Argument type
    template <typename First, typename Second>
    struct MinAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute set first to the minimum of first and second

      /// \param[in,out] first The left-hand argument and result
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first = std::min(first, second);
      }
    }; // class MinAssign

    /// Maximum assign

    /// \tparam Arg Argument type
    template <typename First, typename Second>
    struct MaxAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute set first to the maximum of first and second

      /// \param[in,out] first The left-hand argument and result
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first = std::max(first, second);
      }
    }; // class MaxAssign

    /// Absolute minimum assign

    /// \tparam Arg Argument type
    template <typename First, typename Second>
    struct AbsMinAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute set first to the absolute minimum of first and second

      /// \param[in,out] first The left-hand argument and result
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first = std::min(first, std::abs(second));
      }
    }; // class AbsMinAssign

    /// Absolute maximum assign

    /// \tparam Arg Argument type
    template <typename First, typename Second>
    struct AbsMaxAssign {
      typedef First& first_argument_type; ///< The left-hand argument type
      typedef typename TiledArray::detail::param<Second>::type second_argument_type; ///< The right-hand argument type
      typedef void result_type; ///< The result type

      /// Compute set first to the absolute maximum of first and second

      /// \param[in,out] first The left-hand argument and result
      /// \param second The right-hand argument
      result_type operator()(first_argument_type first, second_argument_type second) const {
        first = std::max(first, std::abs(second));
      }
    }; // class AbsMaxAssign

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED
