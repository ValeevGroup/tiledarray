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
  namespace detail {

    /// Square function object

    /// \tparam T argument and result type
    template <typename T>
    struct Square {
      typedef T result_type;
      typedef T argument_type;

      /// Square \c t

      /// \param t The value to be squared
      /// \return t * t
      result_type operator()(argument_type t) const { return t * t; }

    }; // class Square

    /// Generalization of \c std::plus, but \c First plus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct Plus {
      typedef First first_argument_type; ///< The left-hand argument type
      typedef Second second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type

      /// Compute the sum of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first + \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return first + second;
      }
    }; // class Plus

    /// Generalization of \c std::plus with scaling, but \c First plus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct ScalPlus {
      typedef First first_argument_type; ///< The left-hand argument type
      typedef Second second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalPlus(scalar_type factor) : factor_(factor) { }
      ScalPlus(const ScalPlus<First,Second,Result>& other) : factor_(other.factor_) { }

      // Assignment operator
      ScalPlus<First,Second,Result>&
      operator=(const ScalPlus<First,Second,Result>& other) {
        factor_ = other.factor_;
        return *this;
      }

      /// Compute the scaled sum of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c (first+second)*factor
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return result_type(first + second) * factor_;
      }
    }; // class ScalPlus

    /// Generalization of \c std::minus, but \c First minus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct Minus {
      typedef typename std::add_const<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename std::add_const<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type

      /// Compute the difference of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first - \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return first - second;
      }
    }; // class Minus

    /// Generalization of \c std::minus with scaling, but \c First minus \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct ScalMinus {
      typedef typename std::add_const<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename std::add_const<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalMinus(scalar_type factor) : factor_(factor) { }
      ScalMinus(const ScalMinus<First,Second,Result>& other) : factor_(other.factor_) { }

      // Assignment operator
      ScalMinus<First,Second,Result>& operator=(const ScalMinus<First,Second,Result>& other) {
        factor_ = other.factor_;
        return *this;
      }

      /// Compute the scaled difference of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c (first-second)*factor
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return result_type(first - second) * factor_;
      }
    }; // class ScalMinus

    /// Generalization of \c std::multiplies, but \c First times \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct Multiplies {
      typedef typename std::add_const<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename std::add_const<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type

      /// Compute the product of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first*second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return first * second;
      }
    }; // class Multiples

    /// Generalization of \c std::multiples with scaling, but \c First times \c Second yielding \c Result

    /// \tparam First Left-hand argument type
    /// \tparam Second Right-hand argument type
    /// \tparam Result Result type
    template <typename First, typename Second, typename Result>
    struct ScalMultiplies {
      typedef typename std::add_const<First>::type first_argument_type; ///< The left-hand argument type
      typedef typename std::add_const<Second>::type second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors & assignment operator
      ScalMultiplies(scalar_type factor) : factor_(factor) { }
      ScalMultiplies(const ScalMultiplies<First,Second,Result>& other) : factor_(other.factor_) { }

      // Assignment operator
      ScalMultiplies<First,Second,Result>&
      operator=(const ScalMultiplies<First,Second,Result>& other) {
        factor_ = other.factor_;
        return *this;
      }

      /// Compute the scaled product of \c first and \c second

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return \c first * \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        return result_type(first * second) * factor_;
      }
    }; // class ScalMultiplies


    /// Generalization of \c std::negate, but \c Arg yielding \c Result

    /// \tparam Arg Argument type
    /// \tparam Result Result type
    template <typename Arg, typename Result>
    struct Negate {
      typedef typename std::add_const<Arg>::type argument_type; ///< The argument type
      typedef Result result_type; ///< The result type

      /// Compute the product of \c first and \c second

      /// \param arg The  argument
      /// \return \c -arg
      result_type operator()(argument_type arg) const {
        return -arg;
      }
    }; // class Negate

    /// Generalization of \c std::negate with scaling, but \c Arg yielding \c Result

    /// \tparam Arg Argument type
    /// \tparam Result Result type
    template <typename Arg, typename Result>
    struct ScalNegate {
      typedef typename std::add_const<Arg>::type argument_type; ///< The left-hand argument type
      typedef Result result_type; ///< The result type
      typedef typename detail::scalar_type<Result>::type scalar_type; ///< Scaling factor type

    private:
      scalar_type factor_; ///< scaling factor

    public:

      // Constructors
      ScalNegate(scalar_type factor) : factor_(-factor) { }
      ScalNegate(const ScalNegate<Arg,Result>& other) : factor_(other.factor_) { }

      // Assignment operator
      ScalNegate<Arg,Result>& operator=(const ScalNegate<Arg,Result>& other) {
        factor_ = other.factor_;
        return *this;
      }

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
    /// \tparam Result The result type
    template <typename Arg, typename Result>
    struct Scale {
      typedef typename std::add_const<Arg>::type argument_type;
      typedef Result result_type;
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type;

    private:
      scalar_type factor_; ///< The scaling factor

    public:
      // Constructors
      Scale(const scalar_type factor) : factor_(factor) { }
      Scale(const Scale<Arg, Result>& other) : factor_(other.factor_) { }

      // Assignment operator
      Scale<Arg, Result>& operator=(const Scale<Arg, Result>& other) {
        factor_ = other.factor_;
        return *this;
      }

      /// Scale operation

      /// \param arg The argument to be scaled
      /// \return \c arg*factor
      result_type operator()(const argument_type arg) const {
        return arg * factor_;
      }

    }; // struct Scale

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_MATH_FUNCTIONAL_H__INCLUDED
