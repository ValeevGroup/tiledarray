#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED

#include <TiledArray/math.h>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

namespace TiledArray {

  namespace expressions {

    /// A general expression object

    /// All expression objects are derived from this class, and must implement
    /// an eval() function. The eval() function accepts a single, non-const
    /// argument, which is the result type of the expression. For example:
    /// \code
    /// class MyExpression : public Expression<MyExpression> {
    /// public:
    ///   template <typename Result>
    ///   Result& eval(Result& r) const {
    ///     // ...
    ///   }
    /// };
    /// \endcode
    /// \tparam ExpType The derived expression class type
    template <typename ExpType>
    class Expression {
    private:
      Expression();
    public:
      template <typename Result>
      Result& eval(Result& r) const { return ExpType::eval(r); }

    private:
      template <typename Result>
      Result& eval(const Result& r) const;
    };

    template <typename LeftExp, typename RightExp, template <typename> class Op>
    class BinaryExpression : public Expression<BinaryExpression<LeftExp, RightExp, Op> > {
    public:
      typedef LeftExp left_argument_type;
      typedef RightExp right_argument_type;

      /// Constructs a binary expression

      /// \param left The left-hand argument of the expression
      /// \param right The right-hand argument of the expression
      BinaryExpression(const left_argument_type& left, const right_argument_type& right) :
          left_(left), right_(right)
      { }

      /// Evaluate this expression and place the result into \c r

      /// \tparam Result The result object type
      /// \param r The a non-const reference to the result object
      template <typename Result>
      Result& eval(Result& r) const {
        math::BinaryOp<Result, left_argument_type, right_argument_type, Op> op;
        op(r, left_, right_);

        return r;
      }

    private:
      const left_argument_type& left_;    ///< The left-hand argument
      const right_argument_type& right_;  ///< The right-hand argument
    };

    template <typename Exp, template <typename> class Op>
    class UnaryExpression : public Expression<UnaryExpression<Exp, Op> >{
    public:
      typedef Exp argument_type;

      UnaryExpression(const argument_type& a) : arg_(a) { }

      /// Evaluate this expression and place the result into \c r

      /// \tparam Result The result object type
      /// \param r The a non-const reference to the result object
      template <typename Result>
      Result& eval(Result& r) const {
        math::UnaryOp<Result, argument_type, Op> op;
        op(r, arg_);

        return r;
      }

    private:
      const argument_type& arg_;
    };

    /// Constructs an addition expression

    /// \tparam LeftExp The left-hand expression type
    /// \tparam RightExp The right-hand expression type
    /// \param left A const reference to the left-hand expression object.
    /// \param right A const reference to the right-hand expression object.
    template<typename LeftExp, typename RightExp>
    Expression<BinaryExpression<LeftExp, RightExp, std::plus> >
    operator +(const LeftExp& left, const RightExp& right) {
      return Expression<BinaryExpression<LeftExp, RightExp, std::plus> >(left, right);
    }

    /// Constructs a subtraction expression

    /// \tparam LeftExp The left-hand expression type
    /// \tparam RightExp The right-hand expression type
    /// \param left A const reference to the left-hand expression object.
    /// \param right A const reference to the right-hand expression object.
    template<typename LeftExp, typename RightExp>
    Expression<BinaryExpression<LeftExp, RightExp, std::minus> >
    operator -(const LeftExp& left, const RightExp& right) {
      return Expression<BinaryExpression<LeftExp, RightExp, std::minus> >(left, right);
    }

    /// Constructs a contraction expression

    /// \tparam LeftExp The left-hand expression type
    /// \tparam RightExp The right-hand expression type
    /// \param left A const reference to the left-hand expression object.
    /// \param right A const reference to the right-hand expression object.
    template<typename LeftExp, typename RightExp>
    Expression<BinaryExpression<LeftExp, RightExp, std::multiplies> >
    operator *(const LeftExp& left, const RightExp& right) {
      return Expression<BinaryExpression<LeftExp, RightExp, std::minus> >(left, right);
    }

    /// Constructs a negation expression
    template<typename Exp>
    Expression<UnaryExpression<Exp, std::negate> >
    operator -(const Exp& e) {
      return Expression<UnaryExpression<Exp, std::negate> >(e);
    }

  }  // namespace expressions

}  // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
