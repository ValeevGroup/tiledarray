#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED

#include <TiledArray/array_math.h>
#include <world/array.h>
#include <functional>

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
    template <typename E>
    class Expression {
    public:
      typedef E type;

    protected:
      Expression() { }
      ~Expression() { }

    private:
      Expression& operator=(const Expression<E>&);
    };

    template<class E>
    class ScalarExpression : public Expression<E> {
    public:
      typedef E exp_type;

      inline const exp_type& eval() const {
        return *static_cast<const exp_type*> (this);
      }

      inline exp_type& eval() {
        return *static_cast<exp_type*> (this);
      }
    };

    template <typename T>
    class ScalarReference : public ScalarExpression<ScalarReference<T> > {
    public:
      typedef ScalarReference<T> ScalarReference_;

      typedef T value_type;
      typedef const value_type& const_reference;
      typedef typename madness::if_<std::is_const<T>,
          const_reference,
          value_type &>::type reference;

      // Construction and destruction
      inline explicit ScalarReference(reference ref) : ref_(ref) { }

      // Conversion
      inline operator value_type () const {
        return ref_;
      }

      // Assignment
      inline ScalarReference_& operator=(const ScalarReference_& other) {
        ref_ = other.ref_;
        return *this;
      }

      template <typename E>
      inline ScalarReference_& operator=(const ScalarExpression<E> &other) {
        ref_ = other;
        return *this;
      }

    private:
      reference ref_;
    };

    template <typename T>
    class ScalarValue : public ScalarExpression<ScalarValue<T> > {

      typedef ScalarValue<T> ScalarValue_;
    public:
      typedef T value_type;
      typedef const value_type &const_reference;
      typedef typename madness::if_<std::is_const<T>,
          const_reference,
          value_type &>::type reference;

      // Construction and destruction
      inline ScalarValue() : value_ () {}
      inline ScalarValue(const value_type &value) : value_(value) {}

      inline operator value_type () const { return value_; }

      // Assignment
      inline ScalarValue_& operator=(const ScalarValue_& other) {
        value_ = other.value_;
        return *this;
      }
      template <typename E>
      inline ScalarValue_& operator=(const ScalarExpression<E>& other) {
        value_ = other;
        return *this;
      }

    private:
      value_type value_;
    };
/*
    template <typename E>
    class DimExpression : public Expression<E> {
    public:
      typedef E exp_type;

      static inline unsigned int dim() { return exp_type::dim(); }
    };

    template <typename E>
    class ArrayExpression : public Expression<E> {
    public:
      typedef E exp_type;

      inline const exp_type& operator()() const { return *static_cast<const exp_type*>(this); }
      inline exp_type& operator()() { return *static_cast<exp_type*>(this); }

    };

    template <typename T, std::size_t N>
    class ArrayReference : public ArrayExpression<ArrayReference<T, N> > {
    public:
      typedef ScalarReference<T, N> ScalarReference_;

      typedef std::array<T, N> value_type;
      typedef const value_type& const_reference;
      typedef typename madness::if_<std::is_const<T>,
          const_reference,
          value_type &>::type reference;

      // Construction and destruction
      inline explicit ArrayReference(reference ref) : ref_(ref) { }

      // Conversion
      inline operator value_type () const {
        return ref_;
      }

      // Assignment
      inline ScalarReference_& operator=(const ScalarReference_& other) {
        ref_ = other.ref_;
        return *this;
      }

      template <typename E>
      inline ScalarReference_& operator=(const ScalarExpression<E> &other) {
        ref_ = other;
        return *this;
      }

    private:
      reference ref_;
    };

    template <typename T, std::size_t N>
    class ArrayValue : public ArrayExpression<ArrayValue<T, N> > {
    public:
      typedef ArrayValue<T, N> ArrayValue_;

      typedef std::array<T, N> value_type;
      typedef const value_type &const_reference;
      typedef typename madness::if_<std::is_const<T>,
          const_reference,
          value_type &>::type reference;

      // Construction and destruction
      inline ArrayValue() : value_ () {}
      inline ArrayValue(const value_type &value) : value_(value) {}

      inline operator value_type () const { return value_; }

      // Assignment
      inline ScalarValue_& operator=(const ScalarValue_& other) {
        value_ = other.value_;
        return *this;
      }
      template <typename E>
      inline ScalarValue_& operator=(const ScalarExpression<E>& other) {
        value_ = other;
        return *this;
      }

    private:
      value_type value_;
    };


    template <typename R>
    class RangeExpression : public DimExpression<R> {
    public:
      typedef R exp_type;

      unsigned int dim() const { return exp_type::dim(); }

    };

    template <typename T>
    class TileExpression : public Expression<T> {
    public:
      typedef T exp_type;

      unsigned int dim() const { return exp_type::dim(); }

      inline const exp_type& eval() const { return *static_cast<const exp_type*>(this); }

      inline exp_type& eval() { return *static_cast<exp_type*>(this); }
    };

    template <typename A>
    class ArrayExpression : public Expression<A> {
    public:
      typedef A exp_type;

    };


    template <typename LeftExp, typename RightExp, typename Op>
    class BinaryExpression : public Expression<BinaryExpression<LeftExp, RightExp, Op> > {
    public:
      typedef LeftExp left_argument_type;
      typedef RightExp right_argument_type;

      typedef typename madness::result_of<Op>::type result_type;

      /// Constructs a binary expression

      /// \param left The left-hand argument of the expression
      /// \param right The right-hand argument of the expression
      BinaryExpression(const left_argument_type& left, const right_argument_type& right, Op op = Op()) :
          left_(left), right_(right), op_(op)
      { }

      result_type eval() { return op_(left_.eval(), right_.eval()); }

    public:
      const left_argument_type& left_;    ///< The left-hand argument
      const right_argument_type& right_;  ///< The right-hand argument
      Op op_;
    };

    template <typename Exp, typename Op>
    class UnaryExpression : public Expression<UnaryExpression<Exp, Op> >{
    public:
      typedef Exp argument_type;

      typedef typename madness::result_of<Op>::type result_type;

      UnaryExpression(const argument_type& a, Op op = Op()) : arg_(a), op_(op) { }

      result_type eval() const { return op_(arg_); }

    private:
      const argument_type& arg_;
      Op op_;
    };
*/
  }  // namespace expressions

}  // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
