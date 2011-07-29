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


  }  // namespace expressions

}  // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
