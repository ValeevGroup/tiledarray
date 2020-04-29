# Customizing TiledArray Expressions {#Customizing-Expressions}
End-user interface of TiledArray is a Domain-Specific Language (DSL), embedded in C++, for performing basic arithmetic and user-defined operations on `DistArray` objects. For example, contraction of order-3 tensors to produce an order-2 tensor can be expressed as
```c++
TArray<float> a, b, c;
// initialization of order-3 tensors a, b is not shown

c("i,j") = a("i,k,l") * b("k,j,l"); // Einstein notation described in https://en.wikipedia.org/wiki/Tensor_contraction
```
Even such a simple expression is implemented as a sequence of elementary operations, such as permutation and tensor contraction; understanding what happens under the hood is important for writing optimized TiledArray code.
Furthermore, TiledArray DSL is extensible: users can write their own expressions to provide custom functions or optimizations. 
Understanding how expressions are evaluated is needed for such customizations.

This document is intended for users who

1. want to understand how TiledArray DSL expressions are evaluated so that they can can control and optimize the evaluation of TiledArray expressions, or
2. want to understand how to extend the DSL to suit their needs.

# Implementation

TiledArray DSL is built using the
[expression template](https://en.wikipedia.org/wiki/Expression_templates) idiom.
Lazy evaluation of DSL expressions allows to (heuristically) optimize their
evaluation by minimizing the number of permutations, minimizing memory accesses,
minimizing temporary storage, and combining arithmetic operations; DSL does NOT
more extensive term rewriting, such as operation reordering, factorization
(strength reduction), or common subexpression elimination. Such task can be
performed by the user (with help of an optimizing compiler), provided sufficient
understanding of how the TiledArray DSL expressions are evaluated. For a more
detailed overview of the DSL implementation see
[here](@ref Expressions-in-TiledArray). For detailed examples of the expression
layer see [here](@ref Expression-Layer-Example) and for detailed examples of the
expression engine see [here](@ref Expression-Engine-Example).

## DSL Overview
DSL expressions are evaluated in a multi-stage process, each step of which can be overridden to suit your needs.

1. __construct an expression object__: expression objects are the nodes of the [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) describing a composite expression 
2. __construct expression engine__: expression engines compute the metadata needed to evaluate the expression, such a the following properties of the result:
  1. variable list
  2. structure
    1. `Permutation`
    2. `TiledRange`
    3. shape
  3. data distribution
    1. world
    2. process map
3. __evaluate the expression__: decompose expression into a set of tasks and submit these tasks to the task queue
  1. tile operation
  2. construct the distributed evaluator
  3. schedule tile evaluation tasks

## Expression Interface
Expression objects contain the minimum amount of information required to define the operation. For example, expression objects in TiledArray store `Array` objects, scaling factors, and argument expressions. Note, you can add or modify member variables, member functions, or constructors to suit the needs of your application.

There are three basic types of expression objects:

1. leaf -- no arguments (e.g. an `Array` + variable list)
2. unary -- one argument (e.g. negation)
3. binary -- two arguments (e.g. addition)

**Note:** When you implement your own expression, you should copy and paste the appropriate the expression interface below and fill in the implementation details.

### Leaf Expression Interface
```c++
#include <tiledarray.h>

// Forward declarations
template <typename> class MyLeafExpression;
template <typename> class MyLeafEngine;

namespace TiledArray {
  namespace expressions {

    // Define expression types
    template <typename A>
    struct ExprTrait<MyLeafExpression<A> > {
      typedef A array_type; // The Array type
      typedef MyLeafEngine<A> engine_type; // Expression engine type
      typedef typename TiledArray::detail::scalar_type<A>::type scalar_type;  // Tile scalar type
    };

  } // namespace expressions
} // namespace TiledArray

/// My leaf expression objects
template <typename A>
class MyLeafExpression :
    public Expr<MyLeafExpression<A> >
{
public:
  typedef MyLeafExpression<A> MyLeafExpression_; // This class type
  typedef Expr<MyLeafExpression_> Expr_; // Expression base type
  typedef typename TiledArray::expressions::ExprTrait<MyLeafExpression_>::array_type
      array_type; // The array type
  typedef typename TiledArray::expressions::ExprTrait<MyLeafExpression_>::engine_type
      engine_type; // Expression engine type

private:

  const array_type& array_; ///< The array that this expression
  std::string vars_; ///< The tensor variable string

  // Not allowed
  MyLeafExpression_& operator=(MyLeafExpression_&);

public:

  /// Constructors

  MyLeafExpression(const array_type& array, const std::string& vars) :
    Expr_(), array_(array), vars_(vars) { }

  MyLeafExpression(const MyLeafExpression_& other) :
    array_(other.array_), vars_(other.vars_)
  { }

  const array_type& array() const { return array_; }

  const std::string& vars() const { return vars_; }

}; // class MyLeafExpression
```

### Unary Expression Interface
```c++
#include <tiledarray.h>

// Forward declarations
template <typename> class MyUnaryExpression;
template <typename> class MyUnaryEngine;

namespace TiledArray {
  namespace expressions {

    // Define expression types
    template <typename Arg>
    struct ExprTrait<MyUnaryExpression<Arg> > : 
      public UnaryExprTrait<Arg, MyUnaryEngine> 
    { };

  } // namespace expressions
} // namespace TiledArray


// My expression object
template <typename Arg>
class MyUnaryExpression : public UnaryExpr<MyUnaryExpression<Arg> > {
public:
  typedef MyUnaryExpression <Arg> MyUnaryExpression_; // This class type
  typedef UnaryExpr<MyUnaryExpression_> UnaryExpr_; // Unary base class type
  typedef typename TiledArray::expressions::ExprTrait< MyUnaryExpression_ >::argument_type
      argument_type; // The argument expression type
  typedef typename TiledArray::expressions::ExprTrait< MyUnaryExpression_ >::engine_type
      engine_type; // Expression engine type

private:

  // Not allowed
  MyUnaryExpression_& operator=(const MyUnaryExpression_&);

public:

  // Constructors

  MyUnaryExpression(const argument_type& arg) :
    UnaryExpr_(arg)
  { }

  MyUnaryExpression(const MyUnaryExpression_& other) :
    UnaryExpr_(other)
  { }

}; // class MyScalingExpr

```


### Binary Expression Interface


## Expression Engine Interface

# Example Expression
This example demonstrated how to construct a scaling expression. The TiledArray tile operation `Scal` is used for demonstration purposes, but you are free to substitute your own tile operation. See Customizing Tile Operations for details.

```c++
#include <tiledarray.h>

// Forward declarations
template <typename> class MyScalingExpr;
template <typename> class MyScalingEngine;

namespace TiledArray {
  namespace expressions {

    // Define engine types
    template <typename Arg>
    struct EngineTrait<MyScalingEngine<Arg> > :
      public UnaryEngineTrait<Arg, TiledArray::math::Scal>
    { };

    // Define expression types
    template <typename Arg>
    struct ExprTrait<ScalExpr<Arg> > : 
      public UnaryExprTrait<Arg, ScalEngine> 
    { };

  } // namespace expressions
} // namespace TiledArray

// My scaling expression engine
template <typename Arg>
class MyScalingEngine : public UnaryEngine<MyScalingEngine<Arg> > {
public:
  // Class hierarchy typedefs
  typedef MyScalingEngine<Arg> MyScalingEngine_; // This class type
  typedef UnaryEngine<MyScalingEngine_> UnaryEngine_; // Unary expression engine base type
  typedef typename UnaryEngine_::ExprEngine_ ExprEngine_; // Expression engine base type

  // Argument typedefs
  typedef typename EngineTrait<MyScalingEngine_>::argument_type argument_type; // The argument expression engine type

  // Operational typedefs
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::value_type
      value_type; // The result tile type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::scalar_type
      scalar_type; // Tile scalar type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::op_type
      op_type; // The tile operation type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::policy
      policy; // The result policy type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::dist_eval_type
      dist_eval_type; // The distributed evaluator type

  // Meta data typedefs
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::size_type
      size_type; // Size type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::trange_type
      trange_type; // Tiled range type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::shape_type
      shape_type; // Shape type
  typedef typename TiledArray::expressions::EngineTrait<MyScalingEngine_>::pmap_interface
      pmap_interface; // Process map interface type

private:

  scalar_type factor_; // Scaling factor

public:

  // Constructor
  template <typename A>
  MyScalingEngine(const MyScalingExpr<A>& expr) :
    UnaryEngine_(expr), factor_(expr.factor())
  { }

  // Non-permuting shape factory function
  shape_type make_shape() const {
    return UnaryEngine_::arg_.shape().scale(factor_);
  }

  // Permuting shape factory function
  shape_type make_shape(const Permutation& perm) const {
    return UnaryEngine_::arg_.shape().scale(factor_, perm);
  }

  // Non-permuting tile operation factory function
  op_type make_tile_op() const { return op_type(factor_); }

  // Permuting tile operation factory function
  op_type make_tile_op(const Permutation& perm) const { return op_type(perm, factor_); }

  // Expression identification tag used for printing
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[" << factor_ << "] ";
    return ss.str();
  }

}; // class MyScalingEngine

// Scaling expression
template <typename Arg>
class ScalExpr : public UnaryExpr<ScalExpr<Arg> > {
public:
  typedef MyScalingExpr<Arg> MyScalingExpr_; // This class type
  typedef UnaryExpr<MyScalingExpr_> UnaryExpr_; // Unary base class type
  typedef typename TiledArray::expressions::ExprTrait<MyScalingExpr_>::argument_type argument_type; // The argument expression type
  typedef typename TiledArray::expressions::ExprTrait<MyScalingExpr_>::engine_type engine_type; // Expression engine type
  typedef typename TiledArray::expressions::ExprTrait<MyScalingExpr_>::scalar_type scalar_type; // Scalar type

private:

  scalar_type factor_; ///< The scaling factor

  // Not allowed
  ScalExpr_& operator=(const ScalExpr_&);

public:

  // Constructors

  MyScalingExpr(const argument_type& arg, const scalar_type factor) :
    UnaryExpr_(arg), factor_(factor)
  { }

  MyScalingExpr(const MyScalingExpr_& other, const scalar_type factor) :
    UnaryExpr_(other), factor_(other.factor_ * factor)
  { }

  MyScalingExpr(const MyScalingExpr_& other) : UnaryExpr_(other), factor_(other.factor_) { }

  /// Scaling factor accessor
  scalar_type factor() const { return factor_; }

}; // class MyScalingExpr

// Expression object factory functions

template <typename D, typename Scalar>
inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalExpr<D> >::type
operator*(const Expr<D>& expr, const Scalar& factor) {
  return ScalExpr<D>(expr.derived(), factor);
}

template <typename D, typename Scalar>
inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalExpr<D> >::type
operator*(const Scalar& factor, const Expr<D>& expr) {
  return ScalExpr<D>(expr.derived(), factor);
}

template <typename Arg, typename Scalar>
inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalExpr<Arg> >::type
operator*(const ScalExpr<Arg>& expr, const Scalar& factor) {
  return ScalExpr<Arg>(expr, factor);
}

template <typename Arg, typename Scalar>
inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalExpr<Arg> >::type
operator*(const Scalar& factor, const ScalExpr<Arg>& expr) {
  return ScalExpr<Arg>(expr, factor);
}
```
