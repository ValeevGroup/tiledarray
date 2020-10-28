# Customizing TiledArray Expressions {#Customizing-Expressions}
End-user interface of TiledArray is a Domain-Specific Language (DSL), embedded in C++, for performing basic arithmetic and user-defined operations on `DistArray` objects. For example, contraction of order-3 tensors to produce an order-2 tensor can be expressed as
```.cpp
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
minimizing temporary storage, and combining arithmetic operations; DSL does NOT perform
more extensive term rewriting, such as operation reordering, factorization
(strength reduction), or common subexpression elimination. Such task can be
performed by the user (or, by an optimizing compiler), provided sufficient
understanding of how the TiledArray DSL expressions are evaluated.

## DSL Overview
DSL expressions are evaluated in a multi-stage process, each step of which can be overridden to suit your needs.

1. __construct an expression object__: expression objects are the nodes of the [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree) describing a composite expression 
2. __construct expression engine__: expression engines compute the metadata needed to evaluate the expression, such as the following properties of the result:
  1. index list
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
Expression objects capture the minimum amount of information required to define the operation. Namely, the expression objects capture the syntax of expressions and do not deal with other details (such as how the expression is actually evaluated, sizes of intermediate quantities, etc.). Together the objects represent an Abstract Syntax Tree representation of an expression. Since the expression objects only capture compile-time syntax of an expression, runtime evaluation of the expression involves converting AST to a tree of objects (engines) that contain the information necessary to actually evaluate the expression at runtime (see the [Engine Interface](@ref Engine-Interface) section for more details).

There are three basic types of expression objects:

1. leaf -- no arguments (e.g. an `Array` + index list)
2. unary -- one argument (e.g. negation)
3. binary -- two arguments (e.g. addition)

Non-leaf expression objects in TiledArray keep track of the argument expressions and expresson metadata (such as scaling factors). Leaf expression objects keep track of the target `DistArray` object to which the expression is bound to and the corresponding Einstein convention annotation (if any).

### Expression Example Walkthrough

To understand how expressions are represented and then evaluated let's consider a simple example:

```.cpp
c("i,j") = a("i,k,l") * b("k,j,l"); // contraction over indices k and l
```

Here we assume that `a`, `b`, are `c` are declared elsewhere as `DistArray` objects, and `a` and `b` have been initialized appropriately.

Since TA DSL is embedded into C++, evaluation of this expression follows C++'s operator precedence
[rules](https://en.cppreference.com/w/cpp/language/operator_precedence), which
for this case amounts to:

1. Function call operator (`TsrArray<float>::operator()(const std::string&)`)
2. Multiplication operator
3. Direct assignment

#### Step 1: Form TsrExpr Instances

The call operator, `TArray<float>::operator()(const std::string&)`, creates an
instance of the `TsrExpr<TArray<float>>` class. Each instance holds the string
indices provided to the call operator as well as a reference to the
`TArray<float>` instance whose call operator was invoked. If we define some
temporaries according to:

```.cpp
using tensor_expr = TsrExpr<TArray<float>>; // the full type of the TsrExpr
tensor_expr annotated_c = c("i,j");
tensor_expr annotated_a = a("i,k,l");
tensor_expr annotated_b = b("k,j,l");
```

then conceptually our example becomes:

```.cpp
TArray<float> a, b, c; // a and b are assumed initialized, c initialized below
using tensor_expr = TsrExpr<TArray<float>>; // the full type of the TsrExpr
tensor_expr annotated_c = c("i,j");
tensor_expr annotated_a = a("i,k,l");
tensor_expr annotated_b = b("k,j,l");
annotated_c = annotated_a * annotated_b;
```

This step largely serves as the kicking-off point for creating the abstract
syntax tree (AST) and does little else.

#### Step 2: Multiplication

Step 1 formed leaves for the AST, step 2 creates a branch by resolving the
multiplication operator. In this case
`operator*(const Expr<tensor_expr>&,const Expr<tensor_expr>&)` is selected. The
resulting `MultExpr<tensor_expr, tensor_expr>` instance contains references to
`annotated_a` and `annotated_b`, but again little else has happened. If we define:

```.cpp
MultExpr<tensor_expr, tensor_expr> a_x_b = annotated_a * annotated_b;
```
then conceptually our example becomes:

```.cpp
TArray<float> a, b, c; // a and b are assumed initialized, c initialized below
using tensor_expr = TsrExpr<TArray<float>>; // the full type of the TsrExpr
tensor_expr annotated_c = c("i,j");
tensor_expr annotated_a = a("i,k,l");
tensor_expr annotated_b = b("k,j,l");
MultExpr<tensor_expr, tensor_expr> a_x_b = annotated_a * annotated_b;
annotated_c = a_x_b;
```

#### Step 3: Assignment

This is where the magic happens. For our particular example the overload that
gets called is `annotated_c`'s `operator=`, which has the signature:

```.cpp
template<typename D>
TArray<float> TsrExpr<TArray<float>>::operator=(const Expr<D>&);
```

This passes tensor `c` to the provided expression's `eval_to` member. Inside
`eval_to`:

1. An engine is made
2. The engine is initialized
3. The engine generates a distributed evaluator
4. The distributed evaluator is run
5. Data is moved from the distributed evaluator to `c`.

These five steps contain a lot of detail which we are glossing over at this
level. More details pertaining to how the engine works can be found
[here](@ref Expression-Engine-Example).

#### Generalizing to More Complicated Expressions

The above three steps can be generalized to:

1. Form leaves
2. Form branches
3. Evaluate resulting tree

Step 1, form leaves, converts DSL-objects into expressions which can enter into
the expression layer. With the leaves formed, step 2 creates the branches by
combining: leaves with leaves, leaves with branches, and branches with branches.
The formation of branches is not associative and the exact structure is dictated
by C++'s operator precedence rules. AST formation terminates when the code has
been wrapped up into a single AST and that AST is being assigned to a `TsrExpr`
instance. Once AST formation is complete, step 3 occurs which is to evaluate the
AST and assign the result to the `TsrExpr`.

### Custom Expression Examples
To help you implement your own custom expressions copy and paste the appropriate  expression code templates below and fill in the implementation details.

#### Example: Custom Leaf Expression
```.cpp
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

#### Example: Custom Unary Expression
```.cpp
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


#### Example: Custom Binary Expression

Work in progress.

## Expression Engine Interface

Engine objects result from interpreting the AST using the runtime data (such as annotation strings, tensor ranks, etc.). They form a tree that represents a ready-to-evaluate expression. Evaluation of the engine tree is accomplished by traversing the tree and generating distributed evaluator objects that implement particular operations as DAGs of tasks.

### Example Walkthrough

Let's consider the same expression example as earlier to understand how engines are constructed and what they do:
```.cpp
TArray<float> a, b, c; // a and b are assumed initialized, c initialized below
c("i,j") = a("i,k,l") * b("k,j,l"); // contraction over indices k and l
```

The first engine is created when the AST is assigned to `c("i,j")`. Inside the
assignment operator, the `eval_to` member of the AST is invoked (and provided
`c`). For our present purposes `eval_to` does 3 relevant things:

1. It constructs the overall engine
2. It initializes the overall engine
3. It creates a distributed evaluator

For reference, in the contraction example, the overall engine is of type:

```.cpp
// The type of the engine powering the TsrExpr instances
using tsr_engine = TsrEngine<TArray<float>, Tensor<float>, true>;

// The overall engine type of the equation
MultEngine<tsr_engine, tsr_engine, Tensor<float>>
```

#### Step 1: Construction

The input to constructor for the overall engine is the overall expression to be
evaluated. The constructor itself does little aside from setting the default
values and propagating the  map from the
input expression to the engine.

#### Step 2: Initialization

The overall engine's `init` member is called and provided: the parallel runtime,
the process map, and a `IndexList` instance containing the final tensor
indices (*i.e.*, the indices of the tensor being assigned to). The `init`
member function ultimately calls:

1. `init_vars`
2. `init_struct`
3. `init_distribution`

in that order.


##### Step 2a: Variable Initialization

Variable initialization starts in `MultEngine::init_vars(const IndexList&)`.
This function starts by calling the `init_vars()` members of the engines for the
expressions on the left and right sides of the `*` operator (for leaves of the
AST, like the present case, this is a noop). After giving the expressions on the
left and right a chance to initialize their variables the variables for the left
and right side are used to determine what sort of multiplication this is (a
contraction or a Hadamard-product) and the appropriate base class's `perm_vars`
is then called.

The purpose of the `init_vars()`/`init_vars(const IndexList&)` calls are to
compute the indices of the tensor which results from a particular branch of the
AST. The difference between the two calls is whether or not the final indices
are being provided as a hint to subexpressions (the form accepting one argument
is provided the hint). In addition to simply working out the resulting indices
these functions also work out any permutations (and therefore transposes) needed
to get the indices in the final order.

##### Step 2b: Struct Initialization

This function initializes the details for the resulting tensor, specifically:

1. the permutation to go from the result of the expression to the result tensor
2. the tiling of the result tensor
3. the sparsity of the resulting tensor

Within the `MultEngine` instance this is done first for both the left and right
expressions. Next, taking the type of multiplication occurring into account, the
results of calling `struct_init` on the left and right expressions are combined
to generate the details of the tensor resulting from the multiplication.

#### Step 3: Make a Distributed Evaluator

For our `MultEngine` the type of the distributed evaluator is
`DistEval<Tensor<float>, DensePolicy>`. Again this process is repeated
recursively for the left and right expressions in `MultEngine` resulting in the
distributed evaluators for the left and right expressions, which are then
forwarded to the PIMPL for the DistEval returned by `MultEngine`. The
distributed evaluator is what actually runs the operation.

# Complete Example: Multiplication by scalar
This example demonstrates how to implement scaling expressions by a scalar. The TiledArray tile operation `Scal` is used for demonstration purposes, but you are free to substitute your own tile operation. See the [User Defined Tiles](@ref User-Defined-Tiles) section for more details.

```.cpp
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
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  op_type make_tile_op(const Perm& perm) const {
    return op_type(perm, factor_);
  }

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
