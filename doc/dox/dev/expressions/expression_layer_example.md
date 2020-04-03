# Expression Layer Example {#Expression-Layer-Example}

This page walks through a simple expression breaking down how the
domain-specific language (DSL) is evaluated. We focus on the example:

```.cpp
TArray<float> a, b, c; // a and b are assumed initialized, c initialized below
c("i,j") = a("i,k,l") * b("k,j,l"); // contraction over indices k and l
```

The actual evaluation of this follows C++'s operator precedence
[rules](https://en.cppreference.com/w/cpp/language/operator_precedence), which
for this case amounts to:

1. Function call operator (`TsrArray<float>::operator()(const std::string&)`)
2. Multiplication operator
3. Direct assignment

## Step 1: Form TsrExpr Instances

The call operator, `TArray<float>::operator()(const std::string&)`, creates an
instance of the `TsrExpr<TArray<float>>` class. Each instance holds the string
indices provided to the call operator as well as a reference to the
`TArray<float>` instance whose call operator was invoked. If we define some
temporaries according to:

```.cpp
using tensor_expr = TsrExpr<TArray<float>>; // the full type of the TsrExpr
tensor_expr indexed_c = c("i,j");
tensor_expr indexed_a = a("i,k,l");
tensor_expr indexed_b = b("k,j,l");
```

then conceptually our example becomes:

```.cpp
TArray<float> a, b, c; // a and b are assumed initialized, c initialized below
using tensor_expr = TsrExpr<TArray<float>>; // the full type of the TsrExpr
tensor_expr indexed_c = c("i,j");
tensor_expr indexed_a = a("i,k,l");
tensor_expr indexed_b = b("k,j,l");
indexed_c = indexed_a * indexed_b;
```

This step largely serves as the kicking-off point for creating the abstract
syntax tree (AST) and does little else.

## Step 2: Multiplication

Step 1 formed leaves for the AST, step 2 creates a branch by resolving the
multiplication operator. In this case
`operator*(const Expr<tensor_expr>&,const Expr<tensor_expr>&)` is selected. The
resulting `MultExpr<tensor_expr, tensor_expr>` instance contains references to
`indexed_a` and `indexed_b`, but again little else has happened. If we define:

```.cpp
MultExpr<tensor_expr, tensor_expr> a_x_b = indexed_a * indexed_b;
```
then conceptually our example becomes:

```.cpp
TArray<float> a, b, c; // a and b are assumed initialized, c initialized below
using tensor_expr = TsrExpr<TArray<float>>; // the full type of the TsrExpr
tensor_expr indexed_c = c("i,j");
tensor_expr indexed_a = a("i,k,l");
tensor_expr indexed_b = b("k,j,l");
MultExpr<tensor_expr, tensor_expr> a_x_b = indexed_a * indexed_b;
indexed_c = a_x_b;
```

## Step 3: Assignment

This is where the magic happens. For our particular example the overload that
gets called is `indexed_c`'s `operator=`, which has the signature:

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

## Generalizing to More Complicated Expressions

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
