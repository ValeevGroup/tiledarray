# Expression Engine Example {#Expression-Engine-Example}

This page continues the explanation of how the expression layer works by
focusing on how engines work. Specifically we focus on the engine for the
contraction:

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

## Step 1: Construction

The input to constructor for the overall engine is the overall expression to be
evaluated. The constructor itself does little aside from setting the default
values and propagating the  map from the
input expression to the engine.

## Step 2: Initialization

The overall engine's `init` member is called and provided: the parallel runtime,
the process map, and a `VariableList` instance containing the final tensor
indices (*i.e.*, the indices of the tensor being assigned to). The `init`
member function ultimately calls:

1. `init_vars`
2. `init_struct`
3. `init_distribution`

in that order.


### Step 2a: Variable Initialization

Variable initialization starts in `MultEngine::init_vars(const VariableList&)`.
This function starts by calling the `init_vars()` members of the engines for the
expressions on the left and right sides of the `*` operator (for leaves of the
AST, like the present case, this is a noop). After giving the expressions on the
left and right a chance to initialize their variables the variables for the left
and right side are used to determine what sort of multiplication this is (a
contraction or a Hadamard-product) and the appropriate base class's `perm_vars`
is then called.

The purpose of the `init_vars()`/`init_vars(const VariableList&)` calls are to
compute the indices of the tensor which results from a particular branch of the
AST. The difference between the two calls is whether or not the final indices
are being provided as a hint to subexpressions (the form accepting one argument
is provided the hint). In addition to simply working out the resulting indices
these functions also work out any permutations (and therefore transposes) needed
to get the indices in the final order.

### Step 2b: Struct Initialization

This function initializes the details for the resulting tensor, specifically:

1. the permutation to go from the result of the expression to the result tensor
2. the tiling of the result tensor
3. the sparsity of the resulting tensor

Within the `MultEngine` instance this is done first for both the left and right
expressions. Next, taking the type of multiplication occurring into account, the
results of calling `struct_init` on the left and right expressions are combined
to generate the details of the tensor resulting from the multiplication.

## Step 3: Make a Distributed Evaluator

For our `MultEngine` the type of the distributed evaluator is
`DistEval<Tensor<float>, DensePolicy>`. Again this process is repeated
recursively for the left and right expressions in `MultEngine` resulting in the
distributed evaluators for the left and right expressions, which are then
forwarded to the PIMPL for the DistEval returned by `MultEngine`. The
distributed evaluator is what actually runs the operation.



