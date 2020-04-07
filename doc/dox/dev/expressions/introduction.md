# Expressions in TiledArray {#Expressions-in-TiledArray}

The end-user interface of TiledArray is a Domain-Specific Language (DSL),
embedded in C++, for performing basic arithmetic and user-defined operations on
`DistArray` objects. For example, contraction of order-3 tensors to produce an
order-2 tensor can be expressed as:

```.cpp
TArray<float> a, b, c; // a and b are assumed initialized
c("i,j") = a("i,k,l") * b("k,j,l");
```

Even such a simple expression is implemented as a sequence of elementary
operations, such as permutation and tensor contraction; understanding what
happens under the hood is important for writing optimized TiledArray code.
Furthermore, TiledArray DSL is extensible: users can write their own expressions
to provide custom functions or optimizations. Understanding how expressions are
evaluated is needed for such customizations.

## DSL Pieces

- Expressions - Codify what the user wants us to do. Result in AST.
- Engine - Resolves metadata associated with the operation.
- DistEval - Actually does the operation.

## Examples

The following pages provide detailed analysis of how various parts of
TiledArray's DSL implement the contraction:

```.cpp
TArray<float> a, b, c;
c("i,j") = a("i,k,l") * b("k,j,l");
```

- [Expression Layer Example](@ref Expression-Layer-Example)
- [Expression Engine Example](@ref Expression-Engine-Example)

# Changes Needed

- operator()(std::string) counts commas, which won't work
- VariableList likely will need to understand semicolon
- Permutation class will need to know how to convert inner and outer indices
