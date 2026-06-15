# `dot_inner`: a first-class inner-dot expression for ToT operands

_Status: design, 2026-06-15_

## Problem

After recent work (commits `9602d4f3d`, `5e8d5b4d3`), `TA::einsum` is -- apart
from legacy ops -- exercised in production for exactly **one** family of
operations: **inner reductions of the form ToT \* ToT → T**, where ToT is a
tensor-of-tensors `DistArray<Tensor<Tensor<S>>>` and T is a plain
tensor-of-scalars `DistArray<Tensor<S>>`. For each surviving outer cell the two
inner tensors are fully contracted down to a scalar; the outer modes form a
general product (contraction + Hadamard + external).

Example (the motivating general case):

```cpp
// outer:  (i,j,k) x (j,k) -> (k,i)   [contract j, Hadamard k, external i]
// inner:  (a,b,c) . (b,c,a)          [full dot, permuted alignment]
C("k,i") = A("i,j,k;a,b,c").dot_inner(B("j,k;b,c,a"));
```

We want this expressible directly in the expression DSL, so the remaining
production dependency on `einsum` can be retired and the operation gains a
proper lazy-expression surface.

## Scope (and deliberate non-scope)

**In scope:** a single new expression method, `Expr::dot_inner`, that computes a
per-outer-cell **dot product** over the inner modes while applying the standard
general product over the outer modes. Result tile type is plain T.

**Out of scope -- `dot` only, no general inner reduction.** Routing through the
existing general-product contraction engine fixes *both* combine operations to
summation: the inner GEMM computes `sum over inner indices of product` (a dot),
and the outer-contracted index is accumulated additively by the same GEMM.
A genuinely general inner reduction (e.g. `max`, arbitrary fold) cannot reuse
this machinery -- it would require a bespoke evaluator that walks outer cells,
applies an arbitrary inner pair→scalar op, and folds the contracted outer index
with an arbitrary combiner, i.e. re-growing the einsum-with-custom-op path we
are retiring. No current consumer needs this; every einsum ToT\*ToT→T case is a
contraction. **YAGNI: we implement `dot` only.**

_Future note (not built now):_ if a non-dot inner reduction is ever needed and
the outer pattern is **Hadamard-only** (no contracted outer index, hence no
outer fold to generalize), the operation degenerates to a binary elementwise map
and its natural home is the `foreach`/binary-op path, **not** the contraction
engine.

**Conjugation is not built in.** `dot_inner` is the non-conjugating contraction,
matching TA's `dot` (vs. the conjugating `inner_product`). `inner_product`
semantics are obtained by pushing a lazy `.conj()` onto the left operand:

```cpp
C("k,i") = A("i,j,k;a,b,c").conj().dot_inner(B("j,k;b,c,a"));
```

This keeps the kernel to exactly one path.

## API surface

A new method on the expression base, returning a **lazy expression** (assignable
to a plain-T `DistArray`), *not* a `Future` -- it has surviving outer indices, so
unlike the global `Expr::reduce`/`dot` it is array-valued:

```cpp
// in expressions/expr.h, on Expr<Derived>
template <typename D>
DotInnerExpr</*...*/> dot_inner(const Expr<D>& right) const;
```

- **Operands** must be ToT expressions; the result tile type is
  `Tensor<S>` where `S` is the inner numeric type
  (`value_type::value_type::value_type` of the ToT operand). No custom op
  parameter -- the inner contraction is fully determined by the inner annotations
  (all inner indices contracted; no surviving inner mode).
- **Annotations** carry the full outer general product on the outer slot and the
  inner alignment (e.g. `a,b,c` vs `b,c,a`) on the inner slot. The inner result
  annotation is empty (denest).
- **Name rationale:** anchored on `dot` (non-conjugating), so it stays clear of
  the existing conjugating `inner_product`; `_inner` denotes "over the inner
  (nested) modes." Method form parallels the existing `Expr::reduce`/`dot`
  methods and reads left-to-right.

## What exists today (verified)

`TA::einsum` produces a denested T result from ToT operands through **one** path:
the `DeNest::True` branch (`einsum/tiledarray.h:584-644`). Every ToT*ToT→T call
lands here. Its strategy is exactly the "materialized phantom-unit + squeeze"
(Design B below):

1. Append a phantom-unit label (`⊗₁`) to the right operand's inner annotation and
   run an ordinary ToT*ToT→**ToT** product whose result carries a unit-extent
   inner cell: `C0(c;⊗₁) = A(a;…) * B(b;…,⊗₁)` (`tiledarray.h:631-637`).
2. `foreach`-squeeze each `[1]` inner cell to a scalar
   (`sum_tot_2_tos`, `tiledarray.h:612-644`).

The actual contraction in step 1 is **already native** in the expression layer's
general-product engine for *all* outer patterns -- pure-Hadamard / no-Hadamard
(`tiledarray.h:663-668`), fused-broadcast (`683-709`), no-external (`809-830`),
and the full h+e+i general product including the motivating example
(`generalized-expression`, `tiledarray.h:999-1009`). The legacy per-tile local
kernel (`832-977`) is retained only as the opt-in cross-check oracle. The engine
also already recognizes the phantom-unit result annotation and realizes the inner
product as a flat dot into a `[1]` cell (`cont_engine.h:1542-1580`).

**Conclusion:** the outer general-product machinery is fully reusable and the
inner dot kernel already exists. What does *not* exist is a native T-valued
result -- today the engine always produces a unit-inner ToT that `einsum` squeezes.
There is **no** raw-DSL spelling (`C("k,i") = A("…;…") * B("…;…")` with plain-T
`C`) that denests; `result_of_mult_t<ToT,ToT>` is ToT, so denesting only happens
via the `einsum` phantom-unit dance. `dot_inner` closes exactly that gap.

## Design: native scalar result (no unit cell, no squeeze)

This is "Design A" from the brainstorm -- produce a plain `Tensor<S>` result tile
directly, with no `[1]` inner cell and no `foreach` squeeze.

The only two things that distinguish `dot_inner` from `operator*` on ToT
operands are: **(a)** the inner op writes a *scalar* instead of an inner tensor,
and **(b)** the result tile type is `T` not `ToT`. Everything else -- the outer
permutation optimizer, the outer GemmHelper, `SparseShape` propagation, the
distributed SUMMA/Hadamard scheduling -- is reused verbatim.

There is **no inner GEMM and no unit mode**: the existing phantom-unit element op
already reads both operand inner cells *flat* and computes `acc += lp[j]*rp[j]`
(`cont_engine.h:1571`). `dot_inner` keeps that flat dot and simply targets a
scalar (`result += acc`) instead of a `[1]` cell. Verified that the surrounding
machinery already accepts this: `Tensor::gemm` with a custom element op
(`tensor.h:3220-3227`) is SFINAE'd only on the op being invocable with
`(value_type&, const U&, const V&)` -- no same-element-type constraint -- and
`ContractReduce::operator()` already routes the non-plain case to it
(`contract_reduce.h:431`). The **sole** required engine edit is relaxing the
`ContractReduceBase` taxonomy assert (`contract_reduce.h:71-84`), which currently
forbids a non-nested result for nested operands.

### Result-type deduction

Add a deduction that maps `(ToT, ToT) → DistArray<Tensor<S>, Policy>` for the
`dot_inner` node, rather than the nesting-preserving `result_of_mult_t<ToT,ToT>
→ ToT`. `S` is the inner numeric type.

### Inner op: reuse the existing phantom-unit denest element op

The inner kernel already exists -- `ContEngine` contains a **phantom-unit denest**
path (`cont_engine.h:1542-1580`). When every result inner index is a phantom-unit
label (`⊗ₙ`), i.e. the real inner modes are fully contracted, it installs an
`element_nonreturn_op_` that:

- reads both operand inner cells flat (neither carries the phantom mode -- no
  inner GEMM, no `ContractReduce` rank match), computes the flat non-conjugating
  dot `acc += lp[j] * rp[j]` (`cont_engine.h:1571`), applies `factor_`, and
- accumulates the scalar into the lone element of a unit-extent `[1]^phantom_rank`
  result *cell* (`result.data()[0] += acc`).

This is already exactly the "only a dot is expressible" kernel, and it is the
**only** inner-mode reduction the contraction engine can express -- confirming the
dot-only scope. The outer engine already calls this op repeatedly with `+=` to
fold the contracted outer index, giving the additive outer contraction for free.

`dot_inner`'s change is narrow: instead of writing into a unit-extent inner
**tensor** cell (which einsum then squeezes to a scalar via `foreach`), make
`result_tile_element_type` a **scalar** and have the element op do
`result += acc` directly. The dot loop itself is unchanged. This eliminates both
the materialized unit-inner result ToT and the `foreach` squeeze pass that the
current `einsum` denest performs (see *What exists today*).

### Structural choice

Two ways to host this; recommend the first:

1. **Reuse `MultEngine`/`ContEngine` with a "denest-to-scalar" mode** (a trait or
   ctor flag), plus a thin `DotInnerExpr` node that configures it. Least
   duplication; all outer/shape/scheduling logic is shared. Cost: adds a mode to
   an already-complex engine.
2. **A standalone `DotInnerExpr` + `DotInnerEngine`** that internally drives a
   `ContEngine`. Cleaner separation, but risks copying engine wiring.

Recommend (1), gated behind a clearly-named mode so the contraction engine's
existing paths are untouched when the flag is off.

## Testing

`einsum`'s ToT\*ToT→T path is retained as the **correctness oracle** (it is
already the legacy cross-check vehicle; see `einsum_legacy_subworld()`). Every
`dot_inner` case is compared element-wise against the equivalent `einsum(...)`
string result.

Cases to cover (append near the related ToT/general-product tests in
`tests/CMakeLists.txt` -- order is load-bearing):

- Outer **Hadamard-only**, outer **pure contraction**, outer **external-only**,
  and the **mixed** motivating case (`i,j,k × j,k → k,i`).
- Inner **permuted alignment** (`a,b,c` vs `b,c,a`) and identity alignment.
- **Empty / zero** inner tiles; uneven inner extents (true ToT, non-uniform).
- **Both policies** -- `DensePolicy` and `SparsePolicy` (verify the result
  `SparseShape` matches `einsum`'s; watch the `fill(0)`-on-sparse footgun).
- **Conjugation** via `A.conj().dot_inner(B)` ⇒ `inner_product` semantics.
- `@distributed` coverage: run under **np=1 and np=2** (general product has a
  multi-rank path; do not let it run np=1 only).

Requires `TA_ASSERT_POLICY=TA_ASSERT_THROW` for the throwing-precondition checks
(per the repo's test-build invariant).

## Open questions / verify during implementation

- **Result-type deduction is the crux.** Verified: no native T-valued ToT\*ToT
  path exists today (see *What exists today*); the engine produces a unit-inner
  ToT that `einsum` squeezes. The central implementation question is how to make
  the `dot_inner` node deduce `result_tile_element_type = S` (scalar) and have
  the phantom-unit element op (`cont_engine.h:1542-1580`) write `result += acc`
  into a scalar tile rather than a `[1]` cell -- without disturbing the
  nesting-preserving `operator*` deduction.
- **`SparseShape` for the T result.** Confirm the norm-GEMM shape propagation
  yields a valid T `SparseShape` (inner dot contributes to the outer norm
  estimate) and matches `einsum`.
- **Lazy `.conj()` composition.** Verify `.conj()` composes with `dot_inner` on
  ToT operands (conjugation of inner tensors folded into the dot) without a
  materialization pass.
- **`einsum` follow-up.** Once `dot_inner` is the production path, decide whether
  einsum's ToT\*ToT→T branch delegates to `dot_inner` or is kept solely as the
  oracle.

## Non-goals

- No custom inner-reduction op; no general (non-dot) reduction.
- No new global-scalar reduction; `Expr::reduce`/`dot` are unchanged.
- No change to nesting-preserving ToT\*ToT→ToT `operator*`.
