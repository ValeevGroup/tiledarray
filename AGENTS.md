# Agent Instructions for TiledArray

Canonical guidance for coding agents working on this repo. For installing
dependencies and walking through a first build, see `INSTALL.md`. This file
covers what an agent needs beyond that: the source layout, how tests are
organized, the project-specific gotchas that are easy to get wrong, and the
invariants that downstream consumers (primarily MPQC) rely on.

`CLAUDE.md` is a symlink to this file; there is no vendor-specific variant.

TiledArray sits on top of MADWorld (MADNESS's parallel runtime). Anything
concerning task scheduling, MPI use, active messages, `World` lifetime, or
the MADNESS archive format lives in
[MADNESS `AGENTS.md`](https://github.com/m-a-d-n-e-s-s/madness/blob/master/AGENTS.md)
— do not restate it here. This file is TA-specific.

**How to use this file.** Treat it as a starting map, not authority. The
code is the source of truth — when this file disagrees with what
`grep`/`Read` show you, trust the code and flag the discrepancy so this
file can be updated.

_Last reviewed against master:_ 2026-04-23.

## Platforms

POSIX only — Linux and macOS. Windows is not supported and there are zero
`#ifdef _WIN32` branches in `src/TiledArray`. CI runs Ubuntu and macOS; do
not add Windows-specific code paths.

C++20 is required (`CMakeLists.txt`); older standards are rejected.
`cmake_minimum_required` is `3.21.0`.

## Source tree

Core library lives under `src/TiledArray/`:

- `dist_array.h` — `DistArray<Tile, Policy>`, the main user-facing type.
- `expressions/` — lazy expression DSL (`A("i,j") * B("j,k")` etc.). Nodes
  are built by `operator*`/`+`/`-` and evaluated on assignment.
- `einsum/` — higher-level Einstein-summation API (index strings, range
  parsing, generic contraction plumbing on top of `expressions/`).
- `dist_eval/` — distributed evaluation engine behind expressions (SUMMA
  contraction, unary/binary evaluators, dist-cache).
- `tile_op/` — tile-level op kernels (add, mult, scal, contract-reduce).
  `tile_op/tile_interface.h` is the canonical list of ADL customization
  points (free functions `add`, `subt`, `mult`, `gemm`, `scale`, …) a
  custom tile type must provide — default implementations forward to
  same-named member functions.
- `tile_interface/` — per-op headers (`cast`, `scale`, `trace`, `clone`,
  `permute`, `shift`) wired into those customization points.
- `tensor/` — `TA::Tensor`, view types (shift, of-tensor), tensor algebra.
- `policies/` — `DensePolicy`, `SparsePolicy` (shape type, storage, eval
  policy).
- `pmap/` — process maps (`BlockedPmap`, `CyclicPmap`, `RoundRobinPmap`,
  `HashPmap`, `ReplicatedPmap`).
- `math/` — BLAS/LAPACK wrappers, optional ScaLAPACK, CG solver, etc.
- `conversions/` — interop with `Eigen`, `btas::Tensor`, row-major arrays.
- `device/`, `host/` — GPU (CUDA/HIP) and host-side tile kernels; device
  code is gated by `TILEDARRAY_HAS_DEVICE`.
- `external/` — adapters for MADNESS, Eigen, BTAS, device runtimes, TTG.
- `special/` — diagonal arrays, CP decomposition, retile, random fills.
- `symm/` — permutation/symmetry utilities.
- `util/` — allocators, timers, `bug.cpp` debugger hooks, logging.
- `error.h` — assertion/exception macros (see *Coding conventions*).
- `initialize.h`, `tiledarray.cpp` — `TA::initialize` / `TA::finalize`.

Adjacent trees: `tests/` (Boost.Test suite), `examples/` (sample apps),
`python/` (PyTA bindings, gated by `TA_PYTHON`), `doc/` (Doxygen).

## Build directories

Prefer reusing an existing out-of-tree build directory over creating a new
one — CMake configure of TA is slow because it pulls MADNESS via
FetchContent on a fresh tree. Common local conventions here are
`cmake-build-{debug,release}` and `build*`; if several exist side-by-side,
`grep … CMakeCache.txt` to see which toggles each was configured with
(CUDA/HIP/ScaLAPACK/TTG state diverges).

There is no TA equivalent of MADNESS's `MADNESS_BUILD_MADWORLD_ONLY` —
building the TA library always compiles the full numerical stack. To reduce
scope during iteration, build only the specific CTest target (e.g. a single
test binary, or `check_serial-tiledarray`) instead of `all`.

Dependency behavior: optional deps (MADNESS, BTAS, range-v3, Umpire,
LibreTT, Boost if too old, TTG) follow a `FindOrFetch*` pattern —
`find_package` first, then FetchContent at the commit pinned in
`external/versions.cmake`. Set `TA_EXPERT=ON` to disable the FetchContent
fallback (offline / packager mode): missing deps then fail configure
instead of being built.

## Tests

Tests live in `tests/` as a single monolithic `ta_test` Boost.Test binary
(not per-file CTest entries). `tests/CMakeLists.txt` lists sources in
a **specific order** that the in-process fixture graph depends on — new
tests should be appended near related tests, not alphabetized.

CTest registers:

- `tiledarray/unit/build` — fixture that builds `ta_test` on demand.
- `tiledarray/unit/run-np-1` — runs `ta_test` with MPI world size 1,
  excluding tests tagged `@distributed` (`--run_test=!@distributed`).
- `tiledarray/unit/run-np-2` — MPI world size 2, excluding `@serial` tests.

Convenience targets:

```
cmake --build . --target check-tiledarray          # both np=1 and np=2
cmake --build . --target check_serial-tiledarray   # np=1 only (faster)
```

Under the hood these are named `<target>-tiledarray` via
`add_custom_target_subproject` (from the ValeevGroup cmake kit) so the
plain target names don't collide when TA is pulled in as a subproject.

Two test-time expectations worth knowing:

- **`MAD_NUM_THREADS=2`** is set by CTest (`TA_UNIT_TESTS_ENVIRONMENT` in
  `tests/CMakeLists.txt`). Running `ta_test` by hand without this exposes
  serial-only bugs; set it yourself when debugging.
- **`TA_CUDA_NUM_STREAMS=1`** is also set — some CUDA unit tests still
  assume a single stream.
- **`TA_UT_DISTRIBUTED=1`** is set for the np≥2 runs so fixtures can
  tell which side they're on.

The test suite also requires `TA_ASSERT_POLICY=TA_ASSERT_THROW` to
exercise `BOOST_REQUIRE_THROWS` checks. Any other policy is allowed to
build but silently skips those assertions (CMake warns). Default in
Release/MinSizeRel is `TA_ASSERT_IGNORE`, so an "optimized" build is not a
faithful test build.

A single test can be run directly by passing a Boost.Test filter:

```
./tests/ta_test --run_test=tensor_suite/some_case --log_level=unit_scope
```

There is no Python-level test driver; `tests/` is all C++.

## Project-specific gotchas

- **MPI is required by default.** `ENABLE_MPI=ON` is the default and the
  well-exercised path; `ENABLE_MPI=OFF` routes through MADNESS's
  `stubmpi.h`, which only declares the subset of MPI that `SafeMPI` uses
  (see MADNESS AGENTS.md). Any new code that touches MPI must go through
  `SafeMPI` / `World::mpi` / `World::gop`, not raw `MPI_*` calls.
- **TA forces `DISABLE_WORLD_GET_DEFAULT=ON` on MADNESS.**
  `FindOrFetchMADWorld.cmake` sets this so that `madness::World::get_default()`
  is unavailable inside MADNESS when built as a TA subproject. Library code
  must take an explicit `World&` argument; reaching for the default will
  fail to compile here. TA's own `TiledArray::get_default_world()` is the
  app-level escape hatch — use it sparingly and never in library headers.
- **MADNESS is pinned by commit hash**, not a version tag, in
  `external/versions.cmake` (`TA_TRACKED_MADNESS_TAG`). Bumping it is a
  deliberate, non-trivial event — MADNESS's task-pool ABI and
  serialization format are not covered by SemVer at this pin cadence.
  Touch the pin only when coordinating with a known MADNESS change.
- **MADNESS is FetchContent'd with `MADNESS_BUILD_MADWORLD_ONLY=ON`** —
  `FindOrFetchMADWorld.cmake` sets this so the `mra`/`chem`/`apps` layers
  are not pulled in. If you start leaning on anything outside
  `src/madness/world`, the link will fail and the fix is upstream, not
  local.
- **`MPI_THREAD_MULTIPLE` is mandatory.** Forced via
  `MPI_THREAD "multiple"` in `FindOrFetchMADWorld.cmake`. Not negotiable.
- **`TA_ASSERT_POLICY` affects the ABI**, not just behavior — compiled
  translation units and the unit-test driver must agree. Defaults:
  `TA_ASSERT_THROW` for Debug/RelWithDebInfo, `TA_ASSERT_IGNORE` for
  Release/MinSizeRel (`CMakeLists.txt`). Don't mix. Unit tests require
  `THROW`.
- **`TA_SIGNED_1INDEX_TYPE=ON` is the default** since 1.0.0-alpha.3.
  Flipping it changes the coordinate type across the entire public API
  (ranges, tile shapes, pmaps) — this is an ABI-level toggle, not a local
  choice.
- **BLAS threading.** TA does not enforce sequential BLAS the way MADNESS
  does; the linear-algebra discovery kit (`TA_LINALG_DISCOVERY_KIT=ON`,
  default) configures BLAS++/LAPACK++. For distributed contractions the
  MADWorld caveat still applies — threaded BLAS inside a task
  oversubscribes cores. When in doubt, pin BLAS to one thread.
- **GPU support is off by default.** `TA_CUDA=OFF`, `TA_HIP=OFF`; there is
  no SYCL backend. When either is ON, `TILEDARRAY_HAS_DEVICE` is defined
  and `device/`, LibreTT, and Umpire-managed unified memory get wired in.
  CI does **not** exercise GPU paths, so device changes need local
  validation.
- **ScaLAPACK (`TA_SCALAPACK=ON`) is off by default.** When on, it's
  resolved via `external/scalapackpp.cmake`; failures typically surface
  at link time rather than configure.

## Coding conventions

### Error handling

`src/TiledArray/error.h` defines a small set of macros. Pick by *why*:

- **`TA_ASSERT(expr, …)`** — the sole internal invariant check. Behavior
  is controlled by `TA_ASSERT_POLICY`: throws `TiledArray::Exception`
  (default in Debug), aborts (`TA_ASSERT_ABORT`), or compiles to a no-op
  (`TA_ASSERT_IGNORE`, default in Release). Because Release elides them,
  do **not** use `TA_ASSERT` for checks that must run in production — use
  `TA_EXCEPTION` at the throw site instead. The optional `…` is a message
  for future use; it is currently dropped from the failure string.
- **`TA_EXCEPTION(msg)`** — unconditional throw of `TiledArray::Exception`
  with `file:line` annotation. Use for user-visible errors, unreachable
  branches, unsupported configurations.
- **`TA_USER_ERROR_MESSAGE(msg)`** — stderr diagnostic, compiled out by
  `-DTILEDARRAY_NO_USER_ERROR_MESSAGES` (which the unit-test driver
  defines). Use only at top-level library entry points; library-internal
  code should throw.
- There is no `TA_USER_ASSERT` and no `TA_ASSERT_NOEXCEPT`. Public API
  argument validation that must survive Release should be an
  `if (…) TA_EXCEPTION(…)`, not an assert.

The debugger break-on-exception hook is `TiledArray::exception_break()` —
set a breakpoint there to stop before a throw.

### CMake harness

- **Subproject-scoped targets.** Generic names like `check`, `check_serial`
  would collide when TA is pulled in as a subproject (e.g. by MPQC). Use
  `add_custom_target_subproject(tiledarray <name> …)` (provided by the
  ValeevGroup cmake kit, loaded in the top-level `CMakeLists.txt`); it
  creates `<name>-tiledarray` and hooks it into `<name>` if the parent
  defines one.
- **`add_ta_executable(name sources libs)`** — trivial wrapper around
  `add_executable(... EXCLUDE_FROM_ALL …)`. There is no `add_ta_library`;
  the library is a single target defined in `src/CMakeLists.txt`.
- **Consumable from both install tree and build tree.** The library is
  exported via `install(EXPORT tiledarray …)` plus
  `export(EXPORT tiledarray FILE tiledarray-targets.cmake)` in the top
  `CMakeLists.txt`. Missing install rules on new public headers or
  hard-coded build-tree paths are silent regressions for downstream.
- **`FindOrFetch*` pattern** — see `cmake/modules/FindOrFetch{MADWorld,BTAS,
  RangeV3,UmpireCXXAllocator,TTG}.cmake`. Try `find_package` first, then
  FetchContent at the pinned commit. Honor `TA_EXPERT=ON` (skip
  FetchContent).
- **`CMAKE_FIND_NO_INSTALL_PREFIX=ON`** is set at the top so a stale
  installed TA copy in `CMAKE_INSTALL_PREFIX` doesn't get picked up during
  development. Leave it alone.

### Commit messages

Write plain commit messages describing the change. **Do not append
`Co-Authored-By: …` trailers crediting AI or any other tooling** (Claude, Copilot,
Codex, etc.). `git log` on master has zero such trailers; match that.

### Lint

`.clang-format` is checked in (project style). There is no `.clang-tidy`
at the source root — don't introduce one as part of an unrelated change.

## Consumer-facing invariants

TA's primary downstream is MPQC, which consumes TA headers directly and
depends on stable `DistArray`, expression, and tile-interface behavior.
Preserve these; they are not enforced at compile time and breakage
surfaces far downstream.

### Lifetime & init

- `TiledArray::initialize(argc, argv[, comm][, quiet])` is the entry
  point. It can also be called when MADWorld is already initialized —
  provided the supplied `comm` is MADWorld's default World; otherwise it
  throws.
- `TiledArray::finalize()` **fences the default World** before tearing
  down device state and (if TA initialized MADWorld) calling
  `madness::finalize`. MADWorld cannot be re-initialized, so a process
  that let TA initialize MADWorld gets exactly one init/finalize cycle.
- `TA_SCOPED_INITIALIZE(argc, argv)` / `scoped_finalizer()` give RAII
  shutdown; prefer these in examples and apps.

### `DistArray<Tile, Policy>`

`TA::DistArray<Tile, Policy>` is an **asynchrony-friendly, distributed,
tiled order-N array** whose tiles are of type `Tile`. Understanding the
pieces below is prerequisite to reasoning about anything built on top of
it.

**Tiled range.** The index space is a `TA::TiledRange` — the Cartesian
product of per-mode `TA::TiledRange1`s. Each `TiledRange1` is a
contiguous sequence of tiles that partitions its 1-D element range:
every element belongs to exactly one tile, and empty (zero-width) tiles
are allowed. The underlying element range is a `Range1` whose `lobound`
need not be 0 — tilings over non-zero-based coordinate systems are
first-class and common in practice. The canonical use case is a
sub-array carved out of a zero-based parent: keeping the parent's
coordinate system on the subblock means element/tile coordinates remain
meaningful relative to the original array, without forcing every caller
to switch between block-expression and standalone-array types.

**Policy.** Controls fundamental semantics. Two useful values:
`DensePolicy` (every tile is present and assumed nonzero) and
`SparsePolicy` (each tile is zero or nonzero; only nonzero tiles are
stored). For `SparsePolicy`, sparse structure is carried in the array's
**`SparseShape`** object, which tracks a Frobenius-norm estimate per
tile; a global threshold then decides which tiles count as zero. The
same per-tile norms are how TA **propagates sparse structure through
expressions** — in `C = A * B`, `C`'s shape is estimated from `A`'s and
`B`'s tile norms *before* the contraction, so below-threshold output
tiles are never computed.

**Distribution.** Data layout is a user-selectable **`Pmap`** (see
`pmap/` for `BlockedPmap`, `CyclicPmap`, `RoundRobinPmap`, `HashPmap`,
`ReplicatedPmap`). The array's body is literally a
`madness::WorldObject` distributed hash table from tile index to
`Future<Tile>`. Any rank may call `find(index)` for any tile, including
remote ones; for a remote tile, `find` returns a `Future` to a
**copy** of the remote tile that MADWorld will deliver via an automatic
point-to-point transfer. `find` does **not** cache — each call creates
a new copy. Futures are the primary asynchrony mechanism: a tile may be
pending because of an in-flight transfer, a pending computation, or
because nothing has written it yet.

**Bulk fill.** The canonical ways to populate a `DistArray` are
`init_tiles(f)` and `init_elements(f)` — each spawns tasks that run the
user-provided lambda `f` per tile / per element and writes results into
the array. User-visible construction followed by either init_* call is
the idiomatic "create & fill" sequence; avoid a loop of `set()` calls
from the main thread unless you have a specific reason.

- **Shallow-copy handle with deferred destruction.**
  `DistArray b = a;` shares the underlying `ArrayImpl` (hash-table body
  + shape + pmap). Mutations through `b` are observed through `a`; deep
  copy is `clone()`. Because a distributed taskflow may still be writing
  to or reading from the impl after the last local handle is dropped,
  destruction is lazy — the runtime keeps the impl alive until the next
  fence (see *Deferred impl destruction* below). The tiles themselves
  are separately reference-counted, so "the last `DistArray` handle is
  gone" does not imply "tile memory is freed".
- **Expressions are lazy.** `C("i,j") = A("i,k") * B("k,j")` builds an
  expression tree and, on assignment, enqueues tasks on a result `World`
  chosen by this priority (`expressions/expr.h`, `eval_to`):
  1. if `C` is already initialized, its `World` — `C.world()`;
  2. otherwise, if the expression was tagged with `.set_world(w)`, that `w`;
  3. otherwise, `TiledArray::get_default_world()`.
  A default-constructed / null `C` therefore does **not** bind the
  evaluator to any world of its own — override explicitly when that
  matters, e.g.
  `C("i,j") = (A("i,k") * B("k,j")).set_world(a_world);`.
  `.set_shape(shape)` on a subexpression overrides the result's shape
  similarly. `.set_pmap(pmap)` is an **expert-only** knob: most ops
  ignore the argument pmaps when scheduling work, and the most
  arithmetically intensive one — tensor contraction — maps onto a 2-D
  process grid that does not depend on argument or result pmaps at all.
  Reach for `set_pmap` only when manually tuning work distribution.
  See *Synchronization* below for when (and when not) to fence.
- **Multi-`World` composition.** Every rank is a member of many
  MADNESS `World`s (at minimum the top World and a per-rank local
  World — `World`s are abstractions over MPI sub-communicators).
  `DistArray`s can live on any of them, and expressions can freely
  mix operands from different `World`s. The caveat: for binary ops
  like `+`, `-`, and Hadamard-like `*`, the engine derives the
  intermediate's `World` from one of the arguments (typically the
  left). In a mixed-`World` expression where this matters, set the
  result `World` explicitly with `.set_world(w)` rather than relying
  on the implicit choice.
- **Collective construction.** `DistArray` ctors that take a
  `TiledRange` are collective on the enclosing `World`. All ranks must
  construct the same array in the same order (inherits the MADWorld
  rule).
- **Deferred impl destruction (expert-only).**
  `defer_deleter_to_next_fence()` delays destruction of the
  `DistArray`'s **implementation object** (the shared `ArrayImpl`)
  until the next fence on its `World`. It does **not** guarantee that
  the tiles the impl held live that long — `TA::Tensor` is itself a
  shallow-copy handle, so individual tiles stay alive as long as any
  other reference exists and are freed when the last one drops,
  fence or no fence. The mechanism exists because **lifetime of a
  distributed object can't be managed perfectly with local
  information** — a local handle going out of scope tells you nothing
  about pending tasks on other ranks that still reference the impl;
  the next fence is by definition a point where no such tasks are
  in flight. The reason it isn't always-on (i.e. the reason TA *does*
  destroy ArrayImpls eagerly when it can) is that TA workloads are
  typically resource-intensive and waiting for the next fence
  unnecessarily pins memory. Eager-but-still-lazy destruction is
  implemented in `ArrayImpl::lazy_deleter` (see `array_impl.h`).
  Non-experts should not call `defer_deleter_to_next_fence()` directly.
- **Not thread-safe for concurrent mutation.** Read-only access across
  threads is fine; mutation should be driven from tasks or the main
  thread. Callback forms (`for_each`, `for_each_inplace`, `init_tiles`)
  require the user callable to be thread-safe.

### `TA::Tensor<T, Allocator>`

The default tile type. A dense order-N array of elements of type `T`,
laid out as a **contiguous row-major** sequence in memory. Like
`DistArray`, it is a **shallow-copy handle** to a reference-counted
storage block — `Tensor b = a;` shares the buffer, and mutations through
`b` are observed through `a`. Deep copy is `clone()`. This is why
"destroying the last `DistArray` handle" does not, on its own, free tile
memory: tile handles held elsewhere (including in still-pending tasks)
keep the storage alive.

A `Tensor` is defined by its **`Range`** — a rectilinear N-tuple
integer range. The range is **not necessarily zero-based**: it is
defined by a pair of lobound/upbound N-tuples. This matters because each
tile in a `DistArray` carries a `Range` whose bounds are the tile's
footprint within the enclosing array's coordinate system, so the tile
encodes its own position. User code reading tile element `(i, j)` should
use the tile's `Range`-based indexing (e.g. `tile(i, j)` with `i`, `j`
in the tile's global coordinates) rather than rebasing to zero.

Views into a `Tensor` (slicing, shifts) do not own storage and must not
outlive the tensor they view; see `tensor/` for `TensorInterface` and
`tensor_shift_wrapper` types.

**Permutation is eager.** `TA::Tensor`'s physical layout matches its
logical layout, so `TA::Tensor::permute()` materializes the permuted
data; there is no lazy/strided view. Lazy permutation is a
*tile-type* property — user-defined tile types may opt into it, but
`TA::Tensor` does not. At the expression level, the contraction
engine can fuse a permutation into the GEMM kernel itself (the
permutation is absorbed into `gemm`'s argument layouts), so chains
like `C("j,i") = A("i,k") * B("k,j")` may elide the explicit
transpose of the contraction's output.

### `TA::Tile<T>`

A shallow-copy wrapper around a **deep-copy** tile type `T`. It exists
because `DistArray` and its tile ops are written against shallow-copy
semantics — that lets the runtime pass tiles around (across tasks,
ranks, fences) without either sprinkling `shared_ptr` through user code
or requiring every user tile type to implement efficient move semantics
(some data layouts don't move cheaply). The main real-world instantiation
is `TA::Tile<btas::Tensor<T>>`, since `btas::Tensor` is deep-copy.

Don't reach for `TA::Tile` around `TA::Tensor` — `TA::Tensor` is already
shallow-copy; wrapping it adds indirection without benefit.

### Custom tile types

Authoritative list: free functions declared in
`src/TiledArray/tile_op/tile_interface.h` (`add`, `subt`, `mult`, `gemm`,
`scale`, `permute`, `clone`, `trace`, `neg`, `shift`, …). These are ADL
customization points — to adopt a new tile type non-intrusively, overload
the subset your users actually call. A partial implementation is fine:
`tests/sparse_tile.h`'s `EigenSparseTile` omits `subt` because sparse
tiles are never subtracted in practice, and that's a supported pattern.

Default implementations of the free functions forward to same-named
member functions — `add(a, b)` calls `a.add(b)`. Three canonical shapes
in the tree:

- **`TA::Tensor`** implements all ops as members → no free-function
  overloads needed.
- **`TA::Tile`** has *no* tile-op members (it's a wrapper) → `tile.h`
  supplies free-function overloads that delegate to the wrapped
  deep-copy tile via the same non-intrusive API.
- **`btas::Tensor`** also lacks member ops → `external/btas.h` provides
  free-function overloads **in the `btas` namespace** (so ADL finds
  them, not in `TiledArray`). Device-specific further specializations
  for BTAS-backed unified-memory tiles live in
  `device/btas_um_tensor.h` and dispatch to device BLAS where possible.

**When to actually use `btas::Tensor`.** Almost never. Its remaining
production use is composing the tile type for GPU unified-memory
arrays; PR #531 will let `TA::Tensor` cover that case too, after which
`btas::Tensor` is purely a tile-genericity test vehicle. Don't reach
for it in new TA code unless you are deliberately exercising the
non-`TA::Tensor` path.

### Lazy tiles (`is_lazy_tile`)

Tile types that are computed on demand rather than stored eagerly. The
canonical example is **integral-direct AO integral tensors** in quantum
chemistry: integrals are regenerated each time they're needed instead of
being held in memory. Parametrizing a `DistArray` on a lazy tile type
lets it participate in expressions alongside ordinary arrays; during
evaluation, expression machinery invokes the lazy tile's conversion
operator to its `eval_type` (typically `TA::Tensor` or another
"storage" tile type) — see `Cast<...>` in `expressions/expr.h`.

If `eval_type` is a `Future<...>` rather than a bare tensor type, the
evaluation is asynchronous: the tile spawns a task on the World's
task queue and returns a `Future` to the result. That's the usual
pattern for compute-bound lazy tiles (integrals, on-disk reads).

Compressed-storage tiles are *not* typically implemented as lazy tiles
in TA, even though they conceptually could be: in practice you want
tile ops to act on the compressed representation directly, not
reconstruct the dense `eval_type`, so those are modeled as ordinary
(non-lazy) tiles with custom `tile_interface` overloads.

### Tensor-of-Tensor (ToT)

A nested tile type — `TA::Tensor<TA::Tensor<T>>` — used in MPQC for
PNO-based coupled-cluster methods. The defining property: **inner
tensors have extents that depend on the outer element index** (non-
uniform extents), so the nested form can't be flattened into a regular
rank-(M+N) tensor. This is the natural representation of slice-wise
compressed tensors and is what distinguishes ToT from "just a higher-
rank tensor".

The bipartite annotation syntax (`"i,j;k,l"`) separates outer modes
(left of the semicolon) from inner modes (right of it). Reviewing or
writing expression DSL around a ToT: the semicolon is load-bearing, not
decorative.

**Limitation, not obvious from the code:** the regular expression DSL
does *not* cover the full algebra ToT needs — some binary tensor
operations that arise in ToT work (inner contractions alongside outer
contractions, etc.) are simply not expressible as `A(...) * B(...)`.
The supported entry point for general ToT binary products is
`TA::einsum`. When an agent finds itself writing a ToT expression that
doesn't compile cleanly in the expression DSL, the fix is usually
"switch to `einsum`", not "hack around the DSL".

### Policies & shapes — gotchas

(See the `DistArray` subsection above for the underlying model.)

- **`fill(0)` on a `SparsePolicy` array yields an empty array** — no
  zero-value tiles are materialized, because the shape rejects
  below-threshold tiles. `fill(0)` on a `DensePolicy` array yields all
  zero-filled tiles. This asymmetry is a recurring footgun; when porting
  dense-first code to sparse, fill with a small nonzero value or
  restructure the work to be dense if you need the tiles to exist.
- **`set(idx, tile)` on a sparse array requires a nonzero tile.** TA
  contract-checks (via `TA_ASSERT`) that `idx` is in
  `array.trange().elements_range()` *and* that the shape would not
  filter the tile out (`!is_zero(idx)`). Sparse `DistArray`'s data
  model only stores tiles above threshold; there is no "sparse but
  with explicit zero tiles" mode. If you need an addressable zero tile
  at `(i, j)`, the array has to be dense (or the threshold has to make
  that tile nonzero).
- **No universal `SparseShape` threshold.** Noise propagating through
  nonlinear ops can be amplified arbitrarily, so the threshold is
  application-specific. For "binary" sparsity (a tile is zero exactly
  when its norm is exactly zero), use
  `std::numeric_limits<float>::min()` as the threshold so any nonzero
  norm clears the bar.
- **`TiledRange` / `Range` are immutable post-construction.** To change
  tiling, build a new array with `retile` / `TA::retile` and copy tiles
  over.

### Serialization

`DistArray` inherits `madness::archive::ParallelSerializableObject`.
Serialization goes through `madness::archive` and its format is tied to
the pinned MADNESS version — changing the MADNESS pin can invalidate
on-disk archives. Teach custom tile types to archive by specializing
`madness::archive` traits, not by reaching into TA internals.

### Expression DSL

- Annotation strings (`"i,j"`) are parsed eagerly but evaluation is
  deferred; the same string convention is expected across operands in one
  expression.
- Block expressions (`A("i,j").block(…)`) with empty blocks are a
  recently-stable path — see recent fixes to `preserve_lobound` /
  `eval_to` (PR #535). Empty-block handling should round-trip through
  `block(...)` and assignment without throwing.

### Synchronization

TA's expression / task model is dataflow-first; reach for a fence only
when you actually need one. The hierarchy from cheap to expensive:

- **Pure-dataflow expressions need no fence between them.** Tasks chain
  off upstream `Future<Tile>`s automatically, so
  `D = A * B; E = D + C;` is correct as written — TA wires `E`'s tasks
  to wait on `D`'s tile futures without a `gop.fence` in between.
- **Fence-free mutation idioms.** Use `init_tiles(f)` /
  `init_elements(f)` for fills and `TA::foreach(in, op)` for
  transform/map; each runs the user lambda in tasks and produces a
  populated array without an intervening fence. Unary
  `TA::foreach_inplace(A, op)` can be fence-free, but the **binary
  form `foreach_inplace(A, B)` fences on entry and exit** (commit
  464f50cdb) — both arrays must be in a consistent state before the
  in-place update can run.
- **Observing ops imply the synchronization they need.** `norm2`,
  `dot`, `trace`, `squared_norm`, `.find_local(...).get()`, etc.
  drive completion of the work they depend on. Prefer them when you
  only need the value; you don't need a separate fence on top.
- **`world.await(predicate)` for narrow synchronization.** When you
  only need to wait for the N tasks you just submitted *locally*
  (rather than a global rendezvous), an atomic counter + `await` is
  much cheaper than a fence:

  ```cpp
  std::atomic<size_t> done{0};
  size_t submitted = 0;
  for (...) {
    world.taskq.add([&done] { /* ... */ ++done; });
    ++submitted;
  }
  world.await([&] { return done.load() == submitted; });
  ```

- **`world.gop.fence()` is the brute-force option.** It synchronizes
  all ranks. Use it only when you need a true global rendezvous —
  e.g., before a side effect that has to observe all enqueued work.

## Runtime & deployment

Almost everything here is inherited from MADWorld — refer to MADNESS
`AGENTS.md` for:

- Task backend choice (`MADNESS_TASK_BACKEND={Pthreads,TBB,PaRSEC}`).
  TA enables TTG (`TA_TTG=ON`) only with `PaRSEC`; attempting another
  backend is a configure-time fatal error.
- Thread budgeting (`MAD_NUM_THREADS`, comm thread, MPI progress thread,
  never oversubscribing).
- Rank pinning to L3/NUMA, MPI eager/rendezvous cutoffs, buffer sizing
  (`MAD_BUFFER_SIZE` — TA bulk transfers benefit from raising it past the
  MADNESS default).
- Apple-Silicon + Homebrew OpenMPI HWLOC/OpenCL crash workaround.

### TA-specific environment variables

Set at runtime, read during `initialize` or first use:

| Variable | Default | Effect |
|---|---|---|
| `TA_LINALG_BACKEND` | (auto) | `scalapack`, `lapack`, or `ttg`. Invalid values throw. |
| `TA_LINALG_DISTRIBUTED_MINSIZE` | 4194304 | Matrix-size threshold above which the distributed LA backend is chosen automatically. |
| `TA_SUMMA_MAX_MEMORY` | (none) | Caps SUMMA contraction per-rank memory (`dist_eval/contraction_eval.h`). |
| `TA_SUMMA_MAX_DEPTH` | (none) | Caps SUMMA pipeline depth. |
| `TA_DEVICE_NUM_STREAMS` / `TA_CUDA_NUM_STREAMS` / `TA_HIP_NUM_STREAMS` | impl | Number of device streams per rank. CUDA unit tests require `=1`. |
| `TA_DEVICE_LEGACY_UM_CONVERSION` | unset | Opt into the pre-Umpire unified-memory conversion path for `btas_um_tensor`. |
| `TA_UT_DISTRIBUTED` | unset | Set by CTest for np≥2 runs; test fixtures branch on it. |

## Documentation

Doxygen-generated API docs are published at
<https://valeevgroup.github.io/tiledarray/dox-master/> (master branch).
The `doc/` tree builds those; user-facing install/build notes live in
`INSTALL.md`. There is no ReadTheDocs site for TA.

## Reviewing a PR

Guidance for agents doing PR review — focused on regressions CI won't
catch because its matrix is narrow (no GPUs, no ScaLAPACK on every cell,
no alternate task backends on every cell). Flag these; don't nitpick
style (`.clang-format` is the source of truth).

- **Raw MPI calls** outside `SafeMPI` / `World::mpi` — breaks
  `-DENABLE_MPI=OFF` and bypasses error checking.
- **Blocking MPI collectives** (`MPI_Barrier`, `MPI_Allreduce`, …) —
  push toward `World::gop`.
- **`madness::World::get_default()` in library code** — won't compile
  here because TA forces `DISABLE_WORLD_GET_DEFAULT=ON` on MADNESS, but
  may slip through in header-only contexts or parent-project builds.
  Push for an explicit `World&` parameter.
- **Public-header drift.** TA does **not** guarantee ABI or even
  source-level stability — releases are deliberately rare for that
  reason. That doesn't make breaking changes free: changes to
  `DistArray`, `Tensor`, `TiledRange`, tile-interface signatures, or
  `policies/*` ripple straight into in-tree downstreams (notably MPQC).
  Call out non-cosmetic public-header changes so the bump can be
  coordinated, even if no SemVer line is being crossed.
- **`DensePolicy`/`SparsePolicy` asymmetries.** New expressions or
  reductions should work on both. `fill(0)` and similar "fill with
  below-threshold value" paths are a known footgun on sparse.
- **`TA_ASSERT` standing in for a production check.** Release compiles
  these out; use `TA_EXCEPTION` instead.
- **MADNESS pin drift.** Any change to `TA_TRACKED_MADNESS_TAG` in
  `external/versions.cmake` — confirm the motivating MADNESS commit is
  merged and that TA still builds against it cleanly on both
  `Pthreads` and `PaRSEC` backends.
- **Device path changes without a GPU build run.** CI has no GPU cell;
  ask the author to confirm `-DTA_CUDA=ON` or `-DTA_HIP=ON` still builds
  and the device-tagged tests pass locally.
- **New tests appended out of order** in `tests/CMakeLists.txt`. The
  order is load-bearing — fixtures set up by earlier cases are consumed
  by later ones. New tests go near related tests, not at the end by
  default.
- **Tests that should be `@distributed`** but aren't tagged — they'll
  silently run under np=1 only and miss the multi-rank path. Conversely,
  `@serial` tags keep a test out of np=2 when it assumes world size 1.
- **Archive-format sensitivity.** Anything that changes what a
  `DistArray` writes through `madness::archive` is a silent on-disk
  compat break; call it out.

## Debugging aids

- `TA::exception_break()` — breakpoint here to stop just before any TA
  exception is thrown.
- `src/TiledArray/util/bug.cpp` — `launch_debugger_xterm_hook()` and
  friends; attach GDB/LLDB via xterm when `DISPLAY` is set.
- `TA_TRACE_TASKS=ON` (CMake) — enables MADNESS task tracing for TA
  components that opt in.
- `TA_ENABLE_TILE_OPS_LOGGING=ON` (CMake) — logs tile-op dispatch; set
  `TA_TILE_OPS_LOG_LEVEL` to control verbosity.
- `TA_TRACE_GLOBAL_COMM_STATS=ON` (CMake) — instruments `DistEval` and
  `DistributedStorage` with per-object communication counters.
- `TA_TENSOR_MEM_TRACE=ON` / `TA_TENSOR_MEM_PROFILE=ON` (CMake) —
  instrumented tracing/profiling of `TA::Tensor` memory use.
