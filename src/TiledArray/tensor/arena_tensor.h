/// ToT inner-tile type: pimpl-style pinned tensor backed by an arena cell.
///
/// `ArenaTensor<T, Range>` is one pointer wide. Its referent is a `Cell`
/// (range header + co-located element storage, aligned for both) that the
/// outer tile's arena allocates and owns. The `ArenaTensor` itself is
/// non-owning; copies/moves rebind the pointer. Lifetime is bounded by the
/// outer tile that owns the arena slab.

#ifndef TILEDARRAY_TENSOR_ARENA_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_TENSOR_H__INCLUDED

#include "TiledArray/error.h"
#include "TiledArray/math/blas.h"
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/tensor/type_traits.h"

#include <btas/zb/range.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace TiledArray {

/// Alignment of in-arena element storage, in bytes. Sized to cover the
/// widest common SIMD register (AVX-512 ZMM = 64 B) and a single x86_64
/// cache line. Override at configure time by defining
/// TILEDARRAY_INNER_SIMD_ALIGN to a larger power-of-two (e.g. 128 for
/// two-cache-line floor / Apple-Silicon L1 line size).
#ifndef TILEDARRAY_INNER_SIMD_ALIGN
#define TILEDARRAY_INNER_SIMD_ALIGN 64
#endif

inline constexpr std::size_t kInnerSimdAlign = TILEDARRAY_INNER_SIMD_ALIGN;
static_assert((kInnerSimdAlign & (kInnerSimdAlign - 1)) == 0,
              "kInnerSimdAlign must be a power of two");

template <typename T, typename Range_ = ::btas::zb::RangeNd<>>
class ArenaTensor;

// Forward decls of the free in-place CPOs (defined below). Needed so the
// member compound operators and member in-place CPOs on `ArenaTensor` can
// reference them.
template <typename T, typename R, typename Scalar>
void scale_to(ArenaTensor<T, R>& dst, Scalar factor);
template <typename T, typename R>
void add_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src);
template <typename T, typename R>
void subt_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src);
template <typename T, typename R>
void mult_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src);
template <typename T, typename R, typename Scalar>
void axpy_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src,
             Scalar alpha);

template <typename T, typename Range_>
class ArenaTensor {
 public:
  using value_type = T;
  using numeric_type = typename detail::numeric_type<T>::type;
  using scalar_type = typename detail::scalar_type<T>::type;
  using range_type = Range_;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;

  /// In-arena layout: range header, then padding, then element storage.
  struct Cell {
    range_type range;
  };

  /// Alignment of the element pointer past the cell header. Caller-owned
  /// arena slots must honour this so SIMD loads/stores on `data()` are
  /// aligned without an extra runtime check.
  static constexpr size_type data_alignment() noexcept {
    return alignof(T) > kInnerSimdAlign ? alignof(T) : kInnerSimdAlign;
  }

  /// Offset (in bytes) of the first element past the cell header.
  static constexpr size_type data_offset() noexcept {
    constexpr size_type a = data_alignment();
    return (sizeof(Cell) + a - 1) & ~(a - 1);
  }

  /// Total bytes a cell holding `n` elements consumes in the arena.
  static constexpr size_type cell_size(size_type n) noexcept {
    return data_offset() + n * sizeof(T);
  }

  /// Required alignment of a cell allocation. At least `data_alignment()`
  /// so that `cell_base + data_offset()` lands on a SIMD boundary, and at
  /// least `alignof(Cell)` so the range header is well-aligned.
  static constexpr size_type cell_alignment() noexcept {
    constexpr size_type da = data_alignment();
    return alignof(Cell) > da ? alignof(Cell) : da;
  }

  ArenaTensor() = default;
  ArenaTensor(const ArenaTensor&) = default;
  /// Move construction transfers the view and leaves the source null.
  ArenaTensor(ArenaTensor&& other) noexcept : cell_(other.cell_) {
    other.cell_ = nullptr;
  }
  ~ArenaTensor() = default;

  /// Unified assignment, with two regimes keyed on whether `*this` is bound:
  ///  - bound (non-null) assignee: deep element-wise copy from `src` -- the
  ///    view's storage already exists, so assignment writes into it;
  ///  - null assignee: a shallow rebind of the view to `src`'s cell -- there
  ///    is no storage to deep-copy into.
  /// This must be a user-provided non-template operator: the implicit
  /// copy-assignment (a shallow pointer copy) would otherwise be generated
  /// and, as a non-template exact match, would always shadow the templated
  /// `operator=` below for `ArenaTensor` sources. There is deliberately no
  /// move-assignment -- an rvalue `ArenaTensor` binds here and follows the
  /// same two regimes (moving a view buys nothing over copying it).
  ArenaTensor& operator=(const ArenaTensor& src) {
    if (cell_ == nullptr) {
      cell_ = src.cell_;  // null assignee: rebind the view (shallow)
      return *this;
    }
    return assign_elements_(src);  // bound assignee: deep copy
  }

  /// Construct a view onto a `Cell` (placement-newed by the arena factory).
  explicit ArenaTensor(Cell* cell) noexcept : cell_(cell) {}

  /// True if the view points at a non-null cell.
  explicit operator bool() const noexcept { return cell_ != nullptr; }

  /// True if the view is null (no cell).
  bool empty() const noexcept { return cell_ == nullptr; }

  /// Range of the referenced cell. UB if null.
  const range_type& range() const noexcept {
    TA_ASSERT(cell_ != nullptr);
    return cell_->range;
  }

  /// Pointer to the first element. Null when the view is null.
  pointer data() noexcept {
    if (cell_ == nullptr) return nullptr;
    auto* base = reinterpret_cast<std::byte*>(cell_);
    return std::launder(reinterpret_cast<pointer>(base + data_offset()));
  }

  const_pointer data() const noexcept {
    if (cell_ == nullptr) return nullptr;
    auto* base = reinterpret_cast<const std::byte*>(cell_);
    return std::launder(reinterpret_cast<const_pointer>(base + data_offset()));
  }

  /// Element count of the referenced cell, or 0 if null.
  size_type size() const noexcept {
    return cell_ != nullptr ? cell_->range.volume() : 0;
  }

  reference operator[](size_type i) noexcept {
    TA_ASSERT(cell_ != nullptr);
    return data()[i];
  }
  const_reference operator[](size_type i) const noexcept {
    TA_ASSERT(cell_ != nullptr);
    return data()[i];
  }

  /// Sum of all elements; `value_type{}` for a null view. A scalar
  /// reduction allocates nothing, so it is valid on a view (unlike the
  /// value-returning tensor ops, which are deliberately absent).
  value_type sum() const noexcept {
    value_type acc{};
    if (cell_ == nullptr) return acc;
    const auto* s = data();
    for (size_type i = 0; i < size(); ++i) acc += s[i];
    return acc;
  }

  /// Element-wise deep copy from a non-`ArenaTensor` tensor `src`. Valid only
  /// for a bound (non-null) assignee: a null view has no storage to copy into
  /// and a non-view `src` has no cell to rebind to (use the `ArenaTensor`
  /// overload above for the rebind regime).
  template <typename Src,
            typename = std::enable_if_t<detail::is_tensor_v<Src> &&
                                        !std::is_same_v<Src, ArenaTensor>>>
  ArenaTensor& operator=(const Src& src) {
    TA_ASSERT(cell_ != nullptr &&
              "cannot assign a non-ArenaTensor source to a null ArenaTensor");
    return assign_elements_(src);
  }

  /// In-place compound operators -- ArenaTensor is a view (no allocation),
  /// so it provides only the *mutating* counterparts to the value-returning
  /// `+`, `-`, `*` operators. Each delegates to the same-named free CPO
  /// (forward-declared above, defined later in this header). The pair
  /// (ArenaTensor x ArenaTensor) is the only one needed by TA's kernel
  /// paths. Calls are fully-qualified to avoid recursing into the member
  /// overloads of the same names below.
  ArenaTensor& operator+=(const ArenaTensor& other) {
    ::TiledArray::add_to(*this, other);
    return *this;
  }
  ArenaTensor& operator-=(const ArenaTensor& other) {
    ::TiledArray::subt_to(*this, other);
    return *this;
  }
  ArenaTensor& operator*=(const ArenaTensor& other) {
    ::TiledArray::mult_to(*this, other);
    return *this;
  }
  /// Scalar in-place compound assignment.
  template <typename Scalar>
    requires(detail::is_numeric_v<Scalar>)
  ArenaTensor& operator*=(const Scalar factor) {
    ::TiledArray::scale_to(*this, factor);
    return *this;
  }

  /// Member-call mirrors of the free in-place CPOs. Tile-interface paths
  /// (`add_to(result, arg)`, `subt_to`, etc.) and `Tensor`'s legacy
  /// `inplace_binary` use these. Bodies fully-qualify the free CPO call so
  /// the member doesn't recurse into itself.
  ArenaTensor& add_to(const ArenaTensor& other) {
    ::TiledArray::add_to(*this, other);
    return *this;
  }
  ArenaTensor& subt_to(const ArenaTensor& other) {
    ::TiledArray::subt_to(*this, other);
    return *this;
  }
  ArenaTensor& mult_to(const ArenaTensor& other) {
    ::TiledArray::mult_to(*this, other);
    return *this;
  }
  template <typename Scalar>
    requires(detail::is_numeric_v<Scalar>)
  ArenaTensor& scale_to(const Scalar factor) {
    ::TiledArray::scale_to(*this, factor);
    return *this;
  }
  ArenaTensor& neg_to() {
    ::TiledArray::scale_to(*this, -T(1));
    return *this;
  }

  /// axpy: <tt>*this += other * factor</tt> (axpy semantics; factor scales
  /// only the added operand). Delegates to the free `axpy` CPO that the
  /// outer-cell loop ultimately calls. Distinct from
  /// `add_to(other, factor)` which would be the legacy
  /// `(*this + other) * factor` semantics -- view tile types don't have
  /// `operator+=` returning a value, so we keep the names separated.
  template <typename Scalar>
    requires(detail::is_numeric_v<Scalar>)
  ArenaTensor& axpy_to(const ArenaTensor& other, const Scalar factor) {
    ::TiledArray::axpy_to(*this, other, factor);
    return *this;
  }

  /// axpy + fused permutation. ArenaTensor is a fixed-layout view, so any
  /// non-empty permutation is rejected at runtime.
  template <typename Scalar, typename Perm>
    requires(detail::is_numeric_v<Scalar> && detail::is_permutation_v<Perm>)
  ArenaTensor& axpy_to(const ArenaTensor& other, const Scalar factor,
                       const Perm& perm) {
    TA_EXCEPTION(
        "ArenaTensor::axpy_to(other, factor, perm): inner permutation is not "
        "supported for view cells");
    return *this;
  }

  /// Internal accessor for the cell pointer. Used by the arena factory and
  /// by destruction walks; not part of the user-facing surface.
  Cell* cell() const noexcept { return cell_; }

 private:
  /// Deep element-wise copy into this bound view's storage from any tensor
  /// `src` of matching volume (an `ArenaTensor` or an owning tensor alike).
  template <typename Src>
  ArenaTensor& assign_elements_(const Src& src) {
    TA_ASSERT(cell_ != nullptr);
    TA_ASSERT(size() == static_cast<size_type>(src.size()));
    auto* dst = data();
    const auto* src_data = src.data();
    for (size_type i = 0; i < size(); ++i) dst[i] = src_data[i];
    return *this;
  }

  Cell* cell_ = nullptr;
};

namespace detail {

/// Placement-construct an `ArenaTensor<T, R>` at the given pre-aligned,
/// pre-sized buffer. `buffer` must be at least
/// `ArenaTensor<T,R>::cell_size(range.volume())` bytes and aligned to
/// `ArenaTensor<T,R>::cell_alignment()`. Element storage is
/// value-initialized (zero for arithmetic `T`).
template <typename T, typename R>
ArenaTensor<T, R> make_arena_tensor_in(std::byte* buffer, R range) {
  using Inner = ArenaTensor<T, R>;
  using Cell = typename Inner::Cell;
  TA_ASSERT(buffer != nullptr);
  TA_ASSERT(
      reinterpret_cast<std::uintptr_t>(buffer) % Inner::cell_alignment() == 0);
  const std::size_t n = range.volume();
  Cell* cell = ::new (static_cast<void*>(buffer)) Cell{std::move(range)};
  T* elems = reinterpret_cast<T*>(buffer + Inner::data_offset());
  if constexpr (std::is_trivially_constructible_v<T>) {
    std::memset(elems, 0, n * sizeof(T));
  } else {
    for (std::size_t i = 0; i < n; ++i)
      ::new (static_cast<void*>(elems + i)) T();
  }
  return Inner(cell);
}

/// Destruct in-place. Mirrors `make_arena_tensor_in`'s construction. Safe
/// on a null view (no-op). After this call the cell memory is uninitialized;
/// the arena slab still owns the bytes.
template <typename T, typename R>
void destruct_arena_tensor(ArenaTensor<T, R>& inner) noexcept {
  auto* cell = inner.cell();
  if (cell == nullptr) return;
  const std::size_t n = cell->range.volume();
  if constexpr (!std::is_trivially_destructible_v<T>) {
    T* elems = inner.data();
    for (std::size_t i = 0; i < n; ++i) elems[i].~T();
  }
  if constexpr (!std::is_trivially_destructible_v<R>) {
    cell->~Cell();
  }
}

}  // namespace detail

/// `is_tensor_view<T>` is forward-declared in `tensor/type_traits.h` (primary
/// = `std::false_type`). Specializations for the concrete view types live
/// below; `external/btas.h` adds a spec for `btas::TensorView`. Distinct
/// from `is_tensor_helper`, which is also true for views (they are tensors
/// structurally) -- `is_tensor_view` is the *secondary* gate that opts views
/// out of value-returning member-call paths.

/// True iff `T` is some `ArenaTensor<U, R>` -- the arena-pinned view type.
/// Implies `is_tensor_view_v<T>`. Use this trait only where arena slab
/// machinery is actually managed (e.g. clone, serialize, value-returning
/// add/subt/mult that allocate via `arena_trivial_*_pinned`); for the
/// "no value-returning ops on a view" gating use `is_tensor_view_v` instead.
template <typename T>
struct is_arena_tensor : std::false_type {};
template <typename T, typename R>
struct is_arena_tensor<ArenaTensor<T, R>> : std::true_type {};
template <typename T>
inline constexpr bool is_arena_tensor_v = is_arena_tensor<T>::value;

// Every ArenaTensor is a view.
template <typename T, typename R>
struct is_tensor_view<ArenaTensor<T, R>> : std::true_type {};

namespace detail {

/// Register `ArenaTensor` as a tensor: it has the same `.data()` / `.size()`
/// flat-contiguous-storage shape as `TA::Tensor`. This makes
/// `is_tensor<ArenaTensor>` true and `is_tensor_of_tensor<Tensor<ArenaTensor>>`
/// true via the existing recursion, so kernel-level dispatches
/// (`tensor_reduce`, `inplace_tensor_op`, `tensor_op`, ...) match the same
/// overloads they do for `TA::Tensor<double>` without bespoke arena
/// overloads. To keep ArenaTensor out of value-returning member-call paths
/// (which require allocation that views can't do), `ta_ops_match_tensor` is
/// specialized below to false for `ArenaTensor`.
template <typename T, typename R>
struct is_tensor_helper<ArenaTensor<T, R>> : public std::true_type {};

/// ArenaTensor's element storage is contiguous and row-major.
template <typename T, typename R>
struct is_contiguous_tensor_helper<ArenaTensor<T, R>> : public std::true_type {
};

/// `ArenaTensor` counts as one nesting level, so `Tensor<ArenaTensor<T>>`
/// out-ranks a plain `Tensor<T>`. Without this, `nested_rank<ArenaTensor>`
/// falls through to the primary `= 0` and `einsum`'s `MaxNestedArray` ties a
/// ToT arena array with a plain array, picking the wrong result tile type.
template <typename T, typename R>
constexpr size_t nested_rank<ArenaTensor<T, R>> = 1 + nested_rank<T>;

template <typename T, typename R>
constexpr size_t nested_rank<const ArenaTensor<T, R>> =
    nested_rank<ArenaTensor<T, R>>;

}  // namespace detail

// Note: `detail::TensorInterface` (a.k.a. `TA::TensorMap`) is non-owning,
// but it *does* provide value-returning member arithmetic (`.add()`,
// `.subt()`, ...) that materializes a fresh tensor. So it does NOT
// participate in `is_tensor_view` -- this trait is reserved for views that
// lack value-returning member arith (cannot allocate on their own), like
// `ArenaTensor` and `btas::TensorView`.

}  // namespace TiledArray

// btas::TensorView is btas's existing non-owning view type. Register it as
// a view too. Forward-declared here (signature mirrors btas/tensorview.h)
// to avoid pulling that header into arena_tensor.h.
namespace btas {
template <typename _T, class _Range, class _Storage, class _Policy>
class TensorView;
}  // namespace btas

namespace TiledArray {
template <typename T, class R, class S, class P>
struct is_tensor_view<::btas::TensorView<T, R, S, P>> : std::true_type {};

/// Zero all elements of `dst`. No-op on a null view.
template <typename T, typename R>
void zero(ArenaTensor<T, R>& dst) noexcept {
  if (!dst) return;
  std::memset(dst.data(), 0, dst.size() * sizeof(T));
}

/// Fill `dst` with `value`. No-op on a null view.
template <typename T, typename R, typename U>
void fill(ArenaTensor<T, R>& dst, const U& value) {
  if (!dst) return;
  std::fill_n(dst.data(), dst.size(), static_cast<T>(value));
}

/// `dst *= factor`. No-op on a null view.
template <typename T, typename R, typename Scalar>
void scale_to(ArenaTensor<T, R>& dst, Scalar factor) {
  if (!dst) return;
  auto* d = dst.data();
  const auto n = dst.size();
  for (std::size_t i = 0; i < n; ++i) d[i] *= factor;
}

/// `dst += src`. Asserts both views non-null and shape-compatible.
template <typename T, typename R>
void add_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src) {
  if (!dst || !src) return;
  TA_ASSERT(dst.size() == src.size());
  auto* d = dst.data();
  const auto* s = src.data();
  for (std::size_t i = 0; i < dst.size(); ++i) d[i] += s[i];
}

/// `dst -= src`. Asserts both views non-null and shape-compatible.
template <typename T, typename R>
void subt_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src) {
  if (!dst || !src) return;
  TA_ASSERT(dst.size() == src.size());
  auto* d = dst.data();
  const auto* s = src.data();
  for (std::size_t i = 0; i < dst.size(); ++i) d[i] -= s[i];
}

/// `dst *= src` element-wise. Asserts both views non-null and shape-compatible.
template <typename T, typename R>
void mult_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src) {
  if (!dst || !src) return;
  TA_ASSERT(dst.size() == src.size());
  auto* d = dst.data();
  const auto* s = src.data();
  for (std::size_t i = 0; i < dst.size(); ++i) d[i] *= s[i];
}

/// `dst += src * alpha` (in-place BLAS-like AXPY). Asserts both views
/// non-null and shape-compatible. Argument order matches TA's `_to` CPO
/// convention `(result, arg, factor)`; the BLAS name AXPY captures the
/// semantics (in-place, not value-producing).
template <typename T, typename R, typename Scalar>
void axpy_to(ArenaTensor<T, R>& dst, const ArenaTensor<T, R>& src,
             Scalar alpha) {
  if (!dst || !src) return;
  TA_ASSERT(dst.size() == src.size());
  auto* d = dst.data();
  const auto* s = src.data();
  for (std::size_t i = 0; i < dst.size(); ++i) d[i] += alpha * s[i];
}

/// Sum of squared elements; 0 for null views.
template <typename T, typename R>
auto squared_norm(const ArenaTensor<T, R>& src) noexcept {
  T acc{};
  if (!src) return acc;
  const auto* s = src.data();
  for (std::size_t i = 0; i < src.size(); ++i) acc += s[i] * s[i];
  return acc;
}

/// Copy `src` into a freshly-allocated `Standalone`. Returns a default-
/// constructed (null) `Standalone` when `src` is null.
template <typename Standalone, typename T, typename R>
Standalone materialize(const ArenaTensor<T, R>& src) {
  if (!src) return Standalone();
  Standalone out(src.range());
  std::copy_n(src.data(), src.size(), out.data());
  return out;
}

/// GEMM CPO for `ArenaTensor`: accumulates `result += factor * left * right`
/// via BLAS. The result must be pre-allocated (e.g. zero-initialized by
/// `arena_outer_init`) -- this overload never resizes. More specific
/// than `tile_op/tile_interface.h`'s generic `gemm` template (which would
/// otherwise fall through to a nonexistent `result.gemm(...)` member),
/// so partial ordering picks it for `ArenaTensor` arguments.
template <typename T, typename R, typename Scalar>
auto gemm(ArenaTensor<T, R>& result, const ArenaTensor<T, R>& left,
          const ArenaTensor<T, R>& right, Scalar factor,
          const math::GemmHelper& gemm_helper)
    -> std::enable_if_t<detail::is_numeric_v<Scalar>, ArenaTensor<T, R>&> {
  if (!left || !right) return result;
  TA_ASSERT(bool(result));
  TA_ASSERT(left.range().rank() == gemm_helper.left_rank());
  TA_ASSERT(right.range().rank() == gemm_helper.right_rank());

  using integer = math::blas::integer;
  integer M, N, K;
  gemm_helper.compute_matrix_sizes(M, N, K, left.range(), right.range());

  const integer lda =
      (gemm_helper.left_op() == math::blas::NoTranspose) ? K : M;
  const integer ldb =
      (gemm_helper.right_op() == math::blas::NoTranspose) ? N : K;
  const integer ldc = N;

  math::blas::gemm(gemm_helper.left_op(), gemm_helper.right_op(), M, N, K,
                   static_cast<T>(factor), left.data(), lda, right.data(), ldb,
                   T(1), result.data(), ldc);
  return result;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_ARENA_TENSOR_H__INCLUDED
