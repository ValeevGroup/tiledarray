/// Arena kernels for tensor-of-tensor (ToT) outer tiles.
///
/// One slab-backed builder family, dispatching on the inner-tile type:
///   - `ArenaTensor` inners  -> slab of `Cell`s (range header + element data),
///                              each inner cell is an 8-byte view;
///   - `TA::Tensor` inners   -> slab of element data, each inner `Tensor`
///                              aliases its slice of the slab.
/// `is_arena_tensor_v<Inner>` selects the per-cell layout; everything else
/// (planning, allocation, outer-tile assembly) is shared.

#ifndef TILEDARRAY_TENSOR_ARENA_KERNELS_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_KERNELS_H__INCLUDED

#include "TiledArray/config.h"
#include "TiledArray/error.h"
#include "TiledArray/tensor/arena.h"
#include "TiledArray/tensor/arena_tensor.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace TiledArray {
namespace detail {

namespace {

/// Build outer storage whose deleter owns arena and alias keep-alive state.
template <typename OuterTensor, typename KeepAlive>
std::shared_ptr<typename OuterTensor::value_type[]> make_outer_data(
    std::size_t n_cells, std::shared_ptr<Arena> arena_handle,
    KeepAlive keep_alive) {
  using inner_t = typename OuterTensor::value_type;
  std::allocator<inner_t> allocator;
  inner_t* raw = allocator.allocate(n_cells);
  auto deleter =
      [allocator = std::move(allocator), arena_handle = std::move(arena_handle),
       keep_alive = std::move(keep_alive), n_cells](inner_t* p) mutable {
        for (std::size_t i = 0; i < n_cells; ++i) (p + i)->~inner_t();
        allocator.deallocate(p, n_cells);
        (void)arena_handle;
        (void)keep_alive;
      };
  return std::shared_ptr<inner_t[]>(raw, std::move(deleter));
}

}  // namespace

/// Allocate an arena-backed ToT outer tile with caller-provided inner ranges.
///
/// `inner_range_fn(cell_ordinal)` -> inner `range_type` for each cell ordinal
/// in `[0, outer_range.volume() * batch_sz)`; a zero-volume range yields a
/// deliberately-null inner cell that consumes no arena bytes. Element storage
/// is left zero-initialized when `zero_init` is true. `cell_stride_align` is
/// the minimum byte stride between adjacent cells; it is bumped up to the
/// inner type's natural alignment (`ArenaTensor::cell_alignment()`, or
/// `alignof(T)` for `TA::Tensor` inners).
///
/// This is the *up-front* path: all inner ranges are known, so the total is
/// pre-walked and laid down as a single exactly-sized arena page -- one
/// contiguous slab, no page-tail waste. For one-pass construction where the
/// inner sizes are discovered incrementally, use `ArenaToTBuilder`.
template <typename OuterTensor, typename InnerRangeFn>
OuterTensor arena_outer_init(
    const typename OuterTensor::range_type& outer_range, std::size_t batch_sz,
    InnerRangeFn&& inner_range_fn,
    std::size_t cell_stride_align = kArenaCachelineAlign,
    bool zero_init = true) {
  using InnerT = typename OuterTensor::value_type;
  using T = typename InnerT::value_type;
  using InnerRange = typename InnerT::range_type;
  constexpr bool arena = is_arena_tensor_v<InnerT>;

  std::size_t stride = cell_stride_align;
  if constexpr (arena) {
    if (InnerT::cell_alignment() > stride) stride = InnerT::cell_alignment();
  } else {
    if (alignof(T) > stride) stride = alignof(T);
  }
  // Cells pack at `stride` granularity; the page base must be at least
  // `max_align_t`-aligned.
  const std::size_t slab_align =
      stride > alignof(std::max_align_t) ? stride : alignof(std::max_align_t);

  const std::size_t N_cells = outer_range.volume() * batch_sz;
  std::vector<InnerRange> ranges;
  ranges.reserve(N_cells);
  std::size_t total = 0;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    ranges.emplace_back(inner_range_fn(ord));
    const std::size_t vol = ranges.back().volume();
    if (vol != 0) {
      // `if constexpr`, not a ternary: `InnerT::cell_size` does not exist for
      // a `TA::Tensor` inner, so the non-arena branch must not be formed.
      std::size_t bytes;
      if constexpr (arena)
        bytes = InnerT::cell_size(vol);
      else
        bytes = vol * sizeof(T);
      total += arena_align_up(bytes, stride);
    }
  }

  auto arena_ptr = std::make_shared<Arena>(std::pmr::new_delete_resource(),
                                           kArenaDefaultPageBytes, zero_init);
  // One exact page holds every cell -- subsequent `claim_bytes` calls pack
  // into it in order, reproducing the old single-slab layout.
  if (total > 0) arena_ptr->reserve_page(total, slab_align);
  auto data = make_outer_data<OuterTensor>(N_cells, arena_ptr,
                                           std::shared_ptr<InnerT[]>{});
  OuterTensor result(outer_range, batch_sz, std::move(data));

  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& r = ranges[ord];
    const std::size_t vol = r.volume();
    if (vol == 0) {
      if constexpr (arena) {
        ::new (result.data() + ord) InnerT();
      } else {
        // Rank-0 empties must preserve Tensor's null-data/no-range invariant.
        if (r.rank() == 0)
          ::new (result.data() + ord) InnerT();
        else
          ::new (result.data() + ord) InnerT(r);
      }
    } else if constexpr (arena) {
      auto h = arena_ptr->claim_bytes(InnerT::cell_size(vol), stride);
      ::new (result.data() + ord)
          InnerT(make_arena_tensor_in<T>(h.get(), std::move(r)));
    } else {
      auto h = arena_ptr->claim_bytes(vol * sizeof(T), stride);
      ::new (result.data() + ord)
          InnerT(r, std::shared_ptr<T[]>(h, reinterpret_cast<T*>(h.get())));
    }
  }
  return result;
}

/// Default (no-op) fill for `make_nested_tile` -- leaves element storage
/// zero-initialized.
struct nested_fill_noop {
  template <typename Cell, typename Index>
  void operator()(Cell&, const Index&) const noexcept {}
};

/// One-pass incremental builder for an arena-backed ToT outer tile.
///
/// `make_nested_tile` / `arena_outer_init` pre-walk every inner range before
/// any storage is allocated. `ArenaToTBuilder` instead sizes and binds inner
/// cells one at a time: the caller discovers each inner range and fills the
/// returned cell in a single step, driving its own loop. Backed by the
/// multi-page `Arena`, so no total size is needed up front.
///
/// Cells should be `emplace`d in outer cell-ordinal order -- recommended, not
/// required: a view is a pointer, so any order is correct, but in-order
/// emplacement keeps the page layout cache-friendly for later iteration.
/// `arena_compact` coalesces a finished tile into one contiguous slab.
///
/// A builder, its `Arena`, and the tile under construction are single-thread
/// objects (see `Arena`).
template <typename OuterTensor>
class ArenaToTBuilder {
 public:
  using outer_range_type = typename OuterTensor::range_type;
  using inner_t = typename OuterTensor::value_type;
  using inner_range_t = typename inner_t::range_type;
  using elem_t = typename inner_t::value_type;

  explicit ArenaToTBuilder(const outer_range_type& outer_range,
                           std::size_t batch_sz = 1, bool zero_init = false,
                           std::size_t page_size = kArenaDefaultPageBytes)
      : outer_range_(outer_range),
        batch_sz_(batch_sz),
        n_cells_(outer_range.volume() * batch_sz),
        arena_(std::make_shared<Arena>(std::pmr::new_delete_resource(),
                                       page_size, zero_init)) {
    data_ = make_outer_data<OuterTensor>(n_cells_, arena_,
                                         std::shared_ptr<inner_t[]>{});
    // Cells start null (the deleter destroys all n_cells_); `emplace` binds.
    for (std::size_t ord = 0; ord < n_cells_; ++ord)
      ::new (data_.get() + ord) inner_t();
  }

  /// Size and bind the inner cell at outer cell ordinal `ord` to
  /// `inner_range`, returning a reference to the bound cell for the caller to
  /// fill. A zero-volume range yields an empty cell: an owning inner keeps a
  /// rank>0 range (a rank-0 range stays null), a view inner stays null.
  /// Outer element indices translate via `outer_range().ordinal(idx)`.
  inner_t& emplace(std::size_t ord, inner_range_t inner_range) {
    TA_ASSERT(ord < n_cells_);
    inner_t& cell = data_[ord];
    constexpr bool arena = is_arena_tensor_v<inner_t>;
    const std::size_t vol = inner_range.volume();
    if (vol == 0) {
      // Mirror arena_outer_init: an owning (non-view) inner preserves a
      // rank>0 zero-volume range as an empty-but-ranked tensor; a rank-0
      // range -- and any arena view inner -- leaves the cell default/null.
      if constexpr (!arena) {
        if (inner_range.rank() != 0) cell = inner_t(std::move(inner_range));
      }
      return cell;
    }
    std::size_t stride;
    std::size_t bytes;
    if constexpr (arena) {
      stride = inner_t::cell_alignment();
      bytes = inner_t::cell_size(vol);
    } else {
      stride = alignof(elem_t);
      bytes = vol * sizeof(elem_t);
    }
    // Single-cell tile: lay down one exactly-sized page (corner case b).
    if (n_cells_ == 1 && arena_->empty()) arena_->reserve_page(bytes, stride);
    auto h = arena_->claim_bytes(bytes, stride);
    if constexpr (arena) {
      cell = make_arena_tensor_in<elem_t>(h.get(), std::move(inner_range));
    } else {
      cell = inner_t(
          std::move(inner_range),
          std::shared_ptr<elem_t[]>(h, reinterpret_cast<elem_t*>(h.get())));
    }
    return cell;
  }

  /// Finalize and hand back the assembled outer tile; the builder is spent.
  OuterTensor finish() && {
    return OuterTensor(outer_range_, batch_sz_, std::move(data_));
  }

  std::size_t cell_count() const noexcept { return n_cells_; }
  const outer_range_type& outer_range() const noexcept { return outer_range_; }
  const Arena& arena() const noexcept { return *arena_; }

 private:
  outer_range_type outer_range_;
  std::size_t batch_sz_;
  std::size_t n_cells_;
  std::shared_ptr<Arena> arena_;
  std::shared_ptr<inner_t[]> data_;
};

/// Build one ToT outer tile over `outer_range` in a single pass: each inner
/// cell is sized by `inner_range_fn(outer_element_index)` and immediately
/// filled by `inner_fill_fn(inner_cell&, outer_element_index)` before moving
/// to the next -- no separate all-ranges walk. A zero-volume inner range
/// yields a deliberately-null cell, which `inner_fill_fn` is not invoked on.
/// Cells are zero-initialized, so the default no-op fill still leaves zeroed
/// storage. Backed by `ArenaToTBuilder`.
template <typename OuterTensor, typename InnerRangeFn,
          typename InnerFillFn = nested_fill_noop>
OuterTensor make_nested_tile(
    const typename OuterTensor::range_type& outer_range,
    InnerRangeFn&& inner_range_fn, InnerFillFn&& inner_fill_fn = {}) {
  ArenaToTBuilder<OuterTensor> builder(outer_range, /*batch_sz=*/1,
                                       /*zero_init=*/true);
  const std::size_t N = outer_range.volume();
  for (std::size_t ord = 0; ord < N; ++ord) {
    const auto idx = outer_range.idx(ord);
    auto& cell = builder.emplace(ord, inner_range_fn(idx));
    if (!cell.empty()) inner_fill_fn(cell, idx);
  }
  return std::move(builder).finish();
}

/// Apply a unary fill op while preserving each source inner range.
/// `fill_op(dst_data, src_data, n_elements)` writes the result cell.
template <typename OuterTensor, typename SrcOuterTensor, typename FillOp>
OuterTensor arena_trivial_unary(const SrcOuterTensor& src, FillOp&& fill_op) {
  using elem_t = typename OuterTensor::value_type::value_type;
  using inner_range_t = typename OuterTensor::value_type::range_type;
  // A null inner cell has no range to query (`ArenaTensor::range()` asserts
  // non-null); map it to a default range -> a null result cell.
  auto range_fn = [&src](std::size_t ord) -> inner_range_t {
    const auto& s = src.data()[ord];
    return s.empty() ? inner_range_t{} : s.range();
  };
  // Elementwise kernels pack tight (no cross-cell GEMM to amortize padding);
  // the fill overwrites every element, so the slab need not be zero-init'd.
  OuterTensor result = arena_outer_init<OuterTensor>(src.range(), src.nbatch(),
                                                     range_fn, alignof(elem_t),
                                                     /*zero_init=*/false);
  const std::size_t N_cells = src.range().volume() * src.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& dst = result.data()[ord];
    if (dst.empty()) continue;
    fill_op(dst.data(), src.data()[ord].data(), dst.size());
  }
  return result;
}

/// Coalesce a (possibly multi-page, incrementally built) arena-backed ToT
/// outer tile into a fresh single-page tile: one exact allocation, no page
/// tail waste, inner cells laid out contiguously in outer order. Returns a
/// new tile; `src` is unchanged. A tile already built up-front via
/// `arena_outer_init` is single-page already, so compacting it just
/// deep-copies.
template <typename OuterTensor>
OuterTensor arena_compact(const OuterTensor& src) {
  return arena_trivial_unary<OuterTensor>(
      src,
      [](auto* dst, const auto* s, std::size_t n) { std::copy_n(s, n, dst); });
}

/// Apply a binary fill op using the left operand's inner ranges (asserted
/// equal to the right's per cell). `fill_op(dst, l, r, n_elements)`.
template <typename OuterTensor, typename LeftTensor, typename RightTensor,
          typename FillOp>
OuterTensor arena_trivial_binary(const LeftTensor& left,
                                 const RightTensor& right, FillOp&& fill_op) {
  using elem_t = typename OuterTensor::value_type::value_type;
  using inner_range_t = typename OuterTensor::value_type::range_type;
  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());
  // Union sparsity: a result cell is present if *either* operand cell is.
  // ToT arrays with the same outer shape can still differ in which inner cells
  // are populated within an outer tile (e.g. occ_tile_size>1 aggregates several
  // pairs, some screened to null). A cell present in only one operand is
  // combined against an implicit zero slab below -- correct for the linear ops
  // (add: l+0 / 0+r; subt: l-0 / 0-r) and numerically correct for mult (l*0=0,
  // emitted as an explicit zero tile). Without this, a lone-left cell would
  // read a null right slab (segfault) and a lone-right cell would be silently
  // dropped, losing that addend.
  auto range_fn = [&left, &right](std::size_t ord) -> inner_range_t {
    const auto& l = left.data()[ord];
    if (!l.empty()) return l.range();
    const auto& r = right.data()[ord];
    return r.empty() ? inner_range_t{} : r.range();
  };
  OuterTensor result = arena_outer_init<OuterTensor>(
      left.range(), left.nbatch(), range_fn, alignof(elem_t),
      /*zero_init=*/false);
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  std::vector<elem_t> zeros;  // grown lazily; implicit-zero slab for lone cells
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& dst = result.data()[ord];
    if (dst.empty()) continue;
    const auto& l = left.data()[ord];
    const auto& r = right.data()[ord];
    const std::size_t n = dst.size();
    const bool have_l = !l.empty();
    const bool have_r = !r.empty();
    TA_ASSERT(!have_l || l.size() == n);
    TA_ASSERT(!have_r || r.size() == n);
    if (have_l && have_r) {
      fill_op(dst.data(), l.data(), r.data(), n);
    } else {
      if (zeros.size() < n) zeros.assign(n, elem_t{});
      const elem_t* l_ptr = have_l ? l.data() : zeros.data();
      const elem_t* r_ptr = have_r ? r.data() : zeros.data();
      fill_op(dst.data(), l_ptr, r_ptr, n);
    }
  }
  return result;
}

/// Trivial mixed scalar/ToT outer-Hadamard kernel: `tot_outer` drives the
/// result's outer and per-cell inner ranges; `scalar_outer` supplies one
/// scalar per outer cell. `fill_op(dst, tot_data, scalar_value, n_elements)`.
template <typename OuterTensor, typename ToTSide, typename ScalarSide,
          typename FillOp>
OuterTensor arena_trivial_scaled(const ToTSide& tot_outer,
                                 const ScalarSide& scalar_outer,
                                 FillOp&& fill_op) {
  using elem_t = typename OuterTensor::value_type::value_type;
  using inner_range_t = typename OuterTensor::value_type::range_type;
  TA_ASSERT(tot_outer.range().volume() == scalar_outer.range().volume());
  TA_ASSERT(tot_outer.nbatch() == scalar_outer.nbatch());
  auto range_fn = [&tot_outer](std::size_t ord) -> inner_range_t {
    const auto& t = tot_outer.data()[ord];
    return t.empty() ? inner_range_t{} : t.range();
  };
  OuterTensor result = arena_outer_init<OuterTensor>(
      tot_outer.range(), tot_outer.nbatch(), range_fn, alignof(elem_t),
      /*zero_init=*/false);
  const std::size_t N_cells = tot_outer.range().volume() * tot_outer.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& dst = result.data()[ord];
    if (dst.empty()) continue;
    fill_op(dst.data(), tot_outer.data()[ord].data(), scalar_outer.data()[ord],
            dst.size());
  }
  return result;
}

/// Grow `result` in place so every cell whose current inner cell is null but
/// `more_range_fn(cell_ordinal)` yields a non-empty range becomes an
/// allocated, zero-initialized cell. Data already accumulated in non-empty
/// cells is preserved -- a fresh slab is built and the old cell data copied
/// over. A no-op (no reallocation) when nothing grows, so the steady-state
/// path stays cheap. Used by the SUMMA ToT contraction, which shapes a result
/// tile from its first K-panel only and must extend it for later panels of a
/// contracted-dimension-sparse ToT operand.
template <typename OuterTensor, typename MoreRangeFn>
void arena_tot_grow_inplace(OuterTensor& result, MoreRangeFn&& more_range_fn) {
  using inner_t = typename OuterTensor::value_type;
  using elem_t = typename inner_t::value_type;
  using inner_range_t = typename inner_t::range_type;
  const std::size_t N_cells = result.range().volume() * result.nbatch();
  std::vector<inner_range_t> ranges;
  ranges.reserve(N_cells);
  bool grows = false;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& rc = result.data()[ord];
    if (!rc.empty()) {
      ranges.emplace_back(rc.range());
      continue;
    }
    inner_range_t r = more_range_fn(ord);
    if (r.volume() != 0) grows = true;
    ranges.emplace_back(std::move(r));
  }
  if (!grows) return;
  OuterTensor grown = arena_outer_init<OuterTensor>(
      result.range(), result.nbatch(),
      [&ranges](std::size_t ord) -> inner_range_t { return ranges[ord]; });
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& src = result.data()[ord];
    if (src.empty()) continue;
    auto& dst = grown.data()[ord];
    TA_ASSERT(!dst.empty() && dst.size() == src.size());
    const elem_t* s = src.data();
    elem_t* d = dst.data();
    for (std::size_t i = 0; i < src.size(); ++i) d[i] = s[i];
  }
  result = std::move(grown);
}

/// Accumulate `arg` into `result` (`result += arg`), first growing `result`
/// to the union of the two tiles' inner-cell sparsity. Either tile may be
/// outer-empty. Used to combine two partial contraction results whose
/// disjoint K-panel subsets induced different inner-cell sparsity.
template <typename OuterTensor>
void arena_tot_add_to(OuterTensor& result, const OuterTensor& arg) {
  using inner_t = typename OuterTensor::value_type;
  using elem_t = typename inner_t::value_type;
  using inner_range_t = typename inner_t::range_type;
  if (arg.empty()) return;
  auto arg_range_fn = [&arg](std::size_t ord) -> inner_range_t {
    const auto& a = arg.data()[ord];
    return a.empty() ? inner_range_t{} : a.range();
  };
  if (result.empty()) {
    result =
        arena_outer_init<OuterTensor>(arg.range(), arg.nbatch(), arg_range_fn);
  } else {
    TA_ASSERT(result.range().volume() == arg.range().volume());
    TA_ASSERT(result.nbatch() == arg.nbatch());
    arena_tot_grow_inplace(result, arg_range_fn);
  }
  const std::size_t N_cells = arg.range().volume() * arg.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& src = arg.data()[ord];
    if (src.empty()) continue;
    auto& dst = result.data()[ord];
    TA_ASSERT(!dst.empty() && dst.size() == src.size());
    const elem_t* s = src.data();
    elem_t* d = dst.data();
    for (std::size_t i = 0; i < src.size(); ++i) d[i] += s[i];
  }
}

/// Shallow-permute outer cells while preserving inner storage. The result
/// shares the source's inner storage (arena slab or aliased element data);
/// only the outer-cell array is rebuilt in permuted order.
template <typename OuterTensor, typename SrcOuterTensor, typename Perm>
OuterTensor arena_permute_shallow(const SrcOuterTensor& src, const Perm& perm) {
  using inner_t = typename OuterTensor::value_type;
  TA_ASSERT(perm);
  TA_ASSERT(perm.size() == src.range().rank());
  auto perm_range = perm * src.range();
  const std::size_t N_cells = src.range().volume();
  const std::size_t total_cells = N_cells * src.nbatch();
  auto data = make_outer_data<OuterTensor>(
      total_cells, std::make_shared<Arena>(), src.data_shared());
  OuterTensor result(perm_range, src.nbatch(), std::move(data));
  for (std::size_t s = 0; s < N_cells; ++s) {
    auto src_idx = src.range().idx(s);
    auto tgt_ord = perm_range.ordinal(perm * src_idx);
    for (std::size_t b = 0; b < src.nbatch(); ++b) {
      const std::size_t s_off = b * N_cells + s;
      const std::size_t t_off = b * N_cells + tgt_ord;
      const inner_t& src_inner = src.data()[s_off];
      if constexpr (is_arena_tensor_v<inner_t>) {
        // The view is 8 bytes; copy rebinds it to the same Cell. The source's
        // arena is kept alive by the keep-alive captured in the deleter.
        ::new (result.data() + t_off) inner_t(src_inner);
      } else {
        auto src_inner_data = const_cast<inner_t&>(src_inner).data_shared();
        ::new (result.data() + t_off) inner_t(
            src_inner.range(), src_inner.nbatch(), std::move(src_inner_data));
      }
    }
  }
  return result;
}

/// Permute the inner modes of every cell of a slab-backed ToT outer tile.
///
/// Produces a fresh slab-backed tile with the same outer layout as `src`,
/// but with each inner cell's range and data permuted by `inner_perm`
/// (`result_cell(inner_perm * i) == src_cell(i)`). This is the slab-level
/// counterpart of a per-cell permute: the owning tile allocates one new
/// slab and rewrites every cell, so no view inner cell is ever asked to
/// value-return. `inner_perm` is a plain (non-bipartite) permutation whose
/// rank matches the inner-cell rank.
template <typename OuterTensor, typename SrcOuterTensor, typename Perm>
OuterTensor arena_inner_permute(const SrcOuterTensor& src,
                                const Perm& inner_perm) {
  using inner_t = typename OuterTensor::value_type;
  using elem_t = typename inner_t::value_type;
  using inner_range_t = typename inner_t::range_type;
  TA_ASSERT(inner_perm);
  const std::size_t rank = inner_perm.size();

  // result cell range = inner_perm applied to the src cell range; a null
  // src cell maps to a default (null) range -> a null result cell.
  auto range_fn = [&src, &inner_perm, rank](std::size_t ord) -> inner_range_t {
    const auto& s = src.data()[ord];
    if (s.empty()) return inner_range_t{};
    TA_ASSERT(static_cast<std::size_t>(s.range().rank()) == rank);
    const auto& se = s.range().extent();
    std::vector<std::size_t> ext(rank);
    for (std::size_t d = 0; d < rank; ++d)
      ext[d] = static_cast<std::size_t>(se[d]);
    return inner_range_t(inner_perm * ext);
  };
  // The permute writes every result element exactly once, so no zero-init.
  OuterTensor result = arena_outer_init<OuterTensor>(src.range(), src.nbatch(),
                                                     range_fn, alignof(elem_t),
                                                     /*zero_init=*/false);

  const std::size_t N_cells = src.range().volume() * src.nbatch();
  // Per-cell scratch (rank is fixed across cells); reused, not reallocated.
  std::vector<std::size_t> dstride(rank), w(rank), ctr(rank);
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& dst = result.data()[ord];
    if (dst.empty()) continue;
    const auto& s = src.data()[ord];
    const auto& se = s.range().extent();
    const auto& de = dst.range().extent();
    // row-major strides of the (permuted) destination cell
    dstride[rank - 1] = 1;
    for (std::size_t d = rank - 1; d > 0; --d)
      dstride[d - 1] = dstride[d] * static_cast<std::size_t>(de[d]);
    // w[d] = destination stride contributed by source dimension d, since
    // source dim d maps to destination dim inner_perm[d].
    for (std::size_t d = 0; d < rank; ++d)
      w[d] = dstride[static_cast<std::size_t>(inner_perm[d])];
    // walk the source cell in row-major order, scattering into the dst cell
    ctr.assign(rank, 0);
    const std::size_t vol = s.size();
    const elem_t* sd = s.data();
    elem_t* dd = dst.data();
    for (std::size_t so = 0; so < vol; ++so) {
      std::size_t dofs = 0;
      for (std::size_t d = 0; d < rank; ++d) dofs += w[d] * ctr[d];
      dd[dofs] = sd[so];
      for (std::size_t d = rank; d-- > 0;) {
        if (++ctr[d] < static_cast<std::size_t>(se[d])) break;
        ctr[d] = 0;
      }
    }
  }
  return result;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_ARENA_KERNELS_H__INCLUDED
