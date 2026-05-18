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

#include <cstddef>
#include <memory>
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

/// Allocate a slab-backed ToT outer tile with caller-provided inner ranges.
///
/// `inner_range_fn(cell_ordinal)` -> inner `range_type` for each cell ordinal
/// in `[0, outer_range.volume() * batch_sz)`; a zero-volume range yields a
/// deliberately-null inner cell that consumes no slab bytes. Element storage
/// is left zero-initialized when `zero_init` is true. `cell_stride_align` is
/// the minimum byte stride between adjacent cells; it is bumped up to the
/// inner type's natural alignment (`ArenaTensor::cell_alignment()`, or
/// `alignof(T)` for `TA::Tensor` inners).
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
  // Cells pack at `stride` granularity, but the slab base handed to
  // `Arena::reserve` must be at least `max_align_t`-aligned.
  const std::size_t slab_align =
      stride > alignof(std::max_align_t) ? stride : alignof(std::max_align_t);

  const std::size_t N_cells = outer_range.volume() * batch_sz;
  constexpr std::size_t kNull = static_cast<std::size_t>(-1);
  std::vector<InnerRange> ranges;
  ranges.reserve(N_cells);
  std::vector<std::size_t> offsets(N_cells, 0);
  std::size_t total = 0;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    ranges.emplace_back(inner_range_fn(ord));
    const std::size_t vol = ranges.back().volume();
    if (vol == 0) {
      offsets[ord] = kNull;
    } else {
      offsets[ord] = total;
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

  auto arena_slab = std::make_shared<Arena>();
  if (total > 0) arena_slab->reserve(total, zero_init, slab_align);
  auto data = make_outer_data<OuterTensor>(N_cells, arena_slab,
                                           std::shared_ptr<InnerT[]>{});
  OuterTensor result(outer_range, batch_sz, std::move(data));

  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& r = ranges[ord];
    if (offsets[ord] == kNull) {
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
      // slice<std::byte>(offset, 1) returns an aliased shared_ptr; we only
      // need its raw pointer to placement-new the Cell -- the slab's lifetime
      // is held by `arena_handle` captured in the outer's deleter.
      auto byte_view = arena_slab->template slice<std::byte>(offsets[ord], 1);
      ::new (result.data() + ord)
          InnerT(make_arena_tensor_in<T>(byte_view.get(), std::move(r)));
    } else {
      auto elem_data = arena_slab->template slice<T>(offsets[ord], r.volume());
      ::new (result.data() + ord) InnerT(r, std::move(elem_data));
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

/// Build one ToT outer tile over `outer_range`, two-pass:
///   pass 1: `inner_range_fn(outer_element_index)` -> inner `range_type`
///           sizes every inner cell (zero-volume -> deliberately-null cell);
///   pass 2: `inner_fill_fn(inner_cell&, outer_element_index)` fills each
///           non-null cell. The default fill leaves storage zero-initialized.
/// Dispatches internally on the inner-tile type (see `arena_outer_init`).
template <typename OuterTensor, typename InnerRangeFn,
          typename InnerFillFn = nested_fill_noop>
OuterTensor make_nested_tile(
    const typename OuterTensor::range_type& outer_range,
    InnerRangeFn&& inner_range_fn, InnerFillFn&& inner_fill_fn = {}) {
  // arena_outer_init keys ranges on the cell ordinal; user code keys on the
  // (global) outer element index -- translate via the outer range.
  auto cell_range_fn = [&](std::size_t ord) {
    return inner_range_fn(outer_range.idx(ord));
  };
  OuterTensor result =
      arena_outer_init<OuterTensor>(outer_range, 1, cell_range_fn);
  const std::size_t N = outer_range.volume();
  for (std::size_t ord = 0; ord < N; ++ord) {
    auto& cell = result.data()[ord];
    if (!cell.empty()) inner_fill_fn(cell, outer_range.idx(ord));
  }
  return result;
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
  auto range_fn = [&left](std::size_t ord) -> inner_range_t {
    const auto& l = left.data()[ord];
    return l.empty() ? inner_range_t{} : l.range();
  };
  OuterTensor result = arena_outer_init<OuterTensor>(
      left.range(), left.nbatch(), range_fn, alignof(elem_t),
      /*zero_init=*/false);
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& dst = result.data()[ord];
    if (dst.empty()) continue;
    TA_ASSERT(left.data()[ord].size() == right.data()[ord].size());
    TA_ASSERT(left.data()[ord].size() == dst.size());
    fill_op(dst.data(), left.data()[ord].data(), right.data()[ord].data(),
            dst.size());
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
