/// Arena-backed factory for outer tiles whose inner cells are `ArenaTensor`s.
/// Each non-null inner is a view onto a `Cell` placement-newed into one slab
/// owned by the outer tile. Cell stride and element alignment both respect
/// `kInnerSimdAlign` so element pointers are SIMD-aligned without runtime
/// checks.

#ifndef TILEDARRAY_TENSOR_ARENA_TENSOR_KERNELS_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_TENSOR_KERNELS_H__INCLUDED

#include "TiledArray/config.h"
#include "TiledArray/error.h"
#include "TiledArray/tensor/arena.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/arena_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace TiledArray {
namespace detail {

/// Build an outer tile whose inner cells are arena-pinned `ArenaTensor`s.
///
/// `shape_fn(ord)` is called for each cell ordinal in `[0, outer.volume *
/// batch)` and must return an inner `range_type` (or value convertible to it).
/// A zero-volume range produces a null inner (`cell_ == nullptr`); no slab
/// bytes are consumed for null cells.
///
/// `cell_stride_align` is the minimum byte stride between adjacent cell
/// allocations in the slab. It defaults to `kArenaCachelineAlign` (false-
/// sharing floor) and is bumped up to `InnerT::cell_alignment()` if that
/// is larger (i.e. the SIMD-aware element alignment).
template <typename OuterTensor, typename OuterRange, typename ShapeFn>
OuterTensor arena_outer_init_pinned(
    const OuterRange& outer_range, std::size_t batch_sz, ShapeFn&& shape_fn,
    std::size_t cell_stride_align = kArenaCachelineAlign) {
  using InnerT = typename OuterTensor::value_type;
  using T = typename InnerT::value_type;
  using R = typename InnerT::range_type;

  const std::size_t stride = cell_stride_align > InnerT::cell_alignment()
                                 ? cell_stride_align
                                 : InnerT::cell_alignment();

  const std::size_t N_cells = outer_range.volume() * batch_sz;
  constexpr std::size_t kNullSentinel = static_cast<std::size_t>(-1);

  std::vector<R> ranges;
  ranges.reserve(N_cells);
  std::vector<std::size_t> offsets(N_cells, 0);
  std::size_t total = 0;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    ranges.emplace_back(shape_fn(ord));
    const std::size_t vol = ranges.back().volume();
    if (vol == 0) {
      offsets[ord] = kNullSentinel;
    } else {
      offsets[ord] = total;
      total += arena_align_up(InnerT::cell_size(vol), stride);
    }
  }

  auto arena = std::make_shared<Arena>();
  // Slab base must be at least cell-aligned so interior offsets land
  // exactly on the SIMD boundary defined by InnerT::data_alignment().
  if (total > 0) arena->reserve(total, /*zero_init=*/true, stride);
  auto data =
      make_outer_data<OuterTensor>(N_cells, arena, std::shared_ptr<InnerT[]>{});
  OuterTensor result(outer_range, batch_sz, std::move(data));

  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    if (offsets[ord] == kNullSentinel) {
      ::new (result.data() + ord) InnerT();
    } else {
      // slice<std::byte>(offset, n) returns an aliased shared_ptr; we just
      // need its raw pointer to placement-new the Cell. The slab's lifetime
      // is held by `arena_handle` captured in the outer's deleter.
      auto byte_view = arena->template slice<std::byte>(offsets[ord], 1);
      InnerT inner =
          make_arena_tensor_in<T>(byte_view.get(), std::move(ranges[ord]));
      ::new (result.data() + ord) InnerT(inner);
    }
  }
  return result;
}

/// Trivial unary kernel for `TA::Tensor<ArenaTensor<...>>`: plan from the
/// source's inner shapes, allocate one slab, fill each non-null result cell
/// from the matching source via `fill_op(dst, src, n)`.
template <typename OuterTensor, typename SrcOuterTensor, typename FillOp>
OuterTensor arena_trivial_unary_pinned(const SrcOuterTensor& src,
                                       FillOp&& fill_op) {
  using InnerT = typename OuterTensor::value_type;
  using SrcInnerT = typename SrcOuterTensor::value_type;
  static_assert(std::is_same_v<typename InnerT::range_type,
                               typename SrcInnerT::range_type>,
                "arena_trivial_unary_pinned: result and source inner range "
                "types must match");
  using R = typename InnerT::range_type;

  auto shape_fn = [&src](std::size_t ord) -> R {
    const auto& sinner = src.data()[ord];
    return sinner ? sinner.range() : R();
  };
  OuterTensor result =
      arena_outer_init_pinned<OuterTensor>(src.range(), src.nbatch(), shape_fn);

  const std::size_t N_cells = src.range().volume() * src.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& sinner = src.data()[ord];
    auto& dinner = result.data()[ord];
    if (!sinner) continue;
    TA_ASSERT(bool(dinner));
    TA_ASSERT(sinner.size() == dinner.size());
    fill_op(dinner.data(), sinner.data(), dinner.size());
  }
  return result;
}

/// Trivial binary kernel for `TA::Tensor<ArenaTensor<...>>`: plan from the
/// left operand's inner shapes (asserted equal to right's per cell), one
/// slab, fill each non-null result cell via `fill_op(dst, l, r, n)`.
template <typename OuterTensor, typename LeftTensor, typename RightTensor,
          typename FillOp>
OuterTensor arena_trivial_binary_pinned(const LeftTensor& left,
                                        const RightTensor& right,
                                        FillOp&& fill_op) {
  using InnerT = typename OuterTensor::value_type;
  using LeftInnerT = typename LeftTensor::value_type;
  using RightInnerT = typename RightTensor::value_type;
  static_assert(std::is_same_v<typename InnerT::range_type,
                               typename LeftInnerT::range_type>,
                "arena_trivial_binary_pinned: result and left inner range "
                "types must match");
  static_assert(std::is_same_v<typename InnerT::range_type,
                               typename RightInnerT::range_type>,
                "arena_trivial_binary_pinned: result and right inner range "
                "types must match");
  using R = typename InnerT::range_type;

  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());

  auto shape_fn = [&left](std::size_t ord) -> R {
    const auto& l = left.data()[ord];
    return l ? l.range() : R();
  };
  OuterTensor result = arena_outer_init_pinned<OuterTensor>(
      left.range(), left.nbatch(), shape_fn);

  const std::size_t N_cells = left.range().volume() * left.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& l = left.data()[ord];
    const auto& r = right.data()[ord];
    auto& d = result.data()[ord];
    if (!d) continue;
    TA_ASSERT(bool(l) && bool(r));
    TA_ASSERT(l.size() == r.size());
    TA_ASSERT(l.size() == d.size());
    fill_op(d.data(), l.data(), r.data(), d.size());
  }
  return result;
}

/// Trivial mixed scalar/ArenaTensor outer-Hadamard kernel for
/// `TA::Tensor<scalar> · TA::Tensor<ArenaTensor>` and the symmetric form.
/// `arena_outer` drives the result's outer shape and per-cell inner
/// shapes; `scalar_outer` supplies one scalar per outer cell. `fill_op`
/// is invoked per non-null cell with
/// `(dst, arena_data, scalar_value, n_elements)` and writes the result.
template <typename OuterTensor, typename ArenaSide, typename ScalarSide,
          typename FillOp>
OuterTensor arena_trivial_scaled_pinned(const ArenaSide& arena_outer,
                                        const ScalarSide& scalar_outer,
                                        FillOp&& fill_op) {
  using InnerT = typename OuterTensor::value_type;
  using SrcInnerT = typename ArenaSide::value_type;
  static_assert(std::is_same_v<typename InnerT::range_type,
                               typename SrcInnerT::range_type>,
                "arena_trivial_scaled_pinned: result and arena-side inner "
                "range types must match");
  using R = typename InnerT::range_type;
  using ScalarT = typename ScalarSide::value_type;

  TA_ASSERT(arena_outer.range().volume() == scalar_outer.range().volume());
  TA_ASSERT(arena_outer.nbatch() == scalar_outer.nbatch());

  auto shape_fn = [&arena_outer](std::size_t ord) -> R {
    const auto& inner = arena_outer.data()[ord];
    return inner ? inner.range() : R();
  };
  OuterTensor result = arena_outer_init_pinned<OuterTensor>(
      arena_outer.range(), arena_outer.nbatch(), shape_fn);

  const std::size_t N_cells =
      arena_outer.range().volume() * arena_outer.nbatch();
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& a = arena_outer.data()[ord];
    const ScalarT& s = scalar_outer.data()[ord];
    auto& d = result.data()[ord];
    if (!d) continue;
    TA_ASSERT(bool(a));
    TA_ASSERT(a.size() == d.size());
    fill_op(d.data(), a.data(), s, d.size());
  }
  return result;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_ARENA_TENSOR_KERNELS_H__INCLUDED
