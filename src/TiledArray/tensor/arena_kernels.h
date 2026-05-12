/// Arena kernels for ToT trivial ops and contraction result initialization.

#ifndef TILEDARRAY_TENSOR_ARENA_KERNELS_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_KERNELS_H__INCLUDED

#include "TiledArray/config.h"
#include "TiledArray/error.h"
#include "TiledArray/tensor/arena.h"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace TiledArray {
namespace detail {

namespace {

/// Build outer storage whose deleter owns arena and alias keep-alive state.
template <typename OuterTensor, typename KeepAlive>
std::shared_ptr<typename OuterTensor::value_type[]>
make_outer_data(std::size_t n_cells, std::shared_ptr<Arena> arena_handle,
                KeepAlive keep_alive) {
  using inner_t = typename OuterTensor::value_type;
  std::allocator<inner_t> allocator;
  inner_t* raw = allocator.allocate(n_cells);
  auto deleter = [allocator = std::move(allocator), arena_handle = std::move(arena_handle),
                  keep_alive = std::move(keep_alive), n_cells](inner_t* p) mutable {
    for (std::size_t i = 0; i < n_cells; ++i) (p + i)->~inner_t();
    allocator.deallocate(p, n_cells);
    (void)arena_handle;
    (void)keep_alive;
  };
  return std::shared_ptr<inner_t[]>(raw, std::move(deleter));
}

}

/// Apply a unary fill op while preserving each source inner shape.
template <typename OuterTensor, typename SrcOuterTensor, typename FillOp>
OuterTensor arena_trivial_unary(const SrcOuterTensor& src, FillOp&& fill_op) {
  using inner_t = typename OuterTensor::value_type;
  using elem_t = typename inner_t::value_type;
  const std::size_t N_cells = src.range().volume() * src.nbatch();
  auto shape_fn = [&src](std::size_t ord) -> decltype(auto) {
    return src.data()[ord].range();
  };
  ArenaPlan p = plan(N_cells, shape_fn, sizeof(elem_t), alignof(elem_t));
  auto arena = std::make_shared<Arena>();
  if (p.total_bytes > 0) arena->reserve(p.total_bytes, false);
  auto data =
      make_outer_data<OuterTensor>(N_cells, arena, std::shared_ptr<inner_t[]>{});
  OuterTensor result(src.range(), src.nbatch(), std::move(data));
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& r = src.data()[ord].range();
    const std::size_t n = r.volume();
    if (n == 0) {
      new (result.data() + ord) inner_t(r);
      continue;
    }
    auto elem_data = arena->slice<elem_t>(p.offsets[ord], n);
    new (result.data() + ord) inner_t(r, std::move(elem_data));
    fill_op(result.data()[ord].data(), src.data()[ord].data(), n);
  }
  return result;
}

/// Apply a binary fill op using the left operand's inner shapes.
template <typename OuterTensor, typename LeftTensor, typename RightTensor,
          typename FillOp>
OuterTensor arena_trivial_binary(const LeftTensor& left, const RightTensor& right,
                                FillOp&& fill_op) {
  using inner_t = typename OuterTensor::value_type;
  using elem_t = typename inner_t::value_type;
  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  auto shape_fn = [&left](std::size_t ord) -> decltype(auto) {
    return left.data()[ord].range();
  };
  ArenaPlan p = plan(N_cells, shape_fn, sizeof(elem_t), alignof(elem_t));
  auto arena = std::make_shared<Arena>();
  if (p.total_bytes > 0) arena->reserve(p.total_bytes, false);
  auto data =
      make_outer_data<OuterTensor>(N_cells, arena, std::shared_ptr<inner_t[]>{});
  OuterTensor result(left.range(), left.nbatch(), std::move(data));
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    const auto& r = left.data()[ord].range();
    const std::size_t n = r.volume();
    TA_ASSERT(n == right.data()[ord].range().volume());
    if (n == 0) {
      new (result.data() + ord) inner_t(r);
      continue;
    }
    auto elem_data = arena->slice<elem_t>(p.offsets[ord], n);
    new (result.data() + ord) inner_t(r, std::move(elem_data));
    fill_op(result.data()[ord].data(), left.data()[ord].data(),
            right.data()[ord].data(), n);
  }
  return result;
}

/// Shallow-permute outer cells while preserving inner data aliases.
template <typename OuterTensor, typename SrcOuterTensor, typename Perm>
OuterTensor arena_permute_shallow(const SrcOuterTensor& src, const Perm& perm) {
  using inner_t = typename OuterTensor::value_type;
  TA_ASSERT(perm);
  TA_ASSERT(perm.size() == src.range().rank());
  auto perm_range = perm * src.range();
  const std::size_t N_cells = src.range().volume();
  const std::size_t total_cells = N_cells * src.nbatch();
  const auto src_data_ref = src.data_shared();
  auto data =
      make_outer_data<OuterTensor>(total_cells,
                                   std::make_shared<Arena>(),
                                   std::move(src_data_ref));
  OuterTensor result(perm_range, src.nbatch(), std::move(data));
  for (std::size_t s = 0; s < N_cells; ++s) {
    auto src_idx = src.range().idx(s);
    auto tgt_ord = perm_range.ordinal(perm * src_idx);
    for (std::size_t b = 0; b < src.nbatch(); ++b) {
      const std::size_t s_off = b * N_cells + s;
      const std::size_t t_off = b * N_cells + tgt_ord;
      const inner_t& src_inner = src.data()[s_off];
      auto src_inner_data = const_cast<inner_t&>(src_inner).data_shared();
      new (result.data() + t_off) inner_t(src_inner.range(), src_inner.nbatch(),
                                          std::move(src_inner_data));
    }
  }
  return result;
}

/// Allocate a slab-backed outer tile using caller-provided inner shapes.
/// `alignment` is the per-cell stride alignment (e.g. kArenaCachelineAlign).
template <typename OuterTensor, typename Range, typename ShapeFn>
OuterTensor arena_outer_init(const Range& outer_range, std::size_t batch_sz,
                             ShapeFn&& shape_fn,
                             std::size_t alignment = kArenaCachelineAlign,
                             bool zero_init = true) {
  using inner_t = typename OuterTensor::value_type;
  using elem_t = typename inner_t::value_type;
  using inner_range_t =
      std::decay_t<decltype(shape_fn(std::declval<std::size_t>()))>;
  TA_ASSERT(alignment >= alignof(elem_t));
  const std::size_t N_cells = outer_range.volume() * batch_sz;
  std::vector<inner_range_t> ranges;
  ranges.reserve(N_cells);
  std::vector<std::size_t> offsets(N_cells);
  std::size_t total_bytes = 0;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    offsets[ord] = total_bytes;
    ranges.emplace_back(shape_fn(ord));
    const std::size_t bytes = ranges.back().volume() * sizeof(elem_t);
    total_bytes += arena_align_up(bytes, alignment);
  }
  auto arena = std::make_shared<Arena>();
  // Arena::reserve requires a non-empty slab.
  if (total_bytes > 0) {
    arena->reserve(total_bytes, zero_init);
  }
  auto data =
      make_outer_data<OuterTensor>(N_cells, arena, std::shared_ptr<inner_t[]>{});
  OuterTensor result(outer_range, batch_sz, std::move(data));
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& r = ranges[ord];
    const std::size_t n = r.volume();
    if (n == 0) {
      // Rank-0 empties must preserve Tensor's null-data/no-range invariant.
      if (!r) {
        new (result.data() + ord) inner_t();
      } else {
        new (result.data() + ord) inner_t(r);
      }
    } else {
      auto elem_data = arena->slice<elem_t>(offsets[ord], n);
      new (result.data() + ord) inner_t(r, std::move(elem_data));
    }
  }
  return result;
}

}
}

#endif
