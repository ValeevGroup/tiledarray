/// In-rank data movers for the in-SUMMA two-trange contraction retile.
///
/// Two helpers on arena-backed Tensor-of-Tensor (ToT) outer tiles
/// (`TA::Tensor<TA::ArenaTensor<T>>`):
/// - `arena_gather_block`: physically pack a contiguous block of
///     fine arena ToT tiles into ONE single-page coarse tile. Always valid;
///     constant-stride (strided-DGEMM-eligible) iff the gathered cells along
///     the strided (innermost) axis share inner extent. A non-uniform block is
///     still gathered correctly (right values, single page) -- it just won't be
///     classified strided-eligible, so the kernel falls back per-cell.
/// - `arena_carve_block`: carve a coarse tile back into fine
///     sub-tiles. `view==true` (the default, free direction) hands back
///     zero-copy sub-views sharing the coarse tile's arena storage;
///     `view==false` returns owning single-page copies via `arena_outer_init`.
///
/// Roundtrip invariant: `arena_carve_block(arena_gather_block(fine, ...),
/// fine_ranges)` reproduces each original fine tile's values.

#ifndef TILEDARRAY_TENSOR_ARENA_RETILE_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_RETILE_H__INCLUDED

#include "TiledArray/error.h"
#include "TiledArray/range.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/arena_tensor.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace TiledArray {
namespace detail {

/// Locate, among a row-major block of fine outer tiles, the (tile, local outer
/// ordinal) owning outer element index `idx`. Returns `{tile_index,
/// local_ordinal}`; `tile_index == fine.size()` signals "no fine tile contains
/// `idx`" (a hole -- a deliberately-absent cell). `idx` is the outer element
/// index within the coarse block's coordinate system.
template <typename ArenaOuter, typename Index>
inline std::pair<std::size_t, std::size_t> arena_locate_fine_(
    const std::vector<ArenaOuter>& fine, const Index& idx) {
  for (std::size_t f = 0; f < fine.size(); ++f) {
    const auto& fr = fine[f].range();
    if (fr.includes(idx)) return {f, fr.ordinal(idx)};
  }
  return {fine.size(), 0};
}

/// Gather a contiguous block of fine arena ToT tiles into ONE single-page
/// coarse tile. `fine` are the source outer tiles partitioning the
/// coarse outer block (their outer ranges are sub-boxes of `coarse_outer`,
/// expressed in the same coordinate system); `coarse_outer` is the result outer
/// range; `nbatch` the batch size (shared by every fine tile and the result).
///
/// The coarse tile is laid down up-front as one exact arena page via
/// `arena_outer_init`: each coarse cell takes its inner range -- and then its
/// values -- from the fine cell at the same outer index and batch. A coarse
/// outer position not covered by any fine tile yields a null cell. Always
/// produces a single-page tile; it is constant-stride (strided-eligible) iff
/// the gathered cells along the strided axis share inner extent, which the
/// SUMMA kernel's stride-run classifier decides downstream.
template <typename ArenaOuter>
ArenaOuter arena_gather_block(const std::vector<ArenaOuter>& fine,
                              const Range& coarse_outer, std::size_t nbatch) {
  using inner_t = typename ArenaOuter::value_type;
  using inner_range_t = typename inner_t::range_type;
  using elem_t = typename inner_t::value_type;
  static_assert(is_arena_tensor_v<inner_t>,
                "arena_gather_block requires an arena-backed ToT outer tile");
  TA_ASSERT(nbatch >= 1);

  const std::size_t outer_vol = coarse_outer.volume();

  // Inner range of coarse cell `ord` == inner range of the fine cell at the
  // same outer index and batch (empty if no fine tile covers that position).
  auto range_fn = [&](std::size_t ord) -> inner_range_t {
    const std::size_t b = ord / outer_vol;
    const std::size_t p = ord % outer_vol;
    auto idx = coarse_outer.idx(p);
    auto [f, lord] = arena_locate_fine_<ArenaOuter>(fine, idx);
    if (f == fine.size()) return inner_range_t{};
    const auto& s = fine[f].data()[b * fine[f].range().volume() + lord];
    return s.empty() ? inner_range_t{} : s.range();
  };

  ArenaOuter result = arena_outer_init<ArenaOuter>(
      coarse_outer, nbatch, range_fn, alignof(elem_t), /*zero_init=*/false);

  const std::size_t N_cells = outer_vol * nbatch;
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    auto& dst = result.data()[ord];
    if (dst.empty()) continue;
    const std::size_t b = ord / outer_vol;
    const std::size_t p = ord % outer_vol;
    auto idx = coarse_outer.idx(p);
    auto [f, lord] = arena_locate_fine_<ArenaOuter>(fine, idx);
    TA_ASSERT(f != fine.size());
    const auto& s = fine[f].data()[b * fine[f].range().volume() + lord];
    TA_ASSERT(!s.empty() && s.size() == dst.size());
    std::copy_n(s.data(), s.size(), dst.data());
  }
  return result;
}

/// Carve a coarse arena ToT tile into fine sub-tiles, one per range in
/// `fine_outer`. Each `fine_outer[f]` is a sub-box of `coarse.range()`
/// in the same coordinate system.
///
/// - `view == true` (default, the free direction): each fine sub-tile is a
///    zero-copy outer view -- its inner cells are shallow copies of the coarse
///    tile's `ArenaTensor` views (one pointer each), so they alias the coarse
///    arena slab. The coarse tile's storage is kept alive by the returned
///    sub-tiles' outer-data deleter.
///  - `view == false`: each fine sub-tile is an owning single-page copy built
///    by `arena_gather_block` from the carved-out view, independent of the
///    coarse tile's storage.
template <typename ArenaOuter>
std::vector<ArenaOuter> arena_carve_block(const ArenaOuter& coarse,
                                          const std::vector<Range>& fine_outer,
                                          bool view = true) {
  using inner_t = typename ArenaOuter::value_type;
  static_assert(is_arena_tensor_v<inner_t>,
                "arena_carve_block requires an arena-backed ToT outer tile");
  const std::size_t nbatch = coarse.nbatch();
  const auto& crange = coarse.range();

  std::vector<ArenaOuter> out;
  out.reserve(fine_outer.size());

  for (const auto& fr : fine_outer) {
    const std::size_t fvol = fr.volume();
    const std::size_t n_cells = fvol * nbatch;
    // Build a zero-copy outer view tile: a fresh ArenaTensor[] holding shallow
    // copies of the coarse cells, with the coarse tile kept alive in the
    // deleter (a shallow copy of `coarse` pins its arena slab).
    std::allocator<inner_t> alloc;
    inner_t* raw = alloc.allocate(n_cells);
    auto deleter = [alloc, n_cells, keep_alive = coarse](inner_t* p) mutable {
      for (std::size_t i = 0; i < n_cells; ++i) (p + i)->~inner_t();
      alloc.deallocate(p, n_cells);
      (void)keep_alive;
    };
    std::shared_ptr<inner_t[]> data(raw, std::move(deleter));
    for (std::size_t b = 0; b < nbatch; ++b) {
      for (std::size_t p = 0; p < fvol; ++p) {
        auto idx = fr.idx(p);
        TA_ASSERT(crange.includes(idx));
        const std::size_t cord = b * crange.volume() + crange.ordinal(idx);
        ::new (data.get() + b * fvol + p) inner_t(coarse.data()[cord]);
      }
    }
    ArenaOuter sub(fr, nbatch, std::move(data));
    if (view) {
      out.emplace_back(std::move(sub));
    } else {
      // Owning copy: gather the single-tile view into one fresh single-page
      // tile, independent of the coarse storage.
      std::vector<ArenaOuter> one{std::move(sub)};
      out.emplace_back(arena_gather_block<ArenaOuter>(one, fr, nbatch));
    }
  }
  return out;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_ARENA_RETILE_H__INCLUDED
