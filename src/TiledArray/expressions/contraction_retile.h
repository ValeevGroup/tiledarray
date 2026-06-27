/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  contraction_retile.h
 *  Pure per-axis nesting plan + T-grid -> U-result ordinal map.
 *
 *  This header is PURE: it derives, for a contraction, how a chosen *target*
 *  tiling T nests against the *user* tiling U on each role axis (Hadamard,
 *  SUMMA-M, SUMMA-N, SUMMA-K), classifies which axes are the strided BLAS
 *  axes (exactly as the engine's strided-install gates classify ce_ce vs
 *  ce_e), and maps a coarse T result-grid ordinal to the U-result-trange
 *  ordinals it covers. No MADWorld / MPI; unit-testable in isolation.
 */

#ifndef TILEDARRAY_EXPRESSIONS_CONTRACTION_RETILE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_CONTRACTION_RETILE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/tiled_range1.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace TiledArray::expressions {

/// How a target tiling T nests against the user tiling U on one axis.
enum class NestDir { Identity, Coarsen, Refine };

/// Per-axis nesting of T vs U.
///
/// \c groups is indexed by the *target* (T) tile and maps each T tile to the
/// half-open range [first,last) into the *U* tiling that it covers. For Identity
/// the two tilings coincide and the map is the trivial diagonal
/// (groups[t] == [t,t+1)). For Coarsen U is the finer tiling
/// (\c u_is_finer == true), so each T tile spans several U tiles and groups[t]
/// is a multi-U-tile range. For Refine T is the finer tiling
/// (\c u_is_finer == false), so several T tiles fall inside one U tile and
/// groups[t] == [u,u+1) for the single containing U tile.
struct AxisNest {
  NestDir dir = NestDir::Identity;
  std::vector<std::pair<std::size_t, std::size_t>> groups;
  bool u_is_finer = false;  ///< true on Coarsen
};

struct RetilePlan {
  /// false => all axes Identity => no retile needed (stock path).
  bool active = false;
  /// Per role axis.
  std::vector<AxisNest> hadamard, summaM, summaN, summaK;
  /// Which axes are the strided BLAS axes (pack targets). Classified exactly as
  /// the engine's strided-install gates, which read the INNER contract-reduce
  /// helper (ce_e vs ce_ce -- both off
  /// `contrreduce_op.gemm_helper()`, the inner tile-GEMM helper): ce_e (inner
  /// outer-product, inner num_contract_ranks()==0) rides the contracted index
  /// into BLAS K (\c k_is_blas_k); ce_ce (inner contraction, inner
  /// num_contract_ranks()>=1) rides a SUMMA external into BLAS M
  /// (\c ride_on_M). The classification is index-structure-only (invariant
  /// under retiling).
  bool ride_on_M = false, ride_on_N = false, k_is_blas_k = false;

  /// Map a coarse T result-grid-cell ordinal (within a SUMMA-K slab) to the
  /// U result-trange ordinals it covers. The result grid is laid out
  /// row-major over the result axes (Hadamard, then SUMMA-M, then SUMMA-N).
  /// Pure ordinal math.
  ///
  /// Per axis: Identity/Coarsen -- the T cell covers the group's [first,last)
  /// U tiles (>= 1); Refine on a result axis -- the U tile is covered by
  /// several T cells, so a single T cell maps to the one U tile that contains
  /// it (size 1; the merge happens at finalize). The Cartesian product over
  /// result axes is emitted in U-result row-major order.
  std::vector<std::size_t> u_result_ordinals(
      std::size_t t_grid_ordinal_in_slab) const;

  /// Result axes, in U-result row-major order (Hadamard ++ M ++ N). Used by
  /// u_result_ordinals; populated by make_retile_plan.
  std::vector<AxisNest> result_axes;
  /// Per result axis: number of T tiles (the T grid extent on that axis).
  std::vector<std::size_t> result_t_extent;
  /// Per result axis: number of U tiles (the U-result trange extent).
  std::vector<std::size_t> result_u_extent;

  /// Per-role TARGET (T) tilings, in role-axis order. Populated by
  /// make_retile_plan from the .retile() targets (an empty role keeps the
  /// vector empty, meaning "T == U" on every axis of that role). The refine
  /// path needs the T tile *element* boundaries (not just the T->U ordinal
  /// groups) to carve a U operand/result tile down to the exact T sub-box;
  /// these carry that information. On a Coarsen/Identity axis the engine never
  /// consults these (the gather box is the union of covered U tiles).
  std::vector<TiledRange1> targetH, targetM, targetN, targetK;
};

namespace detail {

/// Boundary set (hashmarks) of a TiledRange1 -- the element coordinates at
/// which tiles begin/end: { lo, b1, b2, ..., hi }.
inline std::vector<std::size_t> retile_hashmarks(const TiledRange1& tr) {
  std::vector<std::size_t> hm;
  const auto src = tr.hashmarks();
  hm.reserve(src.size());
  for (auto x : src) hm.push_back(static_cast<std::size_t>(x));
  return hm;
}

/// Classify how target tiling \c T nests against user tiling \c U on one axis,
/// and build the T-tile -> U-tile group map.
///
/// \c groups is indexed by the *target* (T) tile and maps each T tile to the
/// half-open range [first,last) of U tiles it covers (always a partition of U
/// by the T boundaries):
///   Identity (T == U):  groups[t] == [t, t+1)        (the diagonal).
///   Coarsen  (T coarser, U finer):  each T tile spans several U tiles, so
///                       groups[t] covers all U tiles under it (u_is_finer).
///   Refine   (T finer,   U coarser): several T tiles fall inside one U tile,
///                       so groups[t] == [u, u+1) for the containing U tile.
///
/// Nesting requires that the coarser tiling's boundaries be a subset of the
/// finer tiling's boundaries; a straddling boundary is a non-nesting error and
/// throws TA::Exception. Equal tile counts with different boundaries is also a
/// non-nesting error.
inline AxisNest retile_axis_nest(const TiledRange1& U, const TiledRange1& T) {
  const std::vector<std::size_t> uhm = retile_hashmarks(U);  // U boundaries
  const std::vector<std::size_t> thm = retile_hashmarks(T);  // T boundaries

  AxisNest ax;

  // Direction by tile count. The coarser tiling has fewer boundaries; its
  // boundary set must be a subset of the finer's.
  if (thm.size() == uhm.size()) {
    if (uhm != thm)
      TA_EXCEPTION(
          "contraction_retile: target and user tilings have equal tile counts "
          "but different tile boundaries (non-nesting)");
    ax.dir = NestDir::Identity;
  } else if (thm.size() < uhm.size()) {
    ax.dir = NestDir::Coarsen;  // T coarser => U finer
  } else {
    ax.dir = NestDir::Refine;  // T finer
  }
  ax.u_is_finer = (ax.dir == NestDir::Coarsen);

  // Subset (nesting) check: the COARSER tiling's boundary set must be a subset
  // of the FINER tiling's. Pick coarse/fine and verify every coarse boundary
  // appears in the finer set (linear merge; both are sorted).
  const std::vector<std::size_t>& coarse =
      (thm.size() <= uhm.size()) ? thm : uhm;
  const std::vector<std::size_t>& fine =
      (thm.size() <= uhm.size()) ? uhm : thm;
  {
    std::size_t fj = 0;
    for (std::size_t c : coarse) {
      while (fj < fine.size() && fine[fj] < c) ++fj;
      if (fj >= fine.size() || fine[fj] != c)
        TA_EXCEPTION(
            "contraction_retile: tile boundary straddles a finer tile boundary "
            "(non-nesting tilings)");
    }
  }

  // Build groups[t] = [first U tile, last U tile) covered by T tile t.
  // groups is always indexed by the T (target) tiling and references U tiles.
  //   - U boundaries are { uhm[0], uhm[1], ... }; U tile i = [uhm[i], uhm[i+1]).
  //   - first U tile covered by T tile t = the U tile whose [lo,hi) contains
  //     thm[t]; last (half-open) = the U tile index where uhm == thm[t+1], or
  //     (when T is finer) the same U tile that contains thm[t+1)-epsilon + 1.
  const std::size_t n_t_tiles = thm.empty() ? 0 : thm.size() - 1;
  ax.groups.reserve(n_t_tiles);

  std::size_t ui = 0;  // first U tile (its lower bound is uhm[ui])
  for (std::size_t ti = 0; ti < n_t_tiles; ++ti) {
    const std::size_t t_lo = thm[ti], t_hi = thm[ti + 1];

    // Advance ui so that U tile ui contains t_lo: uhm[ui] <= t_lo < uhm[ui+1].
    while (ui + 1 < uhm.size() && uhm[ui + 1] <= t_lo) ++ui;
    const std::size_t first_u = ui;

    // Find the U tile whose half-open end == t_hi, or the U tile containing
    // t_hi - 1 (Refine: several T tiles in one U tile, t_hi may be interior to
    // U tile uj). last_u is half-open over U tile indices.
    std::size_t uj = ui;
    while (uj + 1 < uhm.size() && uhm[uj + 1] < t_hi) ++uj;
    // uj is now the last U tile that starts before t_hi; it is fully covered
    // iff uhm[uj+1] == t_hi (T coarser/identity) else t_hi is interior to it
    // (Refine), and that single U tile is the only one this T tile touches.
    const std::size_t last_u = uj + 1;  // half-open

    ax.groups.emplace_back(first_u, last_u);

    // The next T tile starts at the next U boundary only when this T boundary
    // coincided with a U boundary; otherwise (Refine) it stays within U tile uj.
    ui = (uj + 1 < uhm.size() && uhm[uj + 1] == t_hi) ? (uj + 1) : uj;
  }
  return ax;
}

/// Build the per-role AxisNest vector. \c targets empty => every U axis is
/// Identity against itself.
inline std::vector<AxisNest> retile_role_axes(
    const std::vector<TiledRange1>& U_axes,
    const std::vector<TiledRange1>& targets) {
  std::vector<AxisNest> out;
  out.reserve(U_axes.size());
  if (targets.empty()) {
    for (const auto& u : U_axes) out.push_back(retile_axis_nest(u, u));
  } else {
    TA_ASSERT(targets.size() == U_axes.size());
    for (std::size_t a = 0; a < U_axes.size(); ++a)
      out.push_back(retile_axis_nest(U_axes[a], targets[a]));
  }
  return out;
}

}  // namespace detail

inline std::vector<std::size_t> RetilePlan::u_result_ordinals(
    std::size_t t_grid_ordinal_in_slab) const {
  // The result-axis arrays must be populated consistently (by make_retile_plan);
  // an unpopulated/inconsistent plan should fail loudly, not silently identity.
  TA_ASSERT(result_axes.size() == result_t_extent.size() &&
            result_axes.size() == result_u_extent.size());
  const std::size_t nax = result_axes.size();
  if (nax == 0) return {t_grid_ordinal_in_slab};

  // Decompose the T grid ordinal into per-axis T-tile indices (row-major).
  std::vector<std::size_t> t_idx(nax);
  std::size_t rem = t_grid_ordinal_in_slab;
  for (std::size_t a = nax; a-- > 0;) {
    const std::size_t ext = result_t_extent[a] ? result_t_extent[a] : 1;
    t_idx[a] = rem % ext;
    rem /= ext;
  }

  // Per axis, map the T-tile index to the half-open range of U-tile indices it
  // covers. groups is indexed by the T tile uniformly across Identity, Coarsen,
  // and Refine (Refine yields a single-U-tile range [u,u+1) since the T cell
  // lies inside one U tile; the merge of several T cells onto that U tile
  // happens at finalize).
  std::vector<std::pair<std::size_t, std::size_t>> u_ranges(nax);
  for (std::size_t a = 0; a < nax; ++a)
    u_ranges[a] = result_axes[a].groups[t_idx[a]];

  // Cartesian product over axes, emitted in U-result row-major order.
  std::vector<std::size_t> out{0};
  for (std::size_t a = 0; a < nax; ++a) {
    const std::size_t u_ext = result_u_extent[a] ? result_u_extent[a] : 1;
    const auto [first, last] = u_ranges[a];
    std::vector<std::size_t> next;
    next.reserve(out.size() * (last - first));
    for (std::size_t base : out)
      for (std::size_t u = first; u < last; ++u)
        next.push_back(base * u_ext + u);
    out.swap(next);
  }
  return out;
}

/// Derive the nesting plan for a contraction whose user operands are tiled by
/// \c left_U and \c right_U, against a target tiling given per role axis
/// (\c targetH / \c targetM / \c targetN / \c targetK).
///
/// Two GEMM helpers play DISTINCT roles and must not be conflated:
///   - \c gh is the OUTER (SUMMA-level) helper. It is used ONLY to partition the
///     user operand axes into roles: left-outer = M, right-outer = N, contracted
///     (inner of the outer helper) = K, with the leading \c n_fused outer modes
///     being the fused (Hadamard) axes.
///   - \c inner_gh is the INNER (per-tile contract-reduce) helper. It is used
///     ONLY to classify the strided-BLAS axes exactly as the engine's
///     strided-install gates do off `contrreduce_op.gemm_helper()`
///:
///     `inner_gh.num_contract_ranks() == 0` (inner outer-product => ce_e =>
///     \c k_is_blas_k) vs `>= 1` (inner contraction => ce_ce => \c ride_on_M).
///
/// Throws TA::Exception if any axis's target tiling does not nest with the
/// user tiling.
inline RetilePlan make_retile_plan(const TiledRange& left_U,
                                   const TiledRange& right_U,
                                   const std::vector<TiledRange1>& targetH,
                                   const std::vector<TiledRange1>& targetM,
                                   const std::vector<TiledRange1>& targetN,
                                   const std::vector<TiledRange1>& targetK,
                                   const math::GemmHelper& gh,
                                   const math::GemmHelper& inner_gh,
                                   unsigned int n_fused) {
  RetilePlan plan;

  // Partition the user operand axes into roles via the GemmHelper. The leading
  // n_fused outer modes are Hadamard (shared by both operands); the remaining
  // left-outer modes are M, the remaining right-outer modes are N, and the
  // contracted (inner) modes are K (taken from the left operand's inner range).
  std::vector<TiledRange1> hU, mU, nU, kU;
  const unsigned int nf = n_fused;

  for (unsigned int i = gh.left_outer_begin(); i < gh.left_outer_end(); ++i) {
    if (i < gh.left_outer_begin() + nf)
      hU.push_back(left_U.dim(i));
    else
      mU.push_back(left_U.dim(i));
  }
  for (unsigned int i = gh.right_outer_begin(); i < gh.right_outer_end(); ++i) {
    // the leading nf right-outer modes mirror the left Hadamard modes; skip
    // them (already captured from the left operand).
    if (i < gh.right_outer_begin() + nf) continue;
    nU.push_back(right_U.dim(i));
  }
  for (unsigned int i = gh.left_inner_begin(); i < gh.left_inner_end(); ++i)
    kU.push_back(left_U.dim(i));

  plan.hadamard = detail::retile_role_axes(hU, targetH);
  plan.summaM = detail::retile_role_axes(mU, targetM);
  plan.summaN = detail::retile_role_axes(nU, targetN);
  plan.summaK = detail::retile_role_axes(kU, targetK);

  // active iff ANY axis on ANY role is non-Identity.
  auto any_non_identity = [](const std::vector<AxisNest>& v) {
    for (const auto& ax : v)
      if (ax.dir != NestDir::Identity) return true;
    return false;
  };
  plan.active = any_non_identity(plan.hadamard) ||
                any_non_identity(plan.summaM) || any_non_identity(plan.summaN) ||
                any_non_identity(plan.summaK);

  // Strided BLAS axis classification, exactly as the install gates classify off
  // the INNER contract-reduce helper (NOT the outer role-partition helper gh):
  //   ce_e  (inner outer-product, inner num_contract_ranks()==0) -> the
  //          contracted index is BLAS K (k_is_blas_k);
  //   ce_ce (inner contraction, inner num_contract_ranks()>=1)   -> a SUMMA
  //          external rides into BLAS M (ride_on_M).
  // ride_on_N is never set by this classifier (neither arm rides into BLAS N).
  // Index-structure-only: depends solely on inner_gh's contracted-rank count.
  if (inner_gh.num_contract_ranks() == 0u) {
    plan.k_is_blas_k = true;
  } else {
    plan.ride_on_M = true;
  }

  // Result axes for the ordinal map, in U-result row-major order:
  // Hadamard ++ M ++ N. Record per-axis T and U tile extents.
  auto append_result = [&plan](const std::vector<AxisNest>& role_nest,
                               const std::vector<TiledRange1>& role_U,
                               const std::vector<TiledRange1>& role_target) {
    for (std::size_t a = 0; a < role_nest.size(); ++a) {
      const AxisNest& ax = role_nest[a];
      plan.result_axes.push_back(ax);
      const std::size_t u_ext =
          static_cast<std::size_t>(role_U[a].tile_extent());
      const std::size_t t_ext =
          role_target.empty()
              ? u_ext
              : static_cast<std::size_t>(role_target[a].tile_extent());
      plan.result_u_extent.push_back(u_ext);
      plan.result_t_extent.push_back(t_ext);
    }
  };
  append_result(plan.hadamard, hU, targetH);
  append_result(plan.summaM, mU, targetM);
  append_result(plan.summaN, nU, targetN);

  // Carry the per-role TARGET tilings so the refine path can compute exact T
  // tile element boxes (an empty role => T == U, no refine on that role).
  plan.targetH = targetH;
  plan.targetM = targetM;
  plan.targetN = targetN;
  plan.targetK = targetK;

  return plan;
}

}  // namespace TiledArray::expressions

#endif  // TILEDARRAY_EXPRESSIONS_CONTRACTION_RETILE_H__INCLUDED
