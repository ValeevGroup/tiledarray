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
 *  mixed_retile_config.h
 *  Hardcoded coarsen targets for the env-var-gated mixed-path SUMMA retile.
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_MIXED_RETILE_CONFIG_H
#define TILEDARRAY_EXPRESSIONS_MIXED_RETILE_CONFIG_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace TiledArray::expressions::detail {

/// Target tile sizes (in elements) used by the env-var-gated auto-retile for
/// mixed (plain x arena-ToT -> arena-ToT) SUMMA contractions. A value of 0 =
/// "coarsen this role to a single tile spanning the full axis extent".
///
/// HARDCODED ON PURPOSE: the optimal coarsening tracks the physical system
/// size, so a maintainer edits these constants per target system size and
/// rebuilds.
///
/// The default strategy coarsens THREE roles:
///   1. the plain-operand BLAS external (SUMMA role M when the plain operand is
///      LEFT, role N when it is RIGHT) -> a single tile (target 0);
///   2. the contracted (SUMMA-K) axis (always shared) -> a single tile;
///   3. the arena-ToT operand's external -- the leftover SUMMA axis (role N when
///      the plain operand is LEFT, role M when it is RIGHT) -> tile size 16.
/// The fused (H) axes are left at the operands' own tiling (empty target).
///
/// COARSEN-ONLY (never refine): every target is fed through `coarsen_tr1`,
/// which only ever MERGES consecutive user (U) tiles onto existing U
/// boundaries; it never splits a tile. So a leftover-SUMMA axis whose tiles are
/// ALREADY >= the target (16) is kept intact (no refine -- refine is
/// unsupported here), while a finer axis is merged up toward 16.
struct MixedRetileConfig {
  /// target tile size for the plain operand's external (SUMMA-M if the plain
  /// operand is LEFT, SUMMA-N if it is RIGHT). 0 => single tile.
  std::size_t plain_external_target = 0;
  /// target tile size for the contracted (SUMMA-K) axis. 0 => single tile.
  std::size_t contracted_target = 0;
  /// target tile size for the arena-ToT operand's external -- the leftover
  /// SUMMA axis (role N if the plain operand is LEFT, role M if it is RIGHT).
  /// Coarsen-only: an axis already coarser than this is left intact (never
  /// refined). 0 => single tile.
  std::size_t tot_external_target = 16;
};

inline constexpr MixedRetileConfig mixed_retile_config{
    /*plain_external_target=*/0,  // collapse M (or N) on the plain operand
    /*contracted_target=*/0,      // collapse K
    /*tot_external_target=*/16,   // coarsen the leftover SUMMA axis to 16
};

/// Mutable enable flag for the mixed auto-retile gate. Default-initialized once
/// per process from the environment variable `TA_SUMMA_AUTO_RETILE` (truthy
/// `1` / `on` / `true` / `yes`, case-insensitive => enabled; unset or anything
/// else => disabled). The getenv read is cached (one-shot static init), but the
/// returned reference is settable so unit tests can flip the gate in-process
/// (mirrors `einsum_legacy_subworld()`); always restore the prior value.
inline bool& mixed_auto_retile() {
  static bool flag = [] {
    const char* s = std::getenv("TA_SUMMA_AUTO_RETILE");
    if (!s) return false;
    std::string t(s);
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return t == "1" || t == "on" || t == "true" || t == "yes";
  }();
  return flag;
}

/// Whether the mixed (plain x arena-ToT -> arena-ToT) auto-retile gate is on.
/// Reads the current value of the settable `mixed_auto_retile()` flag (the
/// getenv read it caches happens once); see that accessor for the semantics.
inline bool mixed_auto_retile_enabled() { return mixed_auto_retile(); }

}  // namespace TiledArray::expressions::detail

#endif  // TILEDARRAY_EXPRESSIONS_MIXED_RETILE_CONFIG_H
