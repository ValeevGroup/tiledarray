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
 *  Coarsen targets for the env-var-gated mixed-path SUMMA retile. All three
 *  roles (BLAS external, BLAS-K, SUMMA external) are runtime-overridable via the
 *  TA_SUMMA_MIXED_RETILE_* environment variables.
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_MIXED_RETILE_CONFIG_H
#define TILEDARRAY_EXPRESSIONS_MIXED_RETILE_CONFIG_H

#include <cstddef>
#include <cstdlib>

namespace TiledArray::expressions::detail {

/// Per-role coarsen targets (tile size in elements) for the mixed
/// (plain x arena-ToT -> arena-ToT) SUMMA auto-retile. A value of 0 leaves that
/// role INTACT; a positive value coarsens it toward that tile size.
///
/// RUNTIME-DRIVEN, NO ENABLE FLAG: every role defaults to 0, so the auto-retile
/// is inert unless at least one TA_SUMMA_MIXED_RETILE_* environment variable
/// requests a positive target (see `mixed_any_retile_requested()`); when none is
/// set the engine reverts to the stock SUMMA path. The compiled-in struct fields
/// are merely the fallback the runtime accessors read when their env var is unset.
/// See the settable accessors `mixed_plain_external_target()`,
/// `mixed_contracted_target()`, and `mixed_tot_external_target()`.
///
/// The THREE controllable roles (env var):
///   1. the plain-operand BLAS external (SUMMA role M when the plain operand is
///      LEFT, role N when it is RIGHT) | TA_SUMMA_MIXED_RETILE_BLAS_EXTERNAL;
///   2. the contracted (SUMMA-K) axis (always shared) | TA_SUMMA_MIXED_RETILE_BLAS_K;
///   3. the arena-ToT operand's external -- the leftover SUMMA axis (role N when
///      the plain operand is LEFT, role M when it is RIGHT)
///      | TA_SUMMA_MIXED_RETILE_SUMMA_EXTERNAL.
/// The fused (H) axes are always left at the operands' own tiling (intact).
///
/// COARSEN-ONLY (never refine): each positive target is fed through `coarsen_tr1`,
/// which only ever MERGES consecutive user (U) tiles onto existing U boundaries;
/// it never splits a tile. So an axis whose tiles are ALREADY >= the target is
/// kept intact, a finer axis is merged up toward the target, and a target >= the
/// axis extent collapses it to a single tile. Per the coarsen-only rule the
/// effective retile size is always >= the current tile size.
struct MixedRetileConfig {
  /// default target tile size for the plain operand's external (SUMMA-M if the
  /// plain operand is LEFT, SUMMA-N if it is RIGHT). 0 => intact.
  /// Runtime accessor: `mixed_plain_external_target()`.
  std::size_t plain_external_target = 0;
  /// default target tile size for the contracted (SUMMA-K) axis. 0 => intact.
  /// Runtime accessor: `mixed_contracted_target()`.
  std::size_t contracted_target = 0;
  /// default target tile size for the arena-ToT operand's external -- the
  /// leftover SUMMA axis (role N if the plain operand is LEFT, role M if it is
  /// RIGHT). 0 => intact. Runtime accessor: `mixed_tot_external_target()`.
  std::size_t tot_external_target = 0;
};

// All three roles default to 0 (= intact). The mixed auto-retile therefore does
// NOTHING unless at least one TA_SUMMA_MIXED_RETILE_* env var requests a
// (positive) coarsen target; when none is set the engine reverts to the stock
// SUMMA path. There is no separate enable flag.
inline constexpr MixedRetileConfig mixed_retile_config{
    /*plain_external_target=*/0,  // intact unless TA_SUMMA_MIXED_RETILE_BLAS_EXTERNAL set
    /*contracted_target=*/0,      // intact unless TA_SUMMA_MIXED_RETILE_BLAS_K set
    /*tot_external_target=*/0,    // intact unless TA_SUMMA_MIXED_RETILE_SUMMA_EXTERNAL set
};

/// Parse a non-negative integer environment variable. Returns `dflt` if the
/// variable is unset, empty, or has trailing non-numeric garbage. A bare `0` is
/// honored (=> leave the role INTACT).
inline std::size_t parse_size_env(const char* name, std::size_t dflt) {
  const char* s = std::getenv(name);
  if (!s) return dflt;
  char* end = nullptr;
  const unsigned long v = std::strtoul(s, &end, 10);
  if (end == s || *end != '\0') return dflt;
  return static_cast<std::size_t>(v);
}

/// Settable runtime targets (in elements) for the three coarsened roles of the
/// mixed (plain x arena-ToT -> arena-ToT) auto-retile. Each is default-initialized
/// once per process from its environment variable, falling back to the compiled-in
/// `mixed_retile_config` field (all 0) when the variable is unset/unparseable:
///
///   accessor                       | env var                             | default
///   -------------------------------|-------------------------------------|--------
///   mixed_plain_external_target()  | TA_SUMMA_MIXED_RETILE_BLAS_EXTERNAL  | 0
///   mixed_contracted_target()      | TA_SUMMA_MIXED_RETILE_BLAS_K         | 0
///   mixed_tot_external_target()    | TA_SUMMA_MIXED_RETILE_SUMMA_EXTERNAL | 0
///
/// `0` => leave that role INTACT (no retile on its axes). A positive value
/// COARSENS that role toward the given tile size (coarsen-only: an axis already
/// >= the target is kept intact, never refined; a target >= the axis extent
/// collapses the axis to a single tile). The getenv read is cached (one-shot
/// static init), but each returned reference is settable so unit tests can
/// override it in-process; always restore the prior value.
inline std::size_t& mixed_plain_external_target() {
  static std::size_t target = parse_size_env(
      "TA_SUMMA_MIXED_RETILE_BLAS_EXTERNAL",
      mixed_retile_config.plain_external_target);
  return target;
}
inline std::size_t& mixed_contracted_target() {
  static std::size_t target = parse_size_env(
      "TA_SUMMA_MIXED_RETILE_BLAS_K", mixed_retile_config.contracted_target);
  return target;
}
inline std::size_t& mixed_tot_external_target() {
  static std::size_t target = parse_size_env(
      "TA_SUMMA_MIXED_RETILE_SUMMA_EXTERNAL",
      mixed_retile_config.tot_external_target);
  return target;
}

/// Whether the mixed (plain x arena-ToT -> arena-ToT) auto-retile should engage:
/// true iff at least one of the three role targets is positive (i.e. some
/// TA_SUMMA_MIXED_RETILE_* env var requests coarsening, or a test set one). When
/// all three are 0 (the default -- none given) the engine reverts to the stock
/// SUMMA path. This replaces the former TA_SUMMA_AUTO_RETILE enable flag.
inline bool mixed_any_retile_requested() {
  return mixed_plain_external_target() != 0 || mixed_contracted_target() != 0 ||
         mixed_tot_external_target() != 0;
}

}  // namespace TiledArray::expressions::detail

#endif  // TILEDARRAY_EXPRESSIONS_MIXED_RETILE_CONFIG_H
