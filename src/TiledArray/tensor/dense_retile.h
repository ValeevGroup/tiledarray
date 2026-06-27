/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  dense_retile.h
 *  In-rank dense block packer for the in-SUMMA two-trange contraction retile.
 *
 *  Dense analogue of arena_retile.h's arena_gather_block, for the MIXED path
 *  where one operand is a plain TA::Tensor<T> (no inner cells, no arena pages).
 *  A plain block scatter: copy each fine tile into its sub-box of the coarse
 *  tile; outer positions covered by no fine tile stay zero (holes). The coarse
 *  tile is contiguous (row-major), so it is always strided-DGEMM-eligible.
 */

#ifndef TILEDARRAY_TENSOR_DENSE_RETILE_H__INCLUDED
#define TILEDARRAY_TENSOR_DENSE_RETILE_H__INCLUDED

#include "TiledArray/error.h"
#include "TiledArray/range.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace TiledArray {
namespace detail {

/// Pack a block of fine plain `TA::Tensor<T>` tiles into ONE contiguous coarse
/// tile (the dense analogue of arena_gather_block). `fine` are the
/// source outer tiles partitioning the coarse block (each `range()` a sub-box
/// of `coarse_outer`, same coordinate system); `coarse_outer` is the result
/// range. Outer positions not covered by any fine tile are left zero (holes).
/// The result is contiguous row-major, hence constant-stride / strided-eligible.
///
/// `nbatch` is required to be 1: the mixed (plain x ToT) contraction this serves
/// has no Hadamard/fused index (H = empty), so the plain operand is never
/// batched. A batched plain operand (fused mixed) is a separate follow-up; the
/// assert makes the unsupported case fail loudly rather than mis-pack.
/// @pre nbatch == 1
template <typename DenseOuter>
DenseOuter dense_gather_block(const std::vector<DenseOuter>& fine,
                              const Range& coarse_outer, std::size_t nbatch) {
  using value_t = typename DenseOuter::value_type;
  TA_ASSERT(nbatch == 1);
  DenseOuter result(coarse_outer);
  std::fill(result.data(), result.data() + result.size(), value_t(0));
  for (const auto& f : fine) {
    if (f.empty()) continue;
    const auto& fr = f.range();
    for (std::size_t p = 0; p < fr.volume(); ++p) {
      auto idx = fr.idx(p);
      TA_ASSERT(coarse_outer.includes(idx));
      result.data()[coarse_outer.ordinal(idx)] = f.data()[p];
    }
  }
  return result;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_DENSE_RETILE_H__INCLUDED
