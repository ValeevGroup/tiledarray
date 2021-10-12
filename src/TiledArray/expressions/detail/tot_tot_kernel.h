/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_DETAIL_TOT_TOT_KERNEL_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_DETAIL_TOT_TOT_KERNEL_H__INCLUDED

#include "TiledArray/expressions/fwd.h"
#include "TiledArray/fwd.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/util/index.h"
#include "TiledArray/util/range.h"

namespace TiledArray::expressions::detail {

template <typename Array, typename... Indices>
auto tot_tot_kernel(TsrExpr<Array> A, TsrExpr<Array> B,
                    std::tuple<Index, Indices...> cs, World &world) {
  // These are the outer indices
  auto a = std::get<0>(idx(A));
  auto b = std::get<0>(idx(B));
  Index c = std::get<0>(cs);

  struct {
    std::string a, b, c;
  } inner;
  if constexpr (std::tuple_size<decltype(cs)>::value == 2) {
    inner.a = ";" + (std::string)std::get<1>(idx(A));
    inner.b = ";" + (std::string)std::get<1>(idx(B));
    inner.c = ";" + (std::string)std::get<1>(cs);
  }

  // these are "Hadamard" (fused) indices
  auto h = a & b & c;

  // no Hadamard indices => standard contraction (or even outer product)
  // same a, b, and c => pure Hadamard
  if (!h || (!(a ^ b) && !(b ^ c))) {
    Array C;
    C(std::string(c) + inner.c) = A * B;
    return C;
  }

  auto e = (a ^ b);
  // contracted indices
  auto i = (a & b) - h;

  TA_ASSERT(e);
  TA_ASSERT(h);

  using range::Range;
  using RangeProduct = range::RangeProduct<Range, index::small_vector<size_t> >;

  using RangeMap = IndexMap<std::string, TiledRange1>;
  auto range_map =
      (RangeMap(a, A.array().trange()) | RangeMap(b, B.array().trange()));

  using TiledArray::Permutation;
  using TiledArray::index::permutation;

  struct Term {
    Array array;
    Index idx;
    Permutation permutation;
    RangeProduct tiles;
    Array local;
    std::string expr;
  };

  Term AB[2] = {{A.array(), a}, {B.array(), b}};

  for (auto &term : AB) {
    auto ei = (e + i & term.idx);
    term.local = Array(world, TiledRange(range_map[ei]));
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
    if (term.idx != h + ei) {
      term.permutation = permutation(term.idx, h + ei);
    }
    term.expr = ei;
  }

  Term C = {Array(world, TiledRange(range_map[c])), c};
  for (auto idx : e) {
    C.tiles *= Range(range_map[idx].tiles_range());
  }
  if (C.idx != h + e) {
    C.permutation = permutation(h + e, C.idx);
  }
  C.expr = e;

  AB[0].expr += inner.a;
  AB[1].expr += inner.b;
  C.expr += inner.c;

  struct {
    RangeProduct tiles;
    std::vector<std::vector<size_t> > batch;
  } H;

  for (auto idx : h) {
    H.tiles *= Range(range_map[idx].tiles_range());
    H.batch.push_back({});
    for (auto r : range_map[idx]) {
      H.batch.back().push_back(Range{r}.size());
    }
  }

  // iterates over tiles of hadamard indices
  using Index = index::Index<size_t>;
  for (Index h : H.tiles) {
    size_t batch = 1;
    for (size_t i = 0; i < h.size(); ++i) {
      batch *= H.batch[i].at(h[i]);
    }
    for (auto &term : AB) {
      term.local = Array(term.local.world(), term.local.trange());
      const Permutation &P = term.permutation;
      for (Index ei : term.tiles) {
        auto tile = term.array.find(apply_inverse(P, h + ei)).get();
        if (P) tile = tile.permute(P);
        auto shape = term.local.trange().tile(ei);
        tile = tile.reshape(shape, batch);
        term.local.set(ei, tile);
      }
    }
    auto &[A, B] = AB;
    C.local(C.expr) = A.local(A.expr) * B.local(B.expr);
    const Permutation &P = C.permutation;
    for (Index e : C.tiles) {
      auto c = apply(P, h + e);
      auto shape = C.array.trange().tile(c);
      shape = apply_inverse(P, shape);
      auto tile = C.local.find(e).get();
      assert(tile.batch_size() == batch);
      tile = tile.reshape(shape);
      if (P) tile = tile.permute(P);
      C.array.set(c, tile);
    }
  }

  return C.array;
}

}  // namespace TiledArray::expressions::detail

#endif
