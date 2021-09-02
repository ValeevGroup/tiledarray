#ifndef TILEDARRAY_EINSUM_H__INCLUDED
#define TILEDARRAY_EINSUM_H__INCLUDED

#include "TiledArray/fwd.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/util/index.h"
#include "TiledArray/util/range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/tiled_range.h"
//#include "TiledArray/util/string.h"

namespace TiledArray::expressions {

/// einsum function without result indices assumes every index present
/// in both @p A and @p B is contracted, or, if there are no free indices,
/// pure Hadamard product is performed.
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template <typename Array>
auto einsum(TsrExpr<Array> A, TsrExpr<Array> B) {
  printf("einsum(A,B)\n");
  auto a = std::get<0>(idx(A));
  auto b = std::get<0>(idx(B));
  Array R;
  R(a ^ b) = A * B;
  return R;
}

/// einsum function with result indices explicitly specified
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @param[in] r result indices
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template<typename Array, typename ... Indices>
auto einsum(
  TsrExpr<Array> A, TsrExpr<Array> B,
  const std::string &cs,
  World &world = get_default_world())
{
  return einsum(A, B, idx<Array>(cs), world);
}

template<typename Array, typename ... Indices>
auto einsum(
  TsrExpr<Array> A, TsrExpr<Array> B,
  std::tuple<Index,Indices...> cs,
  World &world)
{

  auto a = std::get<0>(idx(A));
  auto b = std::get<0>(idx(B));
  Index c = std::get<0>(cs);

  struct { std::string a, b, c; } inner;
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
    C(std::string(c) + inner.c) = A*B;
    return C;
  }

  auto e = (a ^ b);
  // contracted indices
  auto i = (a & b) - h;

  TA_ASSERT(e);
  TA_ASSERT(h);

  using range::Range;
  using RangeProduct = range::RangeProduct<Range, index::small_vector<size_t> >;

  using RangeMap = IndexMap<std::string,TiledRange1>;
  auto range_map = (
    RangeMap(a, A.array().trange()) |
    RangeMap(b, B.array().trange())
  );

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

  Term AB[2] = { { A.array(), a }, { B.array(), b } };

  for (auto &term : AB) {
    auto ei = (e+i & term.idx);
    term.local = Array(world, TiledRange(range_map[ei]));
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
    if (term.idx != h+ei) {
      term.permutation = permutation(term.idx, h+ei);
    }
    term.expr = ei;
  }

  Term C = { Array(world, TiledRange(range_map[c])), c };
  for (auto idx : e) {
    C.tiles *= Range(range_map[idx].tiles_range());
  }
  if (C.idx != h+e) {
    C.permutation = permutation(h+e, C.idx);
  }
  C.expr = e;

  AB[0].expr += inner.a;
  AB[1].expr += inner.b;
  C.expr += inner.c;

  struct {
    RangeProduct tiles;
    std::vector< std::vector<size_t> > batch;
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
        auto tile = term.array.find(apply_inverse(P, h+ei)).get();
        if (P) tile = tile.permute(P);
        auto shape = term.local.trange().tile(ei);
        tile = tile.reshape(shape, batch);
        term.local.set(ei, tile);
      }
    }
    auto& [A,B] = AB;
    C.local(C.expr) = A.local(A.expr) * B.local(B.expr);
    const Permutation &P = C.permutation;
    for (Index e : C.tiles) {
      auto c = apply(P, h+e);
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

}  // namespace TiledArray::expressions

#endif /* TILEDARRAY_EINSUM_H__INCLUDED */
