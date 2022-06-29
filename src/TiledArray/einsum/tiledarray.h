#ifndef TILEDARRAY_EINSUM_H__INCLUDED
#define TILEDARRAY_EINSUM_H__INCLUDED

#include "TiledArray/expressions/fwd.h"
#include "TiledArray/fwd.h"

#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/util/function.h"

namespace TiledArray::expressions {

/// einsum function without result indices assumes every index present
/// in both @p A and @p B is contracted, or, if there are no free indices,
/// pure Hadamard product is performed.
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template <typename T, typename U>
auto einsum(expressions::TsrExpr<T> A, expressions::TsrExpr<U> B) {
  // printf("einsum(A,B)\n");
  auto a = std::get<0>(idx(A));
  auto b = std::get<0>(idx(B));
  return einsum(A, B, std::string(a ^ b));
}

/// einsum function with result indices explicitly specified
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @param[in] r result indices
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template <typename T, typename U, typename... Indices>
auto einsum(expressions::TsrExpr<T> A, expressions::TsrExpr<U> B,
            const std::string &cs, World &world = get_default_world()) {
  static_assert(std::is_same<const T, const U>::value);
  using E = expressions::TsrExpr<const T>;
  return einsum(E(A), E(B), Einsum::idx<T>(cs), world);
}

template <typename Array_, typename... Indices>
auto einsum(expressions::TsrExpr<Array_> A, expressions::TsrExpr<Array_> B,
            std::tuple<Einsum::Index<std::string>, Indices...> cs,
            World &world) {
  using Array = std::remove_cv_t<Array_>;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  Einsum::Index<std::string> c = std::get<0>(cs);

  struct {
    std::string a, b, c;
  } inner;
  if constexpr (std::tuple_size<decltype(cs)>::value == 2) {
    inner.a = ";" + (std::string)std::get<1>(Einsum::idx(A));
    inner.b = ";" + (std::string)std::get<1>(Einsum::idx(B));
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

  TA_ASSERT(e || h);

  using Einsum::index::small_vector;
  using Range = Einsum::Range;
  using RangeProduct = Einsum::RangeProduct<Range, small_vector<size_t> >;

  using RangeMap = Einsum::IndexMap<std::string, TiledRange1>;
  auto range_map =
      (RangeMap(a, A.array().trange()) | RangeMap(b, B.array().trange()));

  using Einsum::index::permutation;
  using TiledArray::Permutation;

  struct Term {
    Array array;
    Einsum::Index<std::string> idx;
    Permutation permutation;
    RangeProduct tiles;
    TiledRange ei_tiled_range;
    Array ei;
    std::string expr;
    bool own(Einsum::Index<size_t> h) const {
      for (Einsum::Index<size_t> ei : tiles) {
        auto idx = apply_inverse(permutation, h + ei);
        if (array.is_local(idx)) return true;
      }
      return false;
    }
  };

  Term AB[2] = {{A.array(), a}, {B.array(), b}};

  for (auto &term : AB) {
    auto ei = (e + i & term.idx);
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

  using Index = Einsum::Index<size_t>;
  using Tensor = typename Array::value_type;

  if constexpr (std::tuple_size<decltype(cs)>::value > 1) {
    TA_ASSERT(e);
  } else if (!e) {  // hadamard reduction
    auto &[A, B] = AB;
    TiledRange trange(range_map[i]);
    RangeProduct tiles;
    for (auto idx : i) {
      tiles *= Range(range_map[idx].tiles_range());
    }
    auto pa = A.permutation;
    auto pb = B.permutation;
    for (Index h : H.tiles) {
      if (!C.array.is_local(h)) continue;
      size_t batch = 1;
      for (size_t i = 0; i < h.size(); ++i) {
        batch *= H.batch[i].at(h[i]);
      }
      Tensor tile(TiledArray::Range{batch});
      for (Index i : tiles) {
        // skip this unless both input tiles exist
        const auto pahi_inv = apply_inverse(pa, h + i);
        const auto pbhi_inv = apply_inverse(pb, h + i);
        if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;

        auto ai = A.array.find(pahi_inv).get();
        auto bi = B.array.find(pbhi_inv).get();
        if (pa) ai = ai.permute(pa);
        if (pb) bi = bi.permute(pb);
        auto shape = trange.tile(i);
        ai = ai.reshape(shape, batch);
        bi = bi.reshape(shape, batch);
        for (size_t k = 0; k < batch; ++k) {
          auto hk = ai.batch(k).dot(bi.batch(k));
          tile[k] = hk;
        }
      }
      auto pc = C.permutation;
      auto shape = apply_inverse(pc, C.array.trange().tile(h));
      tile = tile.reshape(shape);
      if (pc) tile = tile.permute(pc);
      C.array.set(h, tile);
    }
    return C.array;
  }

  // generalized contraction

  for (auto &term : AB) {
    auto ei = (e + i & term.idx);
    term.ei_tiled_range = TiledRange(range_map[ei]);
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
  }

  std::vector<std::shared_ptr<World> > worlds;
  auto fence_then_delete = [](World *world) {
    world->gop.fence();
    delete world;
  };

  // iterates over tiles of hadamard indices
  for (Index h : H.tiles) {
    auto &[A, B] = AB;
    auto own = A.own(h) || B.own(h);
    auto comm = world.mpi.comm().Split(own, world.rank());
    worlds.push_back(std::unique_ptr<World, decltype(fence_then_delete)>(
        new World{comm}, fence_then_delete));
    auto &owners = worlds.back();
    if (!own) continue;
    size_t batch = 1;
    for (size_t i = 0; i < h.size(); ++i) {
      batch *= H.batch[i].at(h[i]);
    }
    for (auto &term : AB) {
      term.ei = Array(*owners, term.ei_tiled_range);
      const Permutation &P = term.permutation;
      auto op_reshape =
          make_op_shared_handle([P, term_ei_trange = term.ei.trange(), batch](
                                    const auto &intile, const auto &ei) {
            auto shape = term_ei_trange.tile(ei);
            return P ? intile.permute(P).reshape(shape, batch)
                     : intile.reshape(shape, batch);
          });
      for (Index ei : term.tiles) {
        auto idx = apply_inverse(P, h + ei);
        if (!term.array.is_local(idx)) continue;
        auto intile_fut = term.array.find(idx);
        using Tile = std::decay_t<decltype(intile_fut.get())>;
        term.ei.set(ei, owners->taskq.add(
                            // wraps generic callable into nongeneric callable
                            [op_reshape](const Tile &tile, const Index &idx) {
                              return op_reshape(tile, idx);
                            },
                            intile_fut, ei));
      }
    }
    C.ei(C.expr) = (A.ei(A.expr) * B.ei(B.expr)).set_world(*owners);
    for (Index e : C.tiles) {
      if (!C.ei.is_local(e)) continue;
      const auto &P = C.permutation;
      auto op_reshape =
          make_op_shared_handle([P, C_array_trange = C.array.trange(), batch](
                                    const auto &intile, const auto &c) {
            assert(intile.batch_size() == batch);
            auto shape = apply_inverse(P, C_array_trange.tile(c));
            return P ? intile.reshape(shape).permute(P) : intile.reshape(shape);
          });
      auto intile_fut = C.ei.find(e);
      using Tile = std::decay_t<decltype(intile_fut.get())>;
      auto c = apply(P, h + e);
      C.array.set(c, owners->taskq.add(
                         // wraps generic callable into nongeneric callable
                         [op_reshape](const Tile &tile, const Index &idx) {
                           return op_reshape(tile, idx);
                         },
                         intile_fut, c));
    }
    // mark for lazy deletion
    A.ei = Array();
    B.ei = Array();
    C.ei = Array();
  }

  return C.array;
}

}  // namespace TiledArray::expressions

namespace TiledArray {

using expressions::einsum;

template <typename T, typename P>
auto einsum(const std::string &expr, const DistArray<T, P> &A,
            const DistArray<T, P> &B, World &world = get_default_world()) {
  namespace string = Einsum::string;
  auto [lhs, rhs] = string::split2(expr, "->");
  auto [a, b] = string::split2(lhs, ",");
  return einsum(A(string::join(a, ",")), B(string::join(b, ",")),
                string::join(rhs, ","));
}

}  // namespace TiledArray

#endif /* TILEDARRAY_EINSUM_H__INCLUDED */
