#ifndef TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED
#define TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED

#include "TiledArray/fwd.h"
#include "TiledArray/dist_array.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/tiled_range.h"

namespace TiledArray::Einsum {

using ::Einsum::index::small_vector;
using Range = ::Einsum::Range;
using RangeMap = ::Einsum::IndexMap<std::string,TiledRange1>;
using RangeProduct = ::Einsum::RangeProduct<Range, small_vector<size_t>>;

using ::Einsum::index::Index;
using ::Einsum::index::IndexMap;

using ::Einsum::index::Permutation;
using ::Einsum::index::permutation;

/// converts the annotation of an expression to an Index
template <typename Array>
auto idx(const std::string &s) {
  using Index = Einsum::Index<std::string>;
  if constexpr (detail::is_tensor_of_tensor_v<typename Array::value_type>) {
    auto semi = std::find(s.begin(), s.end(), ';');
    TA_ASSERT(semi != s.end());
    auto [first,second] = ::Einsum::string::split2(s, ";");
    TA_ASSERT(!first.empty());
    TA_ASSERT(!second.empty());
    return std::tuple<Index, Index>{first, second};
  } else {
    return std::tuple<Index>{s};
  }
}

/// converts the annotation of an expression to an Index
template <typename A, bool Alias>
auto idx(const TiledArray::expressions::TsrExpr<A, Alias> &e) {
  return idx<A>(e.annotation());
}

template<typename Array>
struct ArrayTerm {
  using Tensor = typename Array::value_type;
  Array array;
  Einsum::Index<std::string> idx;
  Permutation permutation;
  RangeProduct tiles;
  TiledRange ei_tiled_range;
  Array ei;
  std::string expr;
  std::vector< std::pair<Einsum::Index<size_t>,Tensor> > local_tiles;
  bool own(Einsum::Index<size_t> h) const {
    for (Einsum::Index<size_t> ei : tiles) {
      auto idx = apply_inverse(permutation, h+ei);
      if (array.is_local(idx)) return true;
    }
    return false;
  }
};

template<typename Array_, typename ... Indices>
auto einsum(
  expressions::TsrExpr<Array_> A,
  expressions::TsrExpr<Array_> B,
  std::tuple<Einsum::Index<std::string>,Indices...> cs,
  World &world)
{

  using Array = std::remove_cv_t<Array_>;
  using Tensor = typename Array::value_type;
  using Shape = typename Array::shape_type;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  Einsum::Index<std::string> c = std::get<0>(cs);

  struct { std::string a, b, c; } inner;
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
    C(std::string(c) + inner.c) = A*B;
    return C;
  }

  auto e = (a ^ b);
  // contracted indices
  auto i = (a & b) - h;

  TA_ASSERT(e || h);

  auto range_map = (
    RangeMap(a, A.array().trange()) |
    RangeMap(b, B.array().trange())
  );

  using TiledArray::Permutation;
  using ::Einsum::index::permutation;

  ArrayTerm<Array> AB[2] = { { A.array(), a }, { B.array(), b } };

  for (auto &term : AB) {
    auto ei = (e+i & term.idx);
    if (term.idx != h+ei) {
      term.permutation = permutation(term.idx, h+ei);
    }
    term.expr = ei;
  }

  ArrayTerm<Array> C = { Array(world, TiledRange(range_map[c])), c };
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

  using Index = Einsum::Index<size_t>;

  if constexpr(std::tuple_size<decltype(cs)>::value > 1) {
    TA_ASSERT(e);
  }
  else if (!e) { // hadamard reduction
    auto& [A,B] = AB;
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
      Tensor tile(TiledArray::Range{batch}, typename Tensor::value_type());
      for (Index i : tiles) {
        // skip this unless both input tiles exist
        const auto pahi_inv = apply_inverse(pa,h+i);
        const auto pbhi_inv = apply_inverse(pb,h+i);
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
    auto ei = (e+i & term.idx);
    term.ei_tiled_range = TiledRange(range_map[ei]);
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
  }

  std::vector< std::shared_ptr<World> > worlds;
  std::vector< std::tuple<Index,Tensor> > local_tiles;

  // iterates over tiles of hadamard indices
  for (Index h : H.tiles) {
    auto& [A,B] = AB;
    auto own = A.own(h) || B.own(h);
    auto comm = world.mpi.comm().Split(own, world.rank());
    worlds.push_back(std::make_unique<World>(comm));
    auto &owners = worlds.back();
    if (!own) continue;
    size_t batch = 1;
    for (size_t i = 0; i < h.size(); ++i) {
      batch *= H.batch[i].at(h[i]);
    }
    for (auto &term : AB) {
      term.local_tiles.clear();
      const Permutation &P = term.permutation;

      for (Index ei : term.tiles) {
        auto idx = apply_inverse(P, h+ei);
        if (!term.array.is_local(idx)) continue;
        if (term.array.is_zero(idx)) continue;
        // TODO no need for immediate evaluation
        auto tile = term.array.find(idx).get();
        if (P) tile = tile.permute(P);
        auto shape = term.ei_tiled_range.tile(ei);
        tile = tile.reshape(shape, batch);
        term.local_tiles.push_back({ei, tile});
      }
      bool replicated = term.array.pmap()->is_replicated();
      term.ei = TiledArray::make_array<Array>(
        *owners,
        term.ei_tiled_range,
        term.local_tiles.begin(),
        term.local_tiles.end(),
        replicated
      );
    }
    C.ei(C.expr) = (A.ei(A.expr) * B.ei(B.expr)).set_world(*owners);
    A.ei.defer_deleter_to_next_fence();
    B.ei.defer_deleter_to_next_fence();
    A.ei = Array();
    B.ei = Array();
    // why omitting this fence leads to deadlock?
    owners->gop.fence();
    for (Index e : C.tiles) {
      if (!C.ei.is_local(e)) continue;
      if (C.ei.is_zero(e)) continue;
      // TODO no need for immediate evaluation
      auto tile = C.ei.find(e).get();
      assert(tile.batch_size() == batch);
      const Permutation &P = C.permutation;
      auto c = apply(P, h+e);
      auto shape = C.array.trange().tile(c);
      shape = apply_inverse(P, shape);
      tile = tile.reshape(shape);
      if (P) tile = tile.permute(P);
      local_tiles.push_back({c, tile});
    }
    // mark for lazy deletion
    C.ei = Array();
  }

  if constexpr (!Shape::is_dense()) {
    TiledRange tiled_range = TiledRange(range_map[c]);
    std::vector< std::pair<Index,float> > tile_norms;
    for (auto& [index,tile] : local_tiles) {
      tile_norms.push_back({index,tile.norm()});
    }
    Shape shape(world, tile_norms, tiled_range);
    C.array = Array(world, TiledRange(range_map[c]), shape);
  }

  for (auto& [index,tile] : local_tiles) {
    if (C.array.is_zero(index)) continue;
    C.array.set(index, tile);
  }

  for (auto &w : worlds) {
    w->gop.fence();
  }

  return C.array;

}

/// Specialized function to compute the einsum between two tensors
/// then dot with a third
template <typename Array_, typename... Indices>
auto dot(
  expressions::TsrExpr<Array_> A,
  expressions::TsrExpr<Array_> B,
  expressions::TsrExpr<Array_> C,
  World &world)
{
  using Array = std::remove_cv_t<Array_>;
  using Tensor = typename Array::value_type;
  using Shape = typename Array::shape_type;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  auto c = std::get<0>(Einsum::idx(C));

  // these are "Hadamard" (fused) indices
  auto h = a & b & c;
  auto ab_e = (a ^ b);
  auto ab_i = (a & b)-h;
  TA_ASSERT(ab_e);

  // no Hadamard indices => standard contraction
  if (!h) {
    Array AB;
    AB(ab_e) = A*B;
    return AB(ab_e).dot(C).get();
  }

  TA_ASSERT(sorted(c) == sorted(h + ab_e));

  auto range_map = (
    RangeMap(a, A.array().trange()) |
    RangeMap(b, B.array().trange()) |
    RangeMap(c, C.array().trange())
  );

  struct {
    RangeProduct tiles;
    std::vector<std::vector<size_t>> batch;
  } H;

  for (auto idx : h) {
    H.tiles *= Range(range_map[idx].tiles_range());
    H.batch.push_back({});
    for (auto r : range_map[idx]) {
      H.batch.back().push_back(Range{r}.size());
    }
  }

  ArrayTerm<Array> terms[3] = {{A.array(), a}, {B.array(), b}, {C.array(), c}};

  for (auto &term : terms) {
    auto ei = (ab_e + ab_i & term.idx);
    if (term.idx != h + ei) {
      term.permutation = permutation(term.idx, h + ei);
    }
    term.expr = ei;
    term.ei_tiled_range = TiledRange(range_map[ei]);
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
  }

  using Index = Einsum::Index<size_t>;
  typename Tensor::value_type result = 0.0;

  // iterates over tiles of hadamard indices
  for (Index h : H.tiles) {
    auto &[A, B, C] = terms;
    size_t batch = 1;
    for (size_t i = 0; i < h.size(); ++i) {
      batch *= H.batch[i].at(h[i]);
    }
    for (auto &term : terms) {
      term.local_tiles.clear();
      const Permutation &P = term.permutation;

      for (Index ei : term.tiles) {
        auto idx = apply_inverse(P, h + ei);
        if (!term.array.is_local(idx)) continue;
        if (term.array.is_zero(idx)) continue;
        // TODO no need for immediate evaluation
        auto tile = term.array.find(idx).get();
        if (P) tile = tile.permute(P);
        auto shape = term.ei_tiled_range.tile(ei);
        tile = tile.reshape(shape, batch);
        term.local_tiles.push_back({ei, tile});
      }
      bool replicated = term.array.pmap()->is_replicated();
      term.ei = TiledArray::make_array<Array>(
          world, term.ei_tiled_range,
          term.local_tiles.begin(),
          term.local_tiles.end(),
          replicated
      );
    }
    result += (A.ei(A.expr) * B.ei(B.expr)).dot(C.ei(C.expr)).get();
    for(auto & term : terms){
      term.ei.defer_deleter_to_next_fence();
      term.ei = Array();
    }

  }

  world.gop.fence();

  return result;

}

} // tiledarray::expressions

namespace TiledArray::expressions {

/// einsum function without result indices assumes every index present
/// in both @p A and @p B is contracted, or, if there are no free indices,
/// pure Hadamard product is performed.
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template<typename T, typename U>
auto einsum(expressions::TsrExpr<T> A, expressions::TsrExpr<U> B) {
  auto a = std::get<0>(idx(A));
  auto b = std::get<0>(idx(B));
  return einsum(A, B, std::string(a^b));
}

/// einsum function with result indices explicitly specified
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @param[in] r result indices
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template<typename T, typename U, typename ... Indices>
auto einsum(
  expressions::TsrExpr<T> A,
  expressions::TsrExpr<U> B,
  const std::string &cs,
  World &world = get_default_world())
{
  static_assert(std::is_same<const T, const U>::value);
  using E = expressions::TsrExpr<const T>;
  return Einsum::einsum(E(A), E(B), Einsum::idx<T>(cs), world);
}

template <typename T, typename U, typename V>
auto dot(expressions::TsrExpr<T> A,
         expressions::TsrExpr<U> B,
         expressions::TsrExpr<V> C,
         World &world = get_default_world())
{
  static_assert(std::is_same<const T, const U>::value);
  static_assert(std::is_same<const T, const V>::value);
  using E = expressions::TsrExpr<const T>;
  return Einsum::dot(E(A), E(B), E(C), world);
}

} // TiledArray::expressions

namespace TiledArray {

using expressions::einsum;
using expressions::dot;

template<typename T, typename P>
auto einsum(
  const std::string &expr,
  const DistArray<T,P> &A,
  const DistArray<T,P> &B,
  World &world = get_default_world())
{
  namespace string = ::Einsum::string;
  auto [lhs,rhs] = string::split2(expr, "->");
  auto [a,b] = string::split2(lhs,",");
  return einsum(
    A(string::join(a,",")),
    B(string::join(b,",")),
    string::join(rhs,","),
    world
  );
}

template<typename T, typename P>
auto dot(
  const std::string &expr,
  const DistArray<T,P> &A,
  const DistArray<T,P> &B,
  const DistArray<T,P> &C,
  World &world = get_default_world())
{
  namespace string = ::Einsum::string;
  auto [a,bc] = string::split2(expr,",");
  auto [b,c] = string::split2(bc,",");
  return dot(
    A(string::join(a,",")),
    B(string::join(b,",")),
    C(string::join(c,",")),
    world
  );
}

}  // namespace TiledArray

#endif // TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED
