//
// Created by Karl Pierce on 8/29/22.
//

#ifndef MPQC_CHEMISTRY_QC_LCAO_CC_PT_EINSUM_DOT_H
#define MPQC_CHEMISTRY_QC_LCAO_CC_PT_EINSUM_DOT_H
#include "TiledArray/fwd.h"
#include "TiledArray/dist_array.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/tiled_range.h"
//#include "TiledArray/util/string.h"

namespace TiledArray::expressions {
/// einsum function with result indices explicitly specified
/// @param[in] A first argument to the product
/// @param[in] B second argument to the product
/// @param[in] r result indices
/// @warning just as in the plain expression code, reductions are a special
/// case; use Expr::reduce()
template <typename T, typename U, typename V, typename... Indices>
auto einsum_dot(expressions::TsrExpr<T> A, expressions::TsrExpr<U> B,
                expressions::TsrExpr<V> D,
                World &world = get_default_world()) {
  static_assert(std::is_same<const T, const U>::value);
  static_assert(std::is_same<const T, const V>::value);
  using E = expressions::TsrExpr<const T>;
  // return dot.dot((einsum(E(A), E(B), Einsum::idx<T>(cs), world))(cs));
  return einsum_dot(E(A), E(B), E(D), world);
}

/// Specialized function to compute the einsum between two tensors
/// then dot with a third
template <typename Array_, typename... Indices>
auto einsum_dot(expressions::TsrExpr<Array_> A,
                     expressions::TsrExpr<Array_> B, expressions::TsrExpr<Array_> C,
                      World &world) {
  using Array = std::remove_cv_t<Array_>;
  using Tensor = typename Array::value_type;
  using Shape = typename Array::shape_type;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  auto c = std::get<0>(Einsum::idx(C));

  // these are "Hadamard" (fused) indices
  auto h = a & b & c;

  // no Hadamard indices => standard contraction (or even outer product)
  // same a, b, and c => pure Hadamard
  /*if (!h || (!(a ^ b) && !(b ^ c))) {
    Array R;
    R(std::string(c)) = A * B;
    return R(std::get<0>(cs)).dot(R).get();
  }*/

  auto e = (a ^ b);
  auto out = c - (h + e);
  TA_ASSERT(! (c - (h + e)));
  // contracted indices
  auto i = (a & b) - h;

  TA_ASSERT(e || h);

  using Einsum::index::small_vector;
  using Range = Einsum::Range;
  using RangeProduct = Einsum::RangeProduct<Range, small_vector<size_t>>;

  using RangeMap = Einsum::IndexMap<std::string, TiledRange1>;
  auto range_map =
      (RangeMap(a, A.array().trange()) | RangeMap(b, B.array().trange()) | RangeMap(c, C.array().trange()));

  using TiledArray::Permutation;
  using Einsum::index::permutation;

  struct Term {
    Array array;
    Einsum::Index<std::string> idx;
    Permutation permutation;
    RangeProduct tiles;
    TiledRange ei_tiled_range;
    Array ei;
    std::string expr;
    std::vector<std::pair<Einsum::Index<size_t>, Tensor>> local_tiles;
    bool own(Einsum::Index<size_t> h) const {
      for (Einsum::Index<size_t> ei : tiles) {
        auto idx = apply_inverse(permutation, h + ei);
        if (array.is_local(idx)) return true;
      }
      return false;
    }
  };

  Term terms[3] = {{A.array(), a}, {B.array(), b}, {C.array(), c}};

  for (auto &term : terms) {
    auto ei = (e + i & term.idx);
    if (term.idx != h + ei) {
      term.permutation = permutation(term.idx, h + ei);
    }
    term.expr = ei;
  }

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

  using Index = Einsum::Index<size_t>;

  typename Tensor::value_type dot_product = 0.0;
  // in this D is simply a vector so it should be fine to dot it
  // with C after the other portions are finished.
  TA_ASSERT(!e, "No external indices");

  // generalized contraction

  for (auto &term : terms) {
    auto ei = (e + i & term.idx);
    term.ei_tiled_range = TiledRange(range_map[ei]);
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
  }

  //std::vector<std::shared_ptr<World>> worlds;

  // iterates over tiles of hadamard indices
  for (Index h : H.tiles) {
    auto &[A, B, C] = terms;
    //auto own = A.own(h) || B.own(h) || C.own(h);
    //auto comm = world.mpi.comm().Split(own, world.rank());
    //worlds.push_back(std::make_unique<World>(comm));
    //auto &owners = worlds.back();
    //if (!own) continue;
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
          world, term.ei_tiled_range, term.local_tiles.begin(),
          term.local_tiles.end(), replicated);
    }
    dot_product += (A.ei(A.expr) * B.ei(B.expr)).dot(C.ei);//.set_world(world);
    for(auto & term : terms){
      term.ei.defer_deleter_to_next_fence();
      term.ei = Array();
    }
    // why omitting this fence leads to deadlock?
    //world.gop.fence();

    // dot the resulting tensor with CD
  }

//  for (auto &w : worlds) {
//    w->gop.fence();
//  }

//  world.gop.sum(dot_product);
//  world.gop.fence();

  return dot_product;
}
} // tiledarray::expressions
#endif  // MPQC_CHEMISTRY_QC_LCAO_CC_PT_EINSUM_DOT_H
