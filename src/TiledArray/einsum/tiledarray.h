#ifndef TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED
#define TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED

#include "TiledArray/dist_array.h"
#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/fwd.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"

namespace TiledArray::Einsum {

using ::Einsum::index::small_vector;
using Range = ::Einsum::Range;
using RangeMap = ::Einsum::IndexMap<std::string, TiledRange1>;
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
    auto [first, second] = ::Einsum::string::split2(s, ";");
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

template <typename Array>
struct ArrayTerm {
  using Tensor = typename Array::value_type;
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

namespace {
template <typename DArrayT>
constexpr bool IsArrayT = detail::is_tensor_v<typename DArrayT::value_type>;

template <typename DArrayToT>
constexpr bool IsArrayToT =
    detail::is_tensor_of_tensor_v<typename DArrayToT::value_type>;

template <typename ArrayT1, typename ArrayT2>
constexpr bool AreArrayT = IsArrayT<ArrayT1> && IsArrayT<ArrayT2>;

template <typename ArrayT1, typename ArrayT2>
constexpr bool AreArrayToT = IsArrayToT<ArrayT1> && IsArrayToT<ArrayT2>;

template <typename ArrayT1, typename ArrayT2>
constexpr bool AreArraySame =
    AreArrayT<ArrayT1, ArrayT2> || AreArrayToT<ArrayT1, ArrayT2>;

}  // namespace

namespace {

///
/// \brief This function replicates a tensor B into a tensor A such that
///        A(a_1,...a_k,i_1,...,i_l) = B(i_1,...,i_l). Evidently, the
///        extents of i_n modes must match in both A and B.
///
/// \tparam Tensor TiledArray::Tensor type.
/// \param to The target tensor.
/// \param from The source tensor that will be replicated into \c to.
///
template <typename Tensor,
          typename = std::enable_if_t<detail::is_nested_tensor_v<Tensor>>>
void replicate_tensor(Tensor &to, Tensor const &from) {
  // assert that corresponding modes have the same extents
  TA_ASSERT(std::equal(from.range().extent().rbegin(),
                       from.range().extent().rend(),
                       to.range().extent().rbegin()));

  // number of elements to be copied
  // (same as the number of elements in @c from)
  auto const N = from.range().volume();
  for (auto i = 0; i < to.range().volume(); i += N)
    std::copy(from.begin(), from.end(), to.data()+i);
}

template <typename Array,
          typename = std::enable_if_t<detail::is_array_v<Array>>>
auto replicate_array(Array from, TiledRange const& prepend_trng) {
  auto const result_rank = prepend_trng.rank() + rank(from);
  container::svector<TiledRange1> tr1s;
  tr1s.reserve(result_rank);
  for (auto const& r : prepend_trng) tr1s.emplace_back(r);
  for (auto const& r : from.trange()) tr1s.emplace_back(r);
  auto const result_trange = TiledRange(tr1s);

  from.make_replicated();

  auto result = make_array<Array>(
      get_default_world(), result_trange,
      [from, res_tr = result_trange.tiles_range(),
       delta_rank = prepend_trng.rank()](auto& tile, auto const& res_rng,
                                         auto res_ord) {
        using std::begin;
        using std::end;
        using std::next;

        typename Array::value_type repped(res_rng);
        auto res_coord_ix = res_tr.idx(res_ord);
        auto from_coord_ix = decltype(res_coord_ix)(
            next(begin(res_coord_ix), delta_rank), end(res_coord_ix));
        replicate_tensor(repped, from.find_local(from_coord_ix).get(false));
        tile = repped;
      });

  //clang-format off
  //  using std::begin;
  //  using std::next;
  //  using std::end;
  //
  //  Array result(get_default_world(), result_trange);
  //
  //  for (auto tile : result) {
  //    auto res_tix = tile.index();
  //    auto from_tix = decltype(res_tix)(next(begin(res_tix),
  //    prepend_trng.rank()), end(res_tix));
  //    if (result.is_local(res_tix) && !result.is_zero(res_tix) &&
  //        !from.is_zero(from_tix)) {
  //      typename Array::value_type
  //      repped(result.trange().make_tile_range(res_tix)); auto found =
  //      from.find_local(from_tix).get(false); replicate_tensor(repped, found);
  //      tile = repped;
  //    }
  //  }
  //clang-format on

  return result;
}

}  // namespace

template <typename ArrayA_, typename ArrayB_, typename... Indices>
auto einsum(expressions::TsrExpr<ArrayA_> A, expressions::TsrExpr<ArrayB_> B,
            std::tuple<Einsum::Index<std::string>, Indices...> cs,
            World &world) {
  using ArrayA = std::remove_cv_t<ArrayA_>;
  using ArrayB = std::remove_cv_t<ArrayB_>;
  using ArrayC = std::conditional_t<
      AreArraySame<ArrayA, ArrayB>, ArrayA,
      std::conditional_t<IsArrayToT<ArrayA>, ArrayA, ArrayB>>;
  using ResultTensor = typename ArrayC::value_type;
  using ResultShape = typename ArrayC::shape_type;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  Einsum::Index<std::string> c = std::get<0>(cs);

  struct {
    std::string a, b, c;
    // Hadamard, external, internal indices for inner tensor
    Einsum::Index<std::string> A, B, C, h, e, i;
  } inner;
  if constexpr (std::tuple_size<decltype(cs)>::value == 2) {
    if constexpr (IsArrayToT<ArrayA>)
      inner.a = ";" + (std::string)std::get<1>(Einsum::idx(A));

    if constexpr (IsArrayToT<ArrayB>)
      inner.b = ";" + (std::string)std::get<1>(Einsum::idx(B));

    static_assert(IsArrayToT<ArrayA> || IsArrayToT<ArrayB>);
    inner.c = ";" + (std::string)std::get<1>(cs);

    Einsum::Index<std::string> a_idx, b_idx, c_idx;
    if constexpr (IsArrayToT<ArrayA>) inner.A = std::get<1>(Einsum::idx(A));
    if constexpr (IsArrayToT<ArrayB>) inner.B = std::get<1>(Einsum::idx(B));
    if constexpr (IsArrayToT<ArrayA> || IsArrayToT<ArrayB>)
      inner.C = std::get<1>(cs);

    inner.h = inner.A & inner.B & inner.C;
    inner.e = (inner.A ^ inner.B);
    inner.i = (inner.A & inner.B) - inner.h;
    TA_ASSERT(!(inner.h && (inner.i || inner.e)) &&
              "General product between inner tensors not supported");
  }

  // these are "Hadamard" (fused) indices
  auto h = a & b & c;

  // external indices
  auto e = (a ^ b);

  // contracted indices
  auto i = (a & b) - h;

  // no Hadamard indices => standard contraction (or even outer product)
  // same a, b, and c => pure Hadamard
  if (!h || (h && !(i || e))) {
    ArrayC C;
    C(std::string(c) + inner.c) = A * B;
    return C;
  }

  TA_ASSERT(e || h);

  auto range_map =
      (RangeMap(a, A.array().trange()) | RangeMap(b, B.array().trange()));

  using ::Einsum::index::permutation;
  using TiledArray::Permutation;

  std::tuple<ArrayTerm<ArrayA>, ArrayTerm<ArrayB>> AB{{A.array(), a},
                                                      {B.array(), b}};

  auto update_perm_and_indices = [&e = std::as_const(e), &i = std::as_const(i),
                                  &h = std::as_const(h)](auto &term) {
    auto ei = (e + i & term.idx);
    if (term.idx != h + ei) {
      term.permutation = permutation(term.idx, h + ei);
    }
    term.expr = ei;
  };

  std::invoke(update_perm_and_indices, std::get<0>(AB));
  std::invoke(update_perm_and_indices, std::get<1>(AB));

  ArrayTerm<ArrayC> C = {ArrayC(world, TiledRange(range_map[c])), c};
  for (auto idx : e) {
    C.tiles *= Range(range_map[idx].tiles_range());
  }
  if (C.idx != h + e) {
    C.permutation = permutation(h + e, C.idx);
  }
  C.expr = e;

  std::get<0>(AB).expr += inner.a;
  std::get<1>(AB).expr += inner.b;

  C.expr += inner.c;

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

  if constexpr (AreArraySame<ArrayA, ArrayB>) {
    if (!e) {  // hadamard reduction
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
        ResultTensor tile(TiledArray::Range{batch},
                          typename ResultTensor::value_type{});
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
            using Ix = ::Einsum::Index<std::string>;
            if constexpr (AreArrayToT<ArrayA, ArrayB>) {
              auto aik = ai.batch(k);
              auto bik = bi.batch(k);
              auto vol = aik.total_size();
              TA_ASSERT(vol == bik.total_size());

              auto &el = tile({k});
              using TensorT = std::remove_reference_t<decltype(el)>;

              auto mult_op = [&inner](auto const &l, auto const &r) -> TensorT {
                return inner.h ? TA::detail::tensor_hadamard(l, inner.A, r,
                                                             inner.B, inner.C)
                               : TA::detail::tensor_contract(l, inner.A, r,
                                                             inner.B, inner.C);
              };

              for (auto i = 0; i < vol; ++i)
                el.add_to(mult_op(aik.data()[i], bik.data()[i]));

            } else {
              auto hk = ai.batch(k).dot(bi.batch(k));
              tile({k}) += hk;
            }
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
  }

  // generalized contraction

  auto update_tr = [&e = std::as_const(e), &i = std::as_const(i),
                    &range_map = std::as_const(range_map)](auto &term) {
    auto ei = (e + i & term.idx);
    term.ei_tiled_range = TiledRange(range_map[ei]);
    for (auto idx : ei) {
      term.tiles *= Range(range_map[idx].tiles_range());
    }
  };

  std::invoke(update_tr, std::get<0>(AB));
  std::invoke(update_tr, std::get<1>(AB));

  std::vector<std::shared_ptr<World>> worlds;
  std::vector<std::tuple<Index, ResultTensor>> local_tiles;

  // iterates over tiles of hadamard indices
  for (Index h : H.tiles) {
    auto &[A, B] = AB;
    auto own = A.own(h) || B.own(h);
    auto comm = world.mpi.comm().Split(own, world.rank());
    worlds.push_back(std::make_unique<World>(comm));
    auto &owners = worlds.back();
    if (!own) continue;
    size_t batch = 1;
    for (size_t i = 0; i < h.size(); ++i) {
      batch *= H.batch[i].at(h[i]);
    }

    auto retile = [&owners, &h = std::as_const(h), batch](auto &term) {
      term.local_tiles.clear();
      const Permutation &P = term.permutation;

      for (Index ei : term.tiles) {
        auto idx = apply_inverse(P, h + ei);
        if (!term.array.is_local(idx)) continue;
        if (term.array.is_zero(idx)) continue;
        // TODO no need for immediate evaluation
        auto tile = term.array.find_local(idx).get();
        if (P) tile = tile.permute(P);
        auto shape = term.ei_tiled_range.tile(ei);
        tile = tile.reshape(shape, batch);
        term.local_tiles.push_back({ei, tile});
      }
      bool replicated = term.array.pmap()->is_replicated();
      term.ei = TiledArray::make_array<decltype(term.array)>(
          *owners, term.ei_tiled_range, term.local_tiles.begin(),
          term.local_tiles.end(), replicated);
    };
    std::invoke(retile, std::get<0>(AB));
    std::invoke(retile, std::get<1>(AB));

    C.ei(C.expr) = (A.ei(A.expr) * B.ei(B.expr)).set_world(*owners);
    A.ei.defer_deleter_to_next_fence();
    B.ei.defer_deleter_to_next_fence();
    A.ei = ArrayA();
    B.ei = ArrayB();
    // why omitting this fence leads to deadlock?
    owners->gop.fence();
    for (Index e : C.tiles) {
      if (!C.ei.is_local(e)) continue;
      if (C.ei.is_zero(e)) continue;
      // TODO no need for immediate evaluation
      auto tile = C.ei.find_local(e).get();
      assert(tile.nbatch() == batch);
      const Permutation &P = C.permutation;
      auto c = apply(P, h + e);
      auto shape = C.array.trange().tile(c);
      shape = apply_inverse(P, shape);
      tile = tile.reshape(shape);
      if (P) tile = tile.permute(P);
      local_tiles.push_back({c, tile});
    }
    // mark for lazy deletion
    C.ei = ArrayC();
  }

  if constexpr (!ResultShape::is_dense()) {
    TiledRange tiled_range = TiledRange(range_map[c]);
    std::vector<std::pair<Index, float>> tile_norms;
    for (auto &[index, tile] : local_tiles) {
      tile_norms.push_back({index, tile.norm()});
    }
    ResultShape shape(world, tile_norms, tiled_range);
    C.array = ArrayC(world, TiledRange(range_map[c]), shape);
  }

  for (auto &[index, tile] : local_tiles) {
    if (C.array.is_zero(index)) continue;
    C.array.set(index, tile);
  }

  for (auto &w : worlds) {
    w->gop.fence();
  }

  return C.array;
}

/// Computes ternary tensor product whose result
/// is a scalar (a ternary dot product). Optimized for the case where
/// the arguments have common (Hadamard) indices.

/// \tparam Array_ a DistArray type
/// \param A an annotated Array_
/// \param B an annotated Array_
/// \param C an annotated Array_
/// \param world the World in which to compute the result
/// \return scalar result
/// \note if \p A , \p B , and \p C share indices computes `A*B` one slice at
/// a time and contracts with the corresponding slice `C`; thus storage of
/// `A*B` is eliminated
template <typename Array_>
auto dot(expressions::TsrExpr<Array_> A, expressions::TsrExpr<Array_> B,
         expressions::TsrExpr<Array_> C, World &world) {
  using Array = std::remove_cv_t<Array_>;
  using Tensor = typename Array::value_type;
  using Shape = typename Array::shape_type;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  auto c = std::get<0>(Einsum::idx(C));

  // these are "Hadamard" (fused) indices
  auto h = a & b & c;
  auto ab_e = (a ^ b);
  auto ab_i = (a & b) - h;
  TA_ASSERT(ab_e);

  // no Hadamard indices => standard contraction
  if (!h) {
    Array AB;
    AB(ab_e) = A * B;
    return AB(ab_e).dot(C).get();
  }

  TA_ASSERT(sorted(c) == sorted(h + ab_e));

  auto range_map =
      (RangeMap(a, A.array().trange()) | RangeMap(b, B.array().trange()) |
       RangeMap(c, C.array().trange()));

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
          world, term.ei_tiled_range, term.local_tiles.begin(),
          term.local_tiles.end(), replicated);
    }
    result += (A.ei(A.expr) * B.ei(B.expr)).dot(C.ei(C.expr)).get();
    for (auto &term : terms) {
      term.ei.defer_deleter_to_next_fence();
      term.ei = Array();
    }
  }

  world.gop.fence();

  return result;
}

}  // namespace TiledArray::Einsum

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
  using ECT = expressions::TsrExpr<const T>;
  using ECU = expressions::TsrExpr<const U>;
  using ResultExprT = std::conditional_t<Einsum::IsArrayToT<T>, T, U>;
  return Einsum::einsum(ECT(A), ECU(B), Einsum::idx<ResultExprT>(cs), world);
}

template <typename T, typename U, typename V>
auto dot(expressions::TsrExpr<T> A, expressions::TsrExpr<U> B,
         expressions::TsrExpr<V> C, World &world = get_default_world()) {
  static_assert(std::is_same<const T, const U>::value);
  static_assert(std::is_same<const T, const V>::value);
  using E = expressions::TsrExpr<const T>;
  return Einsum::dot(E(A), E(B), E(C), world);
}

}  // namespace TiledArray::expressions

namespace TiledArray {

using expressions::dot;
using expressions::einsum;

template <typename T1, typename T2, typename P>
auto einsum(const std::string &expr, const DistArray<T1, P> &A,
            const DistArray<T2, P> &B, World &world = get_default_world()) {
  using ::Einsum::string::join;
  using ::Einsum::string::split2;

  struct {
    std::string A, B, C;
  } annot;

  {
    struct {
      std::string A, B, C;
    } outer;

    struct {
      std::string A, B, C;
    } inner;

    auto [ab, aC] = split2(expr, "->");
    std::tie(outer.C, inner.C) = split2(aC, ";");

    auto [aA, aB] = split2(ab, ",");
    std::tie(outer.A, inner.A) = split2(aA, ";");
    std::tie(outer.B, inner.B) = split2(aB, ";");

    auto combine = [](auto const &outer, auto const &inner) {
      return inner.empty() ? join(outer, ",")
                           : (join(outer, ",") + ";" + join(inner, ","));
    };

    annot.A = combine(outer.A, inner.A);
    annot.B = combine(outer.B, inner.B);
    annot.C = combine(outer.C, inner.C);
  }

  return einsum(A(annot.A), B(annot.B), annot.C, world);
}

/// Computes ternary tensor product whose result
/// is a scalar (a ternary dot product). Optimized for the case where
/// the arguments have common (Hadamard) indices.

/// \tparam T a Tile type
/// \tparam P a Policy type
/// \param expr a numpy-like annotation of the ternary product, e.g. "ij,ik,ijk"
/// will evaluate `(A("i,j")*B("i,k")).dot(C("i,j,k")).get()` \param A a
/// DistArray<T,P> object \param B a DistArray<T,P> object \param C a
/// DistArray<T,P> object \param world the World in which to compute the result
/// \return scalar result
/// \note if \p A , \p B , and \p C share indices computes `A*B` one slice at
/// a time and contracts with the corresponding slice `C`; thus storage of
/// `A*B` is eliminated
template <typename T, typename P>
auto dot(const std::string &expr, const DistArray<T, P> &A,
         const DistArray<T, P> &B, const DistArray<T, P> &C,
         World &world = get_default_world()) {
  namespace string = ::Einsum::string;
  auto [a, bc] = string::split2(expr, ",");
  auto [b, c] = string::split2(bc, ",");
  return dot(A(string::join(a, ",")), B(string::join(b, ",")),
             C(string::join(c, ",")), world);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED
