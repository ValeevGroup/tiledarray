#ifndef TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED
#define TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED

#include "TiledArray/conversions/make_array.h"
#include "TiledArray/dist_array.h"
#include "TiledArray/einsum/einsum_instrument.h"
#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/fwd.h"
#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"

#include <madness/world/thread.h>

#include <optional>

namespace TiledArray {
enum struct DeNest { True, False };
}

namespace TiledArray::detail {

/// Runtime toggle for the legacy per-Hadamard-slab sub-World evaluation of
/// general products in einsum.
///
/// einsum can route a general product (fused + contracted + free indices)
/// through the expression layer's native support (TensorProduct::General,
/// evaluated by the batched Summa: one task graph in one World, no per-slab
/// sub-Worlds) or through the legacy path (one MPI_Comm_split + sub-World +
/// fence per Hadamard slab). The expression route is the DEFAULT; set
/// TA_EINSUM_LEGACY_SUBWORLD in the environment (any non-empty value other
/// than "0"), or assign \c true to the reference returned by this function,
/// to force the legacy path. The legacy implementation is retained
/// indefinitely as a reference for differential testing
/// (TA_EINSUM_DIFFERENTIAL).
///
/// \note the two routes may legitimately differ on block-sparse data: the
/// legacy path derives the result shape from the harvested tile norms and
/// thus hard-zeroes sub-threshold result tiles, while the expression route
/// keeps them (its shape is the standard estimate-derived contraction
/// shape). Per the TA screening philosophy norms are trusted as genuine and
/// no implicit truncation is performed; call truncate() explicitly if the
/// tighter shape is desired.
inline bool &einsum_legacy_subworld() {
  static bool flag = [] {
    const char *e = std::getenv("TA_EINSUM_LEGACY_SUBWORLD");
    return e != nullptr && e[0] != char(0) && std::string_view(e) != "0";
  }();
  return flag;
}

/// Differential-testing mode for einsum's general products: when enabled
/// (TA_EINSUM_DIFFERENTIAL set to a non-empty value other than "0"), every
/// general product is evaluated by BOTH the expression route and the legacy
/// sub-World route; the two results are compared (squared norm of the
/// difference) and mismatches are reported to stderr with the contraction
/// annotation. The legacy result is returned.
inline bool &einsum_differential() {
  static bool flag = [] {
    const char *e = std::getenv("TA_EINSUM_DIFFERENTIAL");
    return e != nullptr && e[0] != '\0' && std::string_view(e) != "0";
  }();
  return flag;
}

}  // namespace TiledArray::detail

namespace TiledArray::Einsum {

using ::Einsum::index::small_vector;
using Range = ::Einsum::Range;
using RangeMap = ::Einsum::IndexMap<std::string, TiledRange1>;
using RangeProduct = ::Einsum::RangeProduct<Range, small_vector<size_t>>;

using ::Einsum::index::Index;
using ::Einsum::index::IndexMap;

using ::Einsum::index::Permutation;
using ::Einsum::index::permutation;

///
/// \tparam T A type that parameterizes ::Einsum::Index<T>.
///
/// This class makes it easier to work with indices involved in a binary
/// tensor multiplication. Also defines a canonical order of the indices.
///
/// Consider an arbitrary binary tensor multiplication annotated as:
///     A(a_1,...,a_m) * B(b_1,...,b_n) -> C(c_1,...,c_l)
/// Note that {c_1,...,c_l} is subset of ({a_1,...,a_m} union {b_1,...,b_n}).
///
/// We define following index types.
///     * Hadamard index: An index that annotates A, B, and C.
///     * Contracted index: An index that annotates A and B but not C.
///     * External index of A: An index that annotates A and C but not B.
///     * External index of B: An index that annotates B and C but not A.
///
/// Defining canonical index ordering.
///     * Hadamard indices are canonically ordered if they appear in the same
///       order in A's annotation.
///     * Contracted indices are canonically ordered if they appear in the same
///       order in A's annotation.
///     * External indices of A are canonically ordered if they appear in the
///       same order in A's annotation.
///     * External indices of B are canonically ordered if they appear in the
///       same order in B's annotation.
///     * Tensor A's indices are canonically ordered if Hadamard, external
///       indices of A, and contracted indices appear in that order and all
///       three index groups are themselves canonically ordered.
///     * Tensor B's indices are canonically ordered if Hadamard, external
///       indices of B, and contracted indices appear in that order and all
///       three index groups are themselves canonically ordered.
///     * Tensor C's indices are canonically ordered if Hadamard, external
///       indices of A and external indices of B appear in that order and all
///       three index groups are themselves canonically ordered.
///
/// Example: Consider the evaluation: A(i,j,p,a,b) * B(j,i,q,b,a) -> C(i,p,j,q).
///          - Hadamard indices: {i,j}
///          - External indices of A: {p}
///          - External indices of B: {q}
///          - Contracted indices: {a, b}
///          All index groups above are canonically ordered.
///          Writing C's indices in canonical order would give: {i,j,p,q}.
///
template <typename T>
class TensorOpIndices {
 public:
  using index_t = ::Einsum::Index<T>;

  TensorOpIndices(index_t const &ixA, index_t const &ixB, index_t const &ixC)
      : orig_indices_({ixA, ixB, ixC}) {
    hadamard_ = ixA & ixB & ixC;
    contracted_ = (ixA & ixB) - ixC;
    external_A_ = (ixA - ixB) & ixC;
    external_B_ = (ixB - ixA) & ixC;
  }

  [[nodiscard]] index_t const &ix_A() const { return orig_indices_[A]; }
  [[nodiscard]] index_t const &ix_B() const { return orig_indices_[B]; }
  [[nodiscard]] index_t const &ix_C() const { return orig_indices_[C]; }

  [[nodiscard]] index_t ix_A_canon() const {
    return hadamard() + external_A() + contracted();
  }

  [[nodiscard]] index_t ix_B_canon() const {
    return hadamard() + external_B() + contracted();
  }

  [[nodiscard]] index_t ix_C_canon() const {
    return hadamard() + external_A() + external_B();
  }

  [[nodiscard]] index_t const &hadamard() const { return hadamard_; }
  [[nodiscard]] index_t const &contracted() const { return contracted_; }
  [[nodiscard]] index_t const &external_A() const { return external_A_; }
  [[nodiscard]] index_t const &external_B() const { return external_B_; }

  [[nodiscard]] Permutation to_canon_A() const {
    return ::Einsum::index::permutation(ix_A(), ix_A_canon());
  }

  [[nodiscard]] Permutation to_canon_B() const {
    return ::Einsum::index::permutation(ix_B(), ix_B_canon());
  }

  [[nodiscard]] Permutation to_canon_C() const {
    return ::Einsum::index::permutation(ix_C(), ix_C_canon());
  }

 private:
  enum { A, B, C, ABC };
  std::array<index_t, ABC> orig_indices_;

  index_t hadamard_, contracted_, external_A_, external_B_;
};

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

// "Denested" companion of a ToT array: drops the inner-tile nesting, leaving
// a regular (non-nested) DistArray. For ToT inputs, the outer tile of the
// denested array is always TA::Tensor — nested inner-tile types (e.g.
// btas::Tensor) are only valid as the *innermost* tile and don't support the
// outer-tile operations einsum needs (permute/reshape/batch/range+lambda
// ctor). So for ToT we drop the inner tile and re-wrap its numeric type in
// TA::Tensor. For non-ToT inputs, the original "drop one level" behavior is
// preserved.
namespace detail_denested {
template <typename Array, typename Enabler = void>
struct denested {
  using type = DistArray<typename Array::value_type::value_type,
                         typename Array::policy_type>;
};
template <typename Array>
struct denested<Array,
                std::enable_if_t<TiledArray::detail::is_tensor_of_tensor_v<
                    typename Array::value_type>>> {
  using type = DistArray<
      TA::Tensor<typename Array::value_type::value_type::numeric_type>,
      typename Array::policy_type>;
};
}  // namespace detail_denested
template <typename Array>
using DeNestedArray = typename detail_denested::denested<Array>::type;

template <typename Array1, typename Array2>
using MaxNestedArray = std::conditional_t<(detail::nested_rank<Array2> >
                                           detail::nested_rank<Array1>),
                                          Array2, Array1>;

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

  if constexpr (TiledArray::is_arena_tensor_v<typename Tensor::value_type>) {
    // arena ToT: an inner cell is an 8-byte view into the outer tile's slab.
    // A plain std::copy of cells would leave `to` aliasing `from`'s slab --
    // dangling once `from` is gone. Build `to` as a fresh slab-backed tile
    // and deep-copy each replicated inner cell's element data.
    using inner_t = typename Tensor::value_type;
    using inner_range_t = typename inner_t::range_type;
    using elem_t = typename inner_t::value_type;
    const auto out_range = to.range();
    const std::size_t M = out_range.volume();
    auto range_fn = [&from, N](std::size_t ord) -> inner_range_t {
      const auto &src = from.data()[ord % N];
      return src.empty() ? inner_range_t{} : src.range();
    };
    to = detail::arena_outer_init<Tensor>(out_range, 1, range_fn,
                                          alignof(elem_t), /*zero_init=*/false);
    for (std::size_t ord = 0; ord < M; ++ord) {
      auto &dst = to.data()[ord];
      if (dst.empty()) continue;
      const auto &src = from.data()[ord % N];
      const elem_t *s = src.data();
      elem_t *d = dst.data();
      for (std::size_t k = 0; k < dst.size(); ++k) d[k] = s[k];
    }
    return;
  }

  for (auto i = 0; i < to.range().volume(); i += N)
    std::copy(from.begin(), from.end(), to.data() + i);
}

///
/// \brief This function is the @c DistArray counterpart of the function
///        @c replicate_tensor(TA::Tensor&, TA::Tensor const&).
///
/// \tparam Array
/// \param from The DistArray to be by-rank replicated.
/// \parama prepend_trng TiledRange1's in this argument will be prepended to the
///         `TiledRange` of the argument array.
/// \return An array whose rank is increased by `prepend_trng.rank()`.
/// \see `replicate_tensor`
///
template <typename Array,
          typename = std::enable_if_t<detail::is_array_v<Array>>>
auto replicate_array(Array from, TiledRange const &prepend_trng) {
  auto const result_rank = prepend_trng.rank() + rank(from);
  container::svector<TiledRange1> tr1s;
  tr1s.reserve(result_rank);
  for (auto const &r : prepend_trng) tr1s.emplace_back(r);
  for (auto const &r : from.trange()) tr1s.emplace_back(r);
  auto const result_trange = TiledRange(tr1s);

  from.make_replicated();
  auto &world = from.world();
  world.gop.fence();

  auto result = make_array<Array>(
      world, result_trange,
      [from, res_tr = result_trange, delta_rank = prepend_trng.rank()](
          auto &tile, auto const &res_rng) {
        using std::begin;
        using std::end;
        using std::next;

        typename Array::value_type repped(res_rng);
        auto res_coord_ix = res_tr.element_to_tile(res_rng.lobound());
        auto from_coord_ix = decltype(res_coord_ix)(
            next(begin(res_coord_ix), delta_rank), end(res_coord_ix));
        if (from.is_zero(from_coord_ix)) return typename Array::scalar_type{0};
        replicate_tensor(repped, from.find_local(from_coord_ix).get(false));
        tile = repped;
        return tile.norm();
      });

  if constexpr (std::is_same_v<typename Array::policy_type, SparsePolicy>)
    result.truncate();

  return result;
}

///
/// Given a rank-N tensor and a ∂-rank such that ∂ in [0,N), returns a new
/// rank-N' tensor (where N' = N - ∂) by summing over the ∂ ranks from the
/// end of the input tensor's range. For example, reduce_modes(A, 2) where
/// A.range().rank() == 5 will result into a new tensor (B) of rank-3 such that
/// B(i,j,k) = Σ_l Σ_m A(i,j,k,l,m).
///
/// \param orig Input Tensor.
/// \param dmodes Reduce this many modes from the end as implied in the
///               range of the input tensor.
/// \return Tensor with reduced rank.
///
template <typename T, typename... Ts>
auto reduce_modes(Tensor<T, Ts...> const &orig, size_t drank) {
  if (drank == 0) return orig;
  TA_ASSERT(orig.nbatch() == 1);
  auto const orig_rng = orig.range();
  TA_ASSERT(orig_rng.rank() > drank);

  auto const result_rng = [orig_rng, drank]() {
    container::vector<Range1> r1s;
    for (auto i = 0; i < orig_rng.rank() - drank; ++i)
      r1s.emplace_back(orig_rng.dim(i));
    return TA::Range(r1s);
  }();

  auto const delta_rng = [orig_rng, drank]() {
    container::vector<Range1> r1s;
    for (auto i = orig_rng.rank() - drank; i < orig_rng.rank(); ++i)
      r1s.emplace_back(orig_rng.dim(i));
    return TA::Range(r1s);
  }();

  auto const delta_vol = delta_rng.volume();

  auto reducer = [orig, delta_vol, delta_rng](auto const &ix) {
    auto orig_ix = ix;
    std::copy(delta_rng.lobound().begin(),  //
              delta_rng.lobound().end(),    //
              std::back_inserter(orig_ix));

    auto beg = orig.data() + orig.range().ordinal(orig_ix);
    auto end = beg + delta_vol;

    // cannot get it done this way: return std::reduce(beg, end);

    typename std::iterator_traits<decltype(beg)>::value_type sum{};
    for (; beg != end; ++beg) sum += *beg;
    return sum;
  };

  return Tensor<T, Ts...>(result_rng, reducer);
}

///
/// \param orig Input DistArray.
/// \param dmodes Reduce this many modes from the end as implied in the
///        tiled range of the input array.
/// \return Array with reduced rank.
/// \see reduce_modes(Tensor<T, Ts...>, size_t)
///
template <typename T, typename P>
auto reduce_modes(TA::DistArray<T, P> orig, size_t drank) {
  TA_ASSERT(orig.trange().rank() > drank);
  if (drank == 0) return orig;

  auto const result_trange = [orig, drank]() {
    container::svector<TiledRange1> tr1s;
    for (auto i = 0; i < (orig.trange().rank() - drank); ++i)
      tr1s.emplace_back(orig.trange().at(i));
    return TiledRange(tr1s);
  }();

  auto const delta_trange = [orig, drank]() {
    container::svector<TiledRange1> tr1s;
    for (auto i = orig.trange().rank() - drank; i < orig.trange().rank(); ++i)
      tr1s.emplace_back(orig.trange().at(i));
    return TiledRange(tr1s);
  }();

  orig.make_replicated();
  orig.world().gop.fence();

  auto make_tile = [orig, delta_trange, drank](auto &tile, auto const &rng) {
    using tile_type = std::remove_reference_t<decltype(tile)>;

    tile_type res(rng, typename tile_type::value_type{});

    bool all_summed_tiles_zeros{true};
    for (auto &&r : delta_trange.tiles_range()) {
      container::svector<TA::Range::index1_type> ix1s = rng.lobound();

      {
        auto d = delta_trange.make_tile_range(r);
        auto dlo = d.lobound();
        std::copy(dlo.begin(), dlo.end(), std::back_inserter(ix1s));
      }

      auto tix = orig.trange().element_to_tile(ix1s);
      if constexpr (std::is_same_v<P, SparsePolicy>)
        if (orig.is_zero(tix)) continue;
      auto got = orig.find_local(tix).get(false);

      res += reduce_modes(got, drank);
      all_summed_tiles_zeros = false;
    }

    if (all_summed_tiles_zeros)
      return typename std::remove_reference_t<decltype(tile)>::scalar_type{0};

    tile = res;
    return res.norm();
  };

  auto result =
      make_array<DistArray<T, P>>(orig.world(), result_trange, make_tile);
  if constexpr (std::is_same_v<P, SparsePolicy>) result.truncate();

  return result;
}

///
/// \tparam Ixs Iterable of indices.
/// \param map A map from the index type of \c Ixs to TiledRange1.
/// \param ixs Iterable of indices.
/// \return TiledRange object.
///
template <typename Ixs>
TiledRange make_trange(RangeMap const &map, Ixs const &ixs) {
  container::svector<TiledRange1> tr1s;
  tr1s.reserve(ixs.size());
  for (auto &&i : ixs) tr1s.emplace_back(map[i]);
  return TiledRange(tr1s);
}

}  // namespace

template <DeNest DeNestFlag = DeNest::False, typename ArrayA_, typename ArrayB_,
          typename... Indices>
auto einsum(expressions::TsrExpr<ArrayA_> A, expressions::TsrExpr<ArrayB_> B,
            std::tuple<Einsum::Index<std::string>, Indices...> cs,
            World &world) {
  // hotfix: process all preceding tasks before entering this code with many
  // blocking calls
  // TODO figure out why having free threads left after blocking MPI split
  //  still not enough to ensure progress
  const auto _ein_entry_t0 =
      detail::einsum_instrument_enabled() ? now() : time_point{};
  world.gop.fence();
  const std::int64_t _ein_entry_fence_ns =
      detail::einsum_instrument_enabled() ? duration_in_ns(_ein_entry_t0, now())
                                          : 0;

  using ArrayA = std::remove_cv_t<ArrayA_>;
  using ArrayB = std::remove_cv_t<ArrayB_>;

  using ArrayC =
      std::conditional_t<DeNestFlag == DeNest::True, DeNestedArray<ArrayA>,
                         MaxNestedArray<ArrayA, ArrayB>>;

  using ResultTensor = typename ArrayC::value_type;
  using ResultShape = typename ArrayC::shape_type;

  auto const &tnsrExprA = A;
  auto const &tnsrExprB = B;

  auto a = std::get<0>(Einsum::idx(A));
  auto b = std::get<0>(Einsum::idx(B));
  Einsum::Index<std::string> c = std::get<0>(cs);

  struct {
    std::string a, b, c;
    // Hadamard, external, internal indices for inner tensor
    Einsum::Index<std::string> A, B, C, h, e, i;
  } inner;

  if constexpr (IsArrayToT<ArrayA>) {
    inner.a = ";" + (std::string)std::get<1>(Einsum::idx(A));
    inner.A = std::get<1>(Einsum::idx(A));
  }

  if constexpr (IsArrayToT<ArrayB>) {
    inner.b = ";" + (std::string)std::get<1>(Einsum::idx(B));
    inner.B = std::get<1>(Einsum::idx(B));
  }

  if constexpr (std::tuple_size<decltype(cs)>::value == 2) {
    static_assert(IsArrayToT<ArrayC>);
    inner.c = ";" + (std::string)std::get<1>(cs);
    inner.C = std::get<1>(cs);
  }

  {
    inner.h = inner.A & inner.B & inner.C;
    inner.e = (inner.A ^ inner.B);
    inner.i = (inner.A & inner.B) - inner.h;
    if constexpr (IsArrayToT<ArrayC>)
      TA_ASSERT(!(inner.h && (inner.i || inner.e)) &&
                "General product between inner tensors not supported");
  }

  // einsum attribution profiler (TA_EINSUM_INSTRUMENT); no-op when disabled
  detail::EinsumCall _ein_call{(std::string)a + inner.a + " * " +
                               (std::string)b + inner.b + " -> " +
                               (std::string)c + inner.c};
  _ein_call.add(detail::EinsumBucket::EntryFence, _ein_entry_fence_ns);

  if constexpr (DeNestFlag == DeNest::True) {
    static_assert(detail::nested_rank<ArrayA> == detail::nested_rank<ArrayB> &&
                  detail::nested_rank<ArrayA> == 2);

    TA_ASSERT(!inner.C &&
              "Denested result cannot have inner-tensor annotation");

    TA_ASSERT(inner.i.size() == inner.A.size() &&
              inner.i.size() == inner.B.size() &&
              "Nested-rank-reduction only supported when the inner tensor "
              "ranks match on the arguments");

    //
    // Strategy. Consider A(ijpab;xy) * B(jiqba;yx) -> C(ipjq), inner xy fully
    // contracted. We reduce the contracted-outer indices ab and the contracted-
    // inner indices xy together with a single ToT x ToT -> ToT contraction
    // whose inner product is annotated to leave a *phantom unit* inner mode
    // (⊗₁) on the result, so the inner cell is a genuine (≥ order-1) unit
    // tensor rather than the unsupported order-0:
    //
    //   C0(ipjq; ⊗₁) = A(ijpab; xy) * B(jiqba; xy,⊗₁)
    //
    // ⊗₁ is appended to B's inner annotation only; B's inner *tensor* is
    // unchanged (⊗₁ is phantom unit -- ContEngine recognizes it and realizes
    // the inner product as a flat dot into a [1] cell, never requiring B to
    // physically carry the extra mode). Each [1] inner cell is then unwrapped
    // to a scalar. This never materializes the uncontracted product, and is
    // correct when an inner extent depends on a contracted-outer index.
    auto sum_tot_2_tos = [](auto const &tot) {
      using tot_t = std::remove_reference_t<decltype(tot)>;
      using numeric_type = typename tot_t::numeric_type;
      TA::Tensor<numeric_type> result(tot.range(), [tot](auto &&ix) {
        // unqualified `sum` so ADL finds the right overload for both
        // TA::Tensor inner (free fn in namespace TiledArray, calls .sum())
        // and btas::Tensor inner (free fn in namespace btas).
        if (!tot(ix).empty())
          return sum(tot(ix));
        else
          return numeric_type{};
      });
      return result;
    };

    // U+2297 CIRCLED TIMES + U+2081 SUBSCRIPT ONE: a reserved phantom-unit
    // inner annotator (see is_phantom_unit_label).
    const std::string phantom_unit = "⊗₁";

    auto a_annot = std::string(a) + inner.a;  // e.g. "ijpab;xy"
    auto b_annot =
        std::string(b) + inner.b + "," + phantom_unit;  // e.g. "jiqba;yx,⊗₁"
    auto c_annot = std::string(c) + ";" + phantom_unit;  // e.g. "ipjq;⊗₁"

    //  C0(c; ⊗₁) = A(a; inner.A) * B(b; inner.B,⊗₁)
    auto C0 = einsum(A.array()(a_annot), B.array()(b_annot), c_annot);

    //  unwrap unit-extent inner cells to scalars
    ArrayC C = TA::foreach<typename ArrayC::value_type>(
        C0, [sum_tot_2_tos](auto &out_tile, auto const &in_tile) {
          out_tile = sum_tot_2_tos(in_tile);
        });
    return C;

  } else {
    // these are "Hadamard" (fused) indices
    auto h = a & b & c;

    // external indices
    auto e = (a ^ b);

    // contracted indices
    auto i = (a & b) - h;

    //
    // *) Pure Hadamard indices: (h && !(i || e)) is true implies
    //   the evaluation can be delegated to the expression layer
    //   for distarrays of both nested and non-nested tensor tiles.
    // *) If no Hadamard indices are present (!h) the evaluation
    //    can be delegated to the expression layer.
    //
    if ((h && !(i || e))  // pure Hadamard
        || !h)            // no Hadamard
    {
      ArrayC C;
      C(std::string(c) + inner.c) = A * B;
      return C;
    }

    TA_ASSERT(e || h);

    auto range_map =
        (RangeMap(a, A.array().trange()) | RangeMap(b, B.array().trange()));

    // special Hadamard
    if (h.size() == a.size() || h.size() == b.size()) {
      TA_ASSERT(!i && e);
      bool const small_a = h.size() == a.size();
      auto const delta_trng = make_trange(range_map, e);
      std::string target_layout = std::string(c) + inner.c;
      ArrayC C;
      if (small_a) {
        auto temp = replicate_array(A.array(), delta_trng);
        std::string temp_layout = std::string(e) + "," + A.annotation();
        C(target_layout) = temp(temp_layout) * B;
      } else {
        auto temp = replicate_array(B.array(), delta_trng);
        std::string temp_layout = std::string(e) + "," + B.annotation();
        C(target_layout) = A * temp(temp_layout);
      }

      return C;
    }

    using ::Einsum::index::permutation;
    using TiledArray::Permutation;

    // Temporary sub-Worlds used by the generalized-contraction path below.
    // Declared before AB/C so it is destroyed *after* them: an ArrayTerm's
    // `.ei` member is a DistArray bound to one of these sub-Worlds, and
    // ~DistArray -> lazy_deleter dereferences that World. If a sub-World
    // outlived only by `worlds` were torn down first, that deref would hit a
    // dead World (e.g. while unwinding an exception thrown mid-contraction).
    std::vector<std::shared_ptr<World>> worlds;

    // RAII fencer: on normal exit and (critically) on exception unwind,
    // fence every live sub-World before it is destroyed. ~DistArray ->
    // lazy_deleter calls world.gop.lazy_sync(...) which enqueues a
    // lazy_sync_children task onto the sub-World's taskq; without a fence
    // those tasks survive into the global ThreadPool past the sub-World's
    // ~World, then trip ~WorldObject's `World::exists(&world)` assertion
    // when some later fence (e.g. an enclosing scope's fence run during
    // unwind) picks them up. Declared *after* `worlds` so it destructs
    // *before* `worlds` (LIFO); destructs *after* AB/C so it sees the
    // tasks they scheduled via lazy_deleter.
    //
    // One fence per sub-World is sufficient: lazy_deleter's fast path
    // skips lazy_sync when invoked from inside fence_impl's do_cleanup
    // (gated by `world.gop.is_in_do_cleanup()`), so the deferred-cleanup
    // path performs direct deletes rather than scheduling cross-rank
    // tasks. Tasks scheduled by *non*-deferred ~DistArray's (e.g. AB
    // during exception unwind) are drained by this fence's drain loop;
    // all participating ranks of a sub-World reach this RAII guard in
    // lockstep at function exit, so their lazy_sync handshakes match up.
    struct FenceSubWorldsOnExit {
      std::vector<std::shared_ptr<World>> &worlds_;
      ~FenceSubWorldsOnExit() {
        for (auto &w : worlds_) {
          if (!w) continue;
          try {
            w->gop.fence();
          } catch (...) {
          }
        }
      }
    } fence_subworlds_on_exit{worlds};

    std::tuple<ArrayTerm<ArrayA>, ArrayTerm<ArrayB>> AB{{A.array(), a},
                                                        {B.array(), b}};

    auto update_perm_and_indices = [&e = std::as_const(e),
                                    &i = std::as_const(i),
                                    &h = std::as_const(h)](auto &term) {
      auto ei = (e + i & term.idx);
      if (term.idx != h + ei) {
        term.permutation = permutation(term.idx, h + ei);
      }
      term.expr = ei;
    };

    std::invoke(update_perm_and_indices, std::get<0>(AB));
    std::invoke(update_perm_and_indices, std::get<1>(AB));

    // construct result, with "dense" DistArray; the array will be
    // reconstructred from local tiles later
    ArrayTerm<ArrayC> C = {ArrayC(world, TiledRange(range_map[c])), c};
    for (auto idx : e) {
      C.tiles *= Range(range_map[idx].tiles_range());
    }
    if (C.idx != h + e) {
      C.permutation = permutation(h + e, C.idx);
    }
    C.expr = e;

    using Index = Einsum::Index<size_t>;

    // this will collect local tiles of C.array, to be used to rebuild C.array
    std::vector<std::pair<Index, ResultTensor>> C_local_tiles;
    auto build_C_array = [&]() {
      C.array = make_array<ArrayC>(world, TiledRange(range_map[c]),
                                   C_local_tiles.begin(), C_local_tiles.end(),
                                   /* replicated = */ false);
    };

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

    if (!e) {  // hadamard reduction

      _ein_call.branch = "hadamard-reduction-local";
      const auto _ein_he_t0 = _ein_call.active ? now() : time_point{};

      auto &[A, B] = AB;
      TiledRange trange(range_map[i]);
      RangeProduct tiles;
      for (auto idx : i) {
        tiles *= Range(range_map[idx].tiles_range());
      }

      // the inner product can be either hadamard or a contraction
      using TensorT = typename decltype(A.array)::value_type::value_type;
      static_assert(
          std::is_same_v<TensorT,
                         typename decltype(A.array)::value_type::value_type>);
      constexpr bool is_tot = detail::is_tensor_v<TensorT>;
      // A non-owning view inner cell (e.g. ArenaTensor) has no value-returning
      // per-cell product; the legacy element-op path below cannot run for it.
      constexpr bool inner_is_view = TiledArray::is_tensor_view_v<TensorT>;
      auto element_hadamard_op =
          (is_tot && inner.h)
              ? std::make_optional(
                    [&inner, plan = detail::TensorHadamardPlan(inner.A, inner.B,
                                                               inner.C)](
                        auto const &l, auto const &r) -> TensorT {
                      if (l.empty() || r.empty()) return TensorT{};
                      return detail::tensor_hadamard(l, r, plan);
                    })
              : std::nullopt;
      auto element_contract_op =
          (is_tot && !inner.h)
              ? std::make_optional(
                    [&inner, plan = detail::TensorContractionPlan(
                                 inner.A, inner.B, inner.C)](
                        auto const &l, auto const &r) -> TensorT {
                      if (l.empty() || r.empty()) return TensorT{};
                      return detail::tensor_contract(l, r, plan);
                    })
              : std::nullopt;
      auto element_product_op = [&inner, &element_hadamard_op,
                                 &element_contract_op](
                                    auto const &l, auto const &r) -> TensorT {
        TA_ASSERT(inner.h ? element_hadamard_op.has_value()
                          : element_contract_op.has_value());
        return inner.h ? element_hadamard_op.value()(l, r)
                       : element_contract_op.value()(l, r);
      };

      auto pa = A.permutation;
      auto pb = B.permutation;
      auto arena_plan = detail::make_regime_a_arena_plan<ResultTensor>(
          A, B, inner, /*inner_perm=*/C.permutation);
      for (Index h : H.tiles) {
        auto const pc = C.permutation;
        auto const c = apply(pc, h);
        if (!C.array.is_local(c)) continue;
        size_t batch = 1;
        for (size_t i = 0; i < h.size(); ++i) {
          batch *= H.batch[i].at(h[i]);
        }
        if (detail::run_regime_a_arena(arena_plan, h, batch, A, B, C,
                                       C_local_tiles, tiles, trange))
          continue;
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
              if constexpr (inner_is_view) {
                // View inner cells (e.g. ArenaTensor) have no value-returning
                // per-cell product; only run_regime_a_arena can produce them.
                // Reaching this legacy path means the arena plan was inactive
                // -- typically a permuted inner contraction (see
                // TODO(arena-einsum-perm) in arena_einsum.h).
                TA_EXCEPTION(
                    "TA::einsum: ToT x ToT product with view inner cells "
                    "(e.g. ArenaTensor) is supported only via the regime-A "
                    "arena fast path, which was inactive for this expression "
                    "(likely a permuted inner contraction)");
              } else {
                auto aik = ai.batch(k);
                auto bik = bi.batch(k);
                auto vol = aik.total_size();
                TA_ASSERT(vol == bik.total_size());

                auto &el = tile({k});

                for (auto i = 0; i < vol; ++i)
                  add_to(el, element_product_op(aik.data()[i], bik.data()[i]));
              }
            } else if constexpr (!AreArraySame<ArrayA, ArrayB>) {
              auto aik = ai.batch(k);
              auto bik = bi.batch(k);
              auto vol = aik.total_size();
              TA_ASSERT(vol == bik.total_size());

              auto &el = tile({k});

              // Fused `el += inner_tensor * scalar` -- no scaled temporary
              // (axpy_to works in-place, so it also supports view inner
              // cells that cannot value-return a scaled tensor).
              using TiledArray::axpy_to;
              for (auto i = 0; i < vol; ++i)
                if constexpr (IsArrayToT<ArrayA>) {
                  axpy_to(el, aik.data()[i], bik.data()[i]);
                } else {
                  axpy_to(el, bik.data()[i], aik.data()[i]);
                }

            } else {
              auto hk = ai.batch(k).dot(bi.batch(k));
              tile({k}) += hk;
            }
          }
        }
        // data is stored as h1 h2 ... but all modes folded as 1 batch dim
        // first reshape to h = (h1 h2 ...)
        // n.b. can't just use shape = C.array.trange().tile(h)
        auto shape = apply_inverse(pc, C.array.trange().tile(c));
        tile = tile.reshape(shape);
        // then permute to target C layout c = (c1 c2 ...)
        if (pc) tile = tile.permute(pc);
        // and move to C_local_tiles
        C_local_tiles.emplace_back(std::move(c), std::move(tile));
      }

      _ein_call.add(detail::EinsumBucket::LocalKernel,
                    _ein_call.active ? duration_in_ns(_ein_he_t0, now()) : 0);

      build_C_array();

      return C.array;
    }  // end: hadamard reduction

    // generalized contraction

    _ein_call.branch = "generalized-subworld";
    const auto _ein_gen_t0 = _ein_call.active ? now() : time_point{};

    if constexpr (IsArrayToT<ArrayC>) {
      if (inner.C != inner.h + inner.e) {
        // when inner tensor permutation is non-trivial (could be potentially
        // elided by extending this function (@c einsum) to take into account
        // of inner tensor's permutations)
        _ein_call.branch = "generalized-inner-perm-recurse";
        auto temp_annot = std::string(c) + ";" + std::string(inner.h + inner.e);
        ArrayC temp = einsum(tnsrExprA, tnsrExprB,
                             Einsum::idx<ArrayC>(temp_annot), world);
        ArrayC result;
        result(std::string(c) + inner.c) = temp(temp_annot);
        return result;
      }
    }

    // Route the general product through the expression layer's native
    // support (TensorProduct::General -> batched Summa: one task graph in
    // one World, no per-slab sub-Worlds) unless the legacy path is forced
    // (see detail::einsum_legacy_subworld). The engine requires the
    // canonical (fused..., left-free..., right-free...) result layout; an
    // arbitrary einsum target is reached by a final permutation assignment.
    // N.B. the inner annotation is already canonical here (the non-trivial
    // inner permutation case recursed above).
    std::optional<ArrayC> expr_route_result;
    if (!detail::einsum_legacy_subworld() || detail::einsum_differential()) {
      _ein_call.branch = "generalized-expression";
      detail::EinsumTimer _t(_ein_call, detail::EinsumBucket::LocalKernel);

      TensorOpIndices<std::string> top(a, b, c);
      const auto c_canon = top.ix_C_canon();
      ArrayC result;
      result(std::string(c_canon) + inner.c) = tnsrExprA * tnsrExprB;
      if (c_canon == c) {
        expr_route_result = std::move(result);
      } else {
        ArrayC result_perm;
        result_perm(std::string(c) + inner.c) =
            result(std::string(c_canon) + inner.c);
        expr_route_result = std::move(result_perm);
      }
      if (!detail::einsum_differential()) return std::move(*expr_route_result);
      _ein_call.branch = "generalized-subworld";
    }

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

    _ein_call.add(detail::EinsumBucket::Setup,
                  _ein_call.active ? duration_in_ns(_ein_gen_t0, now()) : 0);

    // iterates over tiles of hadamard indices
    for (Index h : H.tiles) {
      auto &[A, B] = AB;
      _ein_call.add_slices(1);
      const auto _ein_cs0 = _ein_call.active ? now() : time_point{};
      auto own = A.own(h) || B.own(h);
      auto comm = madness::blocking_invoke(&SafeMPI::Intracomm::Split,
                                           world.mpi.comm(), own, world.rank());
      worlds.push_back(std::make_unique<World>(comm));
      _ein_call.add_subworld();
      _ein_call.add(detail::EinsumBucket::CommSplitWorld,
                    _ein_call.active ? duration_in_ns(_ein_cs0, now()) : 0);
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
      {
        detail::EinsumTimer _t(_ein_call, detail::EinsumBucket::Retile);
        std::invoke(retile, std::get<0>(AB));
        std::invoke(retile, std::get<1>(AB));
      }

      {
        detail::EinsumTimer _t(_ein_call, detail::EinsumBucket::ContractFence);
        C.ei(C.expr) = (A.ei(A.expr) * B.ei(B.expr)).set_world(*owners);
        A.ei.defer_deleter_to_next_fence();
        B.ei.defer_deleter_to_next_fence();
        A.ei = ArrayA();
        B.ei = ArrayB();
        // why omitting this fence leads to deadlock?
        owners->gop.fence();
      }
      {
        detail::EinsumTimer _t(_ein_call, detail::EinsumBucket::Harvest);
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
          C_local_tiles.emplace_back(std::move(c), std::move(tile));
        }
      }
      // mark for lazy deletion
      C.ei = ArrayC();
    }

    {
      detail::EinsumTimer _t(_ein_call, detail::EinsumBucket::Teardown);
      build_C_array();

      for (auto &w : worlds) {
        w->gop.fence();
      }
    }

    if (expr_route_result) {
      // differential mode: compare the expression route against the legacy
      // result
      const std::string c_annot = std::string(c) + inner.c;
      ArrayC diff;
      diff(c_annot) = (*expr_route_result)(c_annot)-C.array(c_annot);
      const double d2 = diff(c_annot).squared_norm().get();
      const double ref2 = C.array(c_annot).squared_norm().get();
      if (!(d2 <= 1e-20 * std::max(ref2, 1.0))) {
        if (world.rank() == 0)
          std::cerr << "!! einsum DIFFERENTIAL MISMATCH: "
                    << (std::string)a + inner.a << " * "
                    << (std::string)b + inner.b << " -> " << c_annot
                    << " : diff2 = " << d2 << ", legacy2 = " << ref2
                    << std::endl;
        // per-tile forensics: compare tile norms of the two routes
        auto tile_norm2 = [](auto const &tile) -> double {
          using TileT =
              std::remove_cv_t<std::remove_reference_t<decltype(tile)>>;
          double n2 = 0;
          if constexpr (TiledArray::detail::is_tensor_of_tensor_v<TileT>) {
            for (std::size_t o = 0; o < tile.range().volume(); ++o) {
              auto const &cell = tile.data()[o];
              if (cell.empty()) continue;
              for (std::size_t e = 0; e < cell.range().volume(); ++e)
                n2 += std::abs(cell.data()[e]) * std::abs(cell.data()[e]);
            }
          } else {
            for (std::size_t e = 0; e < tile.range().volume() * tile.nbatch();
                 ++e)
              n2 += std::abs(tile.data()[e]) * std::abs(tile.data()[e]);
          }
          return n2;
        };
        const auto ntiles = C.array.trange().tiles_range().volume();
        std::size_t n_diff = 0, n_expr_zero = 0, n_legacy_zero = 0,
                    n_printed = 0;
        for (std::size_t ord = 0; ord < ntiles; ++ord) {
          const bool ez = expr_route_result->is_zero(ord);
          const bool lz = C.array.is_zero(ord);
          double en2 =
              ez ? 0.0 : tile_norm2(expr_route_result->find(ord).get());
          double ln2 = lz ? 0.0 : tile_norm2(C.array.find(ord).get());
          const double dd = std::abs(en2 - ln2);
          if (dd <= 1e-14 * std::max(std::max(en2, ln2), 1.0)) continue;
          ++n_diff;
          if (en2 == 0.0) ++n_expr_zero;
          if (ln2 == 0.0) ++n_legacy_zero;
          if (world.rank() == 0 && n_printed < 8) {
            ++n_printed;
            std::cerr << "    tile " << ord << " ("
                      << C.array.trange().tiles_range().idx(ord)
                      << "): expr_n2 = " << en2 << ", legacy_n2 = " << ln2
                      << std::endl;
          }
        }
        if (world.rank() == 0)
          std::cerr << "    summary: " << n_diff << "/" << ntiles
                    << " tiles differ by norm; expr-zero " << n_expr_zero
                    << ", legacy-zero " << n_legacy_zero << std::endl;
      }
    }

    return C.array;
  }
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
template <DeNest DeNestFlag = DeNest::False, typename T, typename U,
          typename... Indices>
auto einsum(expressions::TsrExpr<T> A, expressions::TsrExpr<U> B,
            const std::string &cs, World &world = get_default_world()) {
  using ECT = expressions::TsrExpr<const T>;
  using ECU = expressions::TsrExpr<const U>;

  using ResultExprT =
      std::conditional_t<DeNestFlag == DeNest::True, Einsum::DeNestedArray<T>,
                         Einsum::MaxNestedArray<T, U>>;

  return Einsum::einsum<DeNestFlag>(ECT(A), ECU(B),
                                    Einsum::idx<ResultExprT>(cs), world);
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

template <DeNest DeNestFlag = DeNest::False, typename T1, typename T2,
          typename P>
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

  return einsum<DeNestFlag>(A(annot.A), B(annot.B), annot.C, world);
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
