#ifndef TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED
#define TILEDARRAY_EINSUM_TILEDARRAY_H__INCLUDED

#include "TiledArray/conversions/make_array.h"
#include "TiledArray/dist_array.h"
#include "TiledArray/einsum/index.h"
#include "TiledArray/einsum/range.h"
#include "TiledArray/expressions/fwd.h"
#include "TiledArray/fwd.h"
#include "TiledArray/tiled_range.h"
#include "TiledArray/tiled_range1.h"

namespace TiledArray {
enum struct DeNest { True, False };
}

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

template <typename Array>
using DeNestedArray = DistArray<typename Array::value_type::value_type,
                                typename Array::policy_type>;

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
  world.gop.fence();

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
    // Illustration of steps by an example.
    //
    // Consider the evaluation: A(ijpab;xy) * B(jiqba;yx) -> C(ipjq).
    //
    // Note for the outer indices:
    //      - Hadamard: 'ij'
    //      - External A: 'p'
    //      - External B: 'q'
    //      - Contracted: 'ab'
    //
    // Now C is evaluated in the following steps.
    //  Step I:   A(ijpab;xy) * B(jiqba;yx) -> C0(ijpqab;xy)
    //  Step II:  C0(ijpqab;xy) -> C1(ijpqab)
    //  Step III: C1(ijpqab) -> C2(ijpq)
    //  Step IV:  C2(ijpq) -> C(ipjq)

    auto sum_tot_2_tos = [](auto const &tot) {
      using tot_t = std::remove_reference_t<decltype(tot)>;
      typename tot_t::value_type result(tot.range(), [tot](auto &&ix) {
        if (!tot(ix).empty())
          return tot(ix).sum();
        else
          return typename tot_t::numeric_type{};
      });
      return result;
    };

    auto const oixs = TensorOpIndices(a, b, c);

    struct {
      std::string C0, C1, C2;
    } const Cn_annot{
        std::string(oixs.ix_C_canon() + oixs.contracted()) + inner.a,
        {oixs.ix_C_canon() + oixs.contracted()},
        {oixs.ix_C_canon()}};

    //  Step I:   A(ijpab;xy) * B(jiqba;yx) -> C0(ijpqab;xy)
    auto C0 = einsum(A, B, Cn_annot.C0);

    //  Step II:  C0(ijpqab;xy) -> C1(ijpqab)
    auto C1 = TA::foreach<typename ArrayC::value_type>(
        C0, [sum_tot_2_tos](auto &out_tile, auto const &in_tile) {
          out_tile = sum_tot_2_tos(in_tile);
        });

    //  Step III: C1(ijpqab) -> C2(ijpq)
    auto C2 = reduce_modes(C1, oixs.contracted().size());

    //  Step IV:  C2(ijpq) -> C(ipjq)
    ArrayC C;
    C(c) = C2(Cn_annot.C2);
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
      for (Index h : H.tiles) {
        auto const pc = C.permutation;
        auto const c = apply(pc, h);
        if (!C.array.is_local(c)) continue;
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

              for (auto i = 0; i < vol; ++i)
                el.add_to(element_product_op(aik.data()[i], bik.data()[i]));

            } else if constexpr (!AreArraySame<ArrayA, ArrayB>) {
              auto aik = ai.batch(k);
              auto bik = bi.batch(k);
              auto vol = aik.total_size();
              TA_ASSERT(vol == bik.total_size());

              auto &el = tile({k});

              for (auto i = 0; i < vol; ++i)
                if constexpr (IsArrayToT<ArrayA>) {
                  el.add_to(aik.data()[i].scale(bik.data()[i]));
                } else {
                  el.add_to(bik.data()[i].scale(aik.data()[i]));
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

      build_C_array();

      return C.array;
    }  // end: hadamard reduction

    // generalized contraction

    if constexpr (IsArrayToT<ArrayC>) {
      if (inner.C != inner.h + inner.e) {
        // when inner tensor permutation is non-trivial (could be potentially
        // elided by extending this function (@c einsum) to take into account
        // of inner tensor's permutations)
        auto temp_annot = std::string(c) + ";" + std::string(inner.h + inner.e);
        ArrayC temp = einsum(tnsrExprA, tnsrExprB,
                             Einsum::idx<ArrayC>(temp_annot), world);
        ArrayC result;
        result(std::string(c) + inner.c) = temp(temp_annot);
        return result;
      }
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

    std::vector<std::shared_ptr<World>> worlds;

    // iterates over tiles of hadamard indices
    for (Index h : H.tiles) {
      auto &[A, B] = AB;
      auto own = A.own(h) || B.own(h);
      auto comm = madness::blocking_invoke(&SafeMPI::Intracomm::Split,
                                           world.mpi.comm(), own, world.rank());
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
        C_local_tiles.emplace_back(std::move(c), std::move(tile));
      }
      // mark for lazy deletion
      C.ei = ArrayC();
    }

    build_C_array();

    for (auto &w : worlds) {
      w->gop.fence();
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
