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

#ifndef TILEDARRAY_TEST_TOT_ARRAY_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_TOT_ARRAY_FIXTURE_H__INCLUDED
#include <TiledArray/einsum/tiledarray.h>
#include <tiledarray.h>
#include "unit_test_config.h"
#ifdef TILEDARRAY_HAS_BTAS
#include <TiledArray/conversions/btas.h>
#include <TiledArray/external/btas.h>
#include <btas/generic/contract.h>
#endif

/* Notes:
 *
 * This test suite currently does not test:
 * - wait_for_lazy_cleanup (either overload)
 * - id() (documentation suggests it's not part of the public API)
 * - elements (it's deprecated)
 * - get_world (it's deprecated)
 * - get_pmap (it's deprecated)
 * - register_set_notifier
 */

using namespace TiledArray;

// These are all of the template parameters we are going to test over
using test_params =
    boost::mpl::list<std::tuple<int, Tensor<Tensor<int>>>,
                     std::tuple<float, Tensor<Tensor<float>>>,
                     std::tuple<double, Tensor<Tensor<double>>>
#ifdef TILEDARRAY_HAS_BTAS
                     ,
                     std::tuple<int, Tensor<btas::Tensor<int, Range>>>,
                     std::tuple<float, Tensor<btas::Tensor<float, Range>>>,
                     std::tuple<double, Tensor<btas::Tensor<double, Range>>>
//    ,std::tuple<int, btas::Tensor<btas::Tensor<int, Range>, Range>>,
//    std::tuple<float, btas::Tensor<btas::Tensor<float, Range>, Range>>,
//    std::tuple<double, btas::Tensor<btas::Tensor<double, Range>, Range>>
//    ,std::tuple<int, Tile<btas::Tensor<btas::Tensor<int, Range>, Range>>>,
//    std::tuple<float, Tile<btas::Tensor<btas::Tensor<float, Range>, Range>>>,
//    std::tuple<double, Tile<btas::Tensor<btas::Tensor<double, Range>, Range>>>
#endif
                     >;

// These typedefs unpack the unit test template parameter
//{
template <typename TupleElementType>
using scalar_type = std::tuple_element_t<0, TupleElementType>;

template <typename TupleElementType>
using tile_type = std::tuple_element_t<1, TupleElementType>;

template <typename>
using policy_type = DensePolicy;
//}

// The type of a DistArray consistent with the unit test template parameter
template <typename TupleElementType>
using tensor_type =
    DistArray<tile_type<TupleElementType>, policy_type<TupleElementType>>;

// Type of the object storing the tiling
template <typename TupleElementType>
using trange_type = typename tensor_type<TupleElementType>::trange_type;

// Type of the inner tile
template <typename TupleElementType>
using inner_type = typename tile_type<TupleElementType>::value_type;

// Type of an input archive
using input_archive_type = madness::archive::BinaryFstreamInputArchive;

// Type of an output archive
using output_archive_type = madness::archive::BinaryFstreamOutputArchive;

enum class ShapeComp { True, False };

template <typename TensorT,
          std::enable_if_t<TA::detail::is_tensor_v<TensorT>, bool> = true>
auto random_tensor(TA::Range const& rng) {
  using NumericT = typename TensorT::numeric_type;
  TensorT result{rng};

  std::generate(/*std::execution::par, */
                result.begin(), result.end(),
                TA::detail::MakeRandom<NumericT>::generate_value);
  return result;
}

template <typename TensorT>
auto random_tensor(std::initializer_list<size_t> const& extents) {
  auto lobounds = TA::container::svector<size_t>(extents.size(), 0);
  return random_tensor<TensorT>(TA::Range{lobounds, extents});
}

//
// note: all the inner tensors (elements of the outer tensor)
//       have the same @c inner_rng
//
template <
    typename TensorT,
    std::enable_if_t<TA::detail::is_tensor_of_tensor_v<TensorT>, bool> = true>
auto random_tensor(TA::Range const& outer_rng, TA::Range const& inner_rng) {
  using InnerTensorT = typename TensorT::value_type;
  TensorT result{outer_rng};

  std::generate(/*std::execution::par,*/
                result.begin(), result.end(), [inner_rng]() {
                  return random_tensor<InnerTensorT>(inner_rng);
                });

  return result;
}

template <typename TensorT>
auto random_tensor(TA::Range const& outer_rng,
                   std::initializer_list<size_t> const& inner_extents) {
  TA::container::svector<size_t> lobounds(inner_extents.size(), 0);
  return random_tensor<TensorT>(outer_rng, TA::Range(lobounds, inner_extents));
}

///
/// \tparam Array The type of DistArray to be generated. Cannot be cv-qualified
/// or reference type.
/// \tparam Args TA::Range type for inner tensor if the tile type of the result
/// is a tensor-of-tensor.
/// \param trange The TiledRange of the result DistArray.
/// \param args Either exactly one TA::Range type when the tile type of Array is
/// tensor-of-tensor or nothing.
/// \return Returns a DistArray of type Array whose elements are randomly
/// generated.
/// @note:
/// - Although DistArrays with Sparse policy can be generated all of their
///   tiles are initialized with random values -- technically the returned value
///   is dense.
/// - In case of arrays with tensor-of-tensor tiles, all the inner tensors have
///   the same rank and the same extent of corresponding modes.
///
template <
    typename Array, typename... Args,
    typename =
        std::void_t<typename Array::value_type, typename Array::policy_type>,
    std::enable_if_t<TA::detail::is_nested_tensor_v<typename Array::value_type>,
                     bool> = true>
auto random_array(TA::TiledRange const& trange, Args const&... args) {
  static_assert(
      (sizeof...(Args) == 0 &&
       TA::detail::is_tensor_v<typename Array::value_type>) ||
      (sizeof...(Args) == 1) &&
          (TA::detail::is_tensor_of_tensor_v<typename Array::value_type>));

  using TensorT = typename Array::value_type;
  using PolicyT = typename Array::policy_type;

  auto make_tile_meta = [](auto&&... args) {
    return [=](TensorT& tile, TA::Range const& rng) {
      tile = random_tensor<TensorT>(rng, args...);
      if constexpr (std::is_same_v<TA::SparsePolicy, PolicyT>)
        return tile.norm();
    };
  };

  return TA::make_array<Array>(TA::get_default_world(), trange,
                               make_tile_meta(args...));
}

template <typename Array, typename... Args>
auto random_array(std::initializer_list<std::initializer_list<size_t>> trange,
                  Args&&... args) {
  return random_array<Array>(TA::TiledRange(trange),
                             std::forward<Args>(args)...);
}

///
/// Succinctly call TA::detail::tensor_contract
///
/// \tparam T TA::Tensor type.
/// \param einsum_annot Example annot: 'ik,kj->ij', when @c A is annotated by
/// 'i' and 'k' for its two modes, and @c B is annotated by 'k' and 'j' for the
/// same. The result tensor is rank-2 as well and its modes are annotated by 'i'
/// and 'j'.
/// \return Tensor contraction result.
///
template <typename T, std::enable_if_t<TA::detail::is_tensor_v<T>, bool> = true>
auto tensor_contract(std::string const& einsum_annot, T const& A, T const& B) {
  using ::Einsum::string::split2;
  auto [ab, aC] = split2(einsum_annot, "->");
  auto [aA, aB] = split2(ab, ",");

  return TA::detail::tensor_contract(A, aA, B, aB, aC);
}

using PartialPerm = TA::container::svector<std::pair<size_t, size_t>>;

template <typename T>
PartialPerm partial_perm(::Einsum::index::Index<T> const& from,
                         ::Einsum::index::Index<T> const& to) {
  PartialPerm result;
  for (auto i = 0; i < from.size(); ++i)
    if (auto found = to.find(from[i]); found != to.end())
      result.emplace_back(i, std::distance(to.begin(), found));
  return result;
}

template <typename T, typename = std::enable_if_t<
                          TA::detail::is_random_access_container_v<T>>>
void apply_partial_perm(T& to, T const& from, PartialPerm const& p) {
  for (auto [f, t] : p) {
    TA_ASSERT(f < from.size() && t < to.size() && "Invalid permutation used");
    to[t] = from[f];
  }
}

enum struct TensorProduct { General, Dot, Invalid };

struct ProductSetup {
  TensorProduct product_type{TensorProduct::Invalid};

  PartialPerm
      // - {<k,v>} index at kth position in C appears at vth position in A
      //   and so on...
      // - {<k,v>} is sorted by k
      C_to_A,
      C_to_B,
      I_to_A,  // 'I' implies for contracted indices
      I_to_B;
  size_t       //
      rank_A,  //
      rank_B,
      rank_C,  //
      rank_H,
      rank_E,  //
      rank_I;

  ProductSetup() = default;

  template <typename T,
            typename = std::enable_if_t<TA::detail::is_annotation_v<T>>>
  ProductSetup(T const& aA, T const& aB, T const& aC) {
    using Indices = ::Einsum::index::Index<typename T::value_type>;

    struct {
      // A, B, C tensor indices
      // H, E, I Hadamard, external, and internal indices
      Indices A, B, C, H, E, I;
    } const ixs{Indices(aA),     Indices(aB),
                Indices(aC),     (ixs.A & ixs.B & ixs.C),
                (ixs.A ^ ixs.B), ((ixs.A & ixs.B) - ixs.H)};

    rank_A = ixs.A.size();
    rank_B = ixs.B.size();
    rank_C = ixs.C.size();
    rank_H = ixs.H.size();
    rank_E = ixs.E.size();
    rank_I = ixs.I.size();

    C_to_A = partial_perm(ixs.C, ixs.A);
    C_to_B = partial_perm(ixs.C, ixs.B);
    I_to_A = partial_perm(ixs.I, ixs.A);
    I_to_B = partial_perm(ixs.I, ixs.B);

    using TP = decltype(product_type);

    if (rank_A + rank_B != 0 && rank_C != 0)
      product_type = TP::General;
    else if (rank_A == rank_B && rank_B != 0 && rank_C == 0)
      product_type = TP::Dot;
    else
      product_type = TP::Invalid;
  }

  template <typename ArrayLike,
            typename = std::enable_if_t<
                TA::detail::is_annotation_v<typename ArrayLike::value_type>>>
  ProductSetup(ArrayLike const& arr)
      : ProductSetup(std::get<0>(arr), std::get<1>(arr), std::get<2>(arr)) {}

  [[nodiscard]] bool valid() const noexcept {
    return product_type != TensorProduct::Invalid;
  }
};

///
/// Example: To represent A("ik;ac") * B("kj;cb") -> C("ij;ab")
///
/// Method 1:
/// ---
/// construct with a single argument std::string("ij;ac,kj;cb->ij;ab");
/// - the substring "<outer_indices>;<inner_indices>"
///   annotates a single object (DistArray, Tensor etc.)
/// - "<A_indices>,<B_indices>" implies two distinct annotations (for A and B)
///   separated by a comma
/// - the right hand side of '->' annotates the result.
/// - Note: the only use of comma is to separate A's and B's annotations.
///
/// Method 2:
/// ---
/// construct with three arguments:
///   std::string("i,k;a,c"), std::string("k,j;c,b"), std::string("i,j;a,b")
///   - Note the use of comma.
///
class OuterInnerSetup {
  ProductSetup outer_;
  ProductSetup inner_;

 public:
  OuterInnerSetup(std::string const& annot) {
    using ::Einsum::string::split2;
    using Ix = ::Einsum::index::Index<char>;

    enum { A, B, C };
    std::array<std::string, 3> O;
    std::array<std::string, 3> I;

    auto [ab, aC] = split2(annot, "->");
    std::tie(O[C], I[C]) = split2(aC, ";");

    auto [aA, aB] = split2(ab, ",");
    std::tie(O[A], I[A]) = split2(aA, ";");
    std::tie(O[B], I[B]) = split2(aB, ";");
    outer_ = ProductSetup(Ix(O[A]), Ix(O[B]), Ix(O[C]));
    inner_ = ProductSetup(Ix(I[A]), Ix(I[B]), Ix(I[C]));
  }

  template <int N>
  OuterInnerSetup(const char (&s)[N]) : OuterInnerSetup{std::string(s)} {}

  OuterInnerSetup(std::string const& annotA, std::string const& annotB,
                  std::string const& annotC) {
    using ::Einsum::string::split2;
    using Ix = ::Einsum::index::Index<std::string>;

    enum { A, B, C };
    std::array<std::string, 3> O;
    std::array<std::string, 3> I;
    std::tie(O[A], I[A]) = split2(annotA, ";");
    std::tie(O[B], I[B]) = split2(annotB, ";");
    std::tie(O[C], I[C]) = split2(annotC, ";");
    outer_ = ProductSetup(Ix(O[A]), Ix(O[B]), Ix(O[C]));
    inner_ = ProductSetup(Ix(I[A]), Ix(I[B]), Ix(I[C]));
  }

  [[nodiscard]] auto const& outer() const noexcept { return outer_; }

  [[nodiscard]] auto const& inner() const noexcept { return inner_; }
};

namespace {

auto make_perm(PartialPerm const& pp) {
  TA::container::svector<TA::Permutation::index_type> p(pp.size());
  for (auto [k, v] : pp) p[k] = v;
  return TA::Permutation(p);
}

template <typename Result, typename Tensor, typename... Setups,
          typename = std::enable_if_t<TA::detail::is_nested_tensor_v<Tensor>>>
inline Result general_product(Tensor const& t, typename Tensor::numeric_type s,
                              ProductSetup const& setup,
                              Setups const&... args) {
  static_assert(std::is_same_v<Result, Tensor>);
  static_assert(sizeof...(args) == 0,
                "To-Do: Only scalar times once-nested tensor supported now");
  return t.scale(s, make_perm(setup.C_to_A).inv());
}

template <typename Result, typename Tensor, typename... Setups,
          typename = std::enable_if_t<TA::detail::is_nested_tensor_v<Tensor>>>
inline Result general_product(typename Tensor::numeric_type s, Tensor const& t,
                              ProductSetup const& setup,
                              Setups const&... args) {
  static_assert(std::is_same_v<Result, Tensor>);
  static_assert(sizeof...(args) == 0,
                "To-Do: Only scalar times once-nested tensor supported now");
  return t.scale(s, make_perm(setup.C_to_B).inv());
}

}  // namespace

template <
    typename Result, typename TensorA, typename TensorB, typename... Setups,
    typename =
        std::enable_if_t<TA::detail::is_nested_tensor_v<TensorA, TensorB>>>
Result general_product(TensorA const& A, TensorB const& B,
                       ProductSetup const& setup, Setups const&... args) {
  using TA::detail::max_nested_rank;
  using TA::detail::nested_rank;

  static_assert(std::is_same_v<typename TensorA::numeric_type,
                               typename TensorB::numeric_type>);

  static_assert(max_nested_rank<TensorA, TensorB> == sizeof...(args) + 1);

  TA_ASSERT(setup.valid());

  constexpr bool is_tot = max_nested_rank<TensorA, TensorB> > 1;

  if constexpr (std::is_same_v<Result, typename TensorA::numeric_type>) {
    //
    // tensor dot product evaluation
    // T * T -> scalar
    // ToT * ToT -> scalar
    //
    static_assert(nested_rank<TensorA> == nested_rank<TensorB>);

    TA_ASSERT(setup.rank_C == 0 &&
              "Attempted to evaluate dot product when the product setup does "
              "not allow");

    Result result{};

    for (auto&& ix_A : A.range()) {
      TA::Range::index_type ix_B(setup.rank_B, 0);
      apply_partial_perm(ix_B, ix_A, setup.I_to_B);

      if constexpr (is_tot) {
        auto const& lhs = A(ix_A);
        auto const& rhs = B(ix_B);
        result += general_product<Result>(lhs, rhs, args...);
      } else
        result += A(ix_A) * B(ix_B);
    }

    return result;
  } else {
    //
    // general product:
    // T * T -> T
    // ToT * T -> ToT
    // ToT * ToT -> ToT
    // ToT * ToT -> T
    //

    static_assert(nested_rank<Result> <= max_nested_rank<TensorA, TensorB>,
                  "Tensor product not supported with increased nested rank in "
                  "the result");

    // creating the contracted TA::Range
    TA::Range const rng_I = [&setup, &A, &B]() {
      TA::container::svector<TA::Range1> rng1_I(setup.rank_I, TA::Range1{});
      for (auto [f, t] : setup.I_to_A)
        // I_to_A implies I[f] == A[t]
        rng1_I[f] = A.range().dim(t);

      return TA::Range(rng1_I);
    }();

    // creating the target TA::Range.
    TA::Range const rng_C = [&setup, &A, &B]() {
      TA::container::svector<TA::Range1> rng1_C(setup.rank_C, TA::Range1{0, 0});
      for (auto [f, t] : setup.C_to_A)
        // C_to_A implies C[f] = A[t]
        rng1_C[f] = A.range().dim(t);

      for (auto [f, t] : setup.C_to_B)
        // C_to_B implies C[f] = B[t]
        rng1_C[f] = B.range().dim(t);

      auto zero_r1 = [](TA::Range1 const& r) { return r == TA::Range1{0, 0}; };

      TA_ASSERT(std::none_of(rng1_C.begin(), rng1_C.end(), zero_r1));

      return TA::Range(rng1_C);
    }();

    Result C{rng_C};

    // do the computation
    for (auto ix_C : rng_C) {
      // finding corresponding indices of A, and B.
      TA::Range::index_type ix_A(setup.rank_A, 0), ix_B(setup.rank_B, 0);
      apply_partial_perm(ix_A, ix_C, setup.C_to_A);
      apply_partial_perm(ix_B, ix_C, setup.C_to_B);

      if (setup.rank_I == 0) {
        if constexpr (is_tot) {
          C(ix_C) = general_product<typename Result::value_type>(
              A(ix_A), B(ix_B), args...);
        } else {
          TA_ASSERT(!(ix_A.empty() && ix_B.empty()));
          C(ix_C) = ix_A.empty()   ? B(ix_B)
                    : ix_B.empty() ? A(ix_B)
                                   : A(ix_A) * B(ix_B);
        }
      } else {
        typename Result::value_type temp{};
        for (auto ix_I : rng_I) {
          apply_partial_perm(ix_A, ix_I, setup.I_to_A);
          apply_partial_perm(ix_B, ix_I, setup.I_to_B);
          if constexpr (is_tot)
            temp += general_product<typename Result::value_type>(
                A(ix_A), B(ix_B), args...);
          else {
            TA_ASSERT(!(ix_A.empty() || ix_B.empty()));
            temp += A(ix_A) * B(ix_B);
          }
        }
        C(ix_C) = temp;
      }
    }

    return C;
  }
}

template <typename TileC, typename TileA, typename TileB, typename... Setups>
auto general_product(TA::DistArray<TileA, TA::DensePolicy> A,
                     TA::DistArray<TileB, TA::DensePolicy> B,
                     ProductSetup const& setup, Setups const&... args) {
  using TA::detail::max_nested_rank;
  using TA::detail::nested_rank;
  static_assert(nested_rank<TileC> <= max_nested_rank<TileA, TileB>);
  static_assert(nested_rank<TileC> != 0);
  TA_ASSERT(setup.product_type == TensorProduct::General);

  auto& world = TA::get_default_world();

  A.make_replicated();
  B.make_replicated();
  world.gop.fence();

  TA::Tensor<TileA> tensorA{A.trange().tiles_range()};
  for (auto&& ix : tensorA.range()) tensorA(ix) = A.find_local(ix).get(false);

  TA::Tensor<TileB> tensorB{B.trange().tiles_range()};
  for (auto&& ix : tensorB.range()) tensorB(ix) = B.find_local(ix).get(false);

  auto result_tensor = general_product<TA::Tensor<TileC>>(
      tensorA, tensorB, setup, setup, args...);

  TA::TiledRange result_trange;
  {
    auto const rank = result_tensor.range().rank();
    auto const result_range = result_tensor.range();

    TA::container::svector<TA::container::svector<size_t>> tr1s(rank, {0});

    TA::container::svector<size_t> const ix_hi(result_range.upbound());
    for (auto d = 0; d < rank; ++d) {
      TA::container::svector<size_t> ix(result_range.lobound());
      for (auto& i = ix[d]; i < ix_hi[d]; ++i) {
        auto const& elem_tensor = result_tensor(ix);
        auto& tr1 = tr1s[d];
        tr1.emplace_back(tr1.back() + elem_tensor.range().extent(d));
      }
    }

    TA::container::svector<TA::TiledRange1> tr1s_explicit;
    tr1s_explicit.reserve(tr1s.size());
    for (auto const& v : tr1s) tr1s_explicit.emplace_back(v.begin(), v.end());

    result_trange = TA::TiledRange(tr1s_explicit);
  }

  TA::DistArray<TileC, TA::DensePolicy> C(world, result_trange);

  for (auto it : C) {
    if (C.is_local(it.index())) it = result_tensor(it.index());
  }
  return C;
}

template <typename TileA, typename TileB, typename... Setups>
auto general_product(TA::DistArray<TileA, TA::DensePolicy> A,
                     TA::DistArray<TileB, TA::DensePolicy> B,
                     Setups const&... args) {
  using TA::detail::nested_rank;
  using TileC = std::conditional_t<(nested_rank<TileB> > nested_rank<TileA>),
                                   TileB, TileA>;
  return general_product<TileC>(A, B, args...);
}

template <DeNest DeNestFlag = DeNest::False, typename ArrayA, typename ArrayB,
          typename = std::enable_if_t<TA::detail::is_array_v<ArrayA, ArrayB>>>
auto manual_eval(OuterInnerSetup const& setups, ArrayA A, ArrayB B) {
  constexpr auto mnr = TA::detail::max_nested_rank<ArrayA, ArrayB>;
  static_assert(mnr == 1 || mnr == 2);

  auto const& outer = setups.outer();
  auto const& inner = setups.inner();

  TA_ASSERT(outer.valid());

  if constexpr (mnr == 2) {
    TA_ASSERT(inner.valid());
    if constexpr (DeNestFlag == DeNest::True) {
      // reduced nested rank in result
      using TA::detail::nested_rank;
      static_assert(nested_rank<ArrayA> == nested_rank<ArrayB>);
      TA_ASSERT(inner.rank_C == 0);
      using TileC = typename ArrayA::value_type::value_type;
      return general_product<TileC>(A, B, outer, inner);
    } else
      return general_product(A, B, outer, inner);
  } else {
    return general_product(A, B, outer);
  }
}

#ifdef TILEDARRAY_HAS_BTAS

template <typename T, typename = std::enable_if_t<TA::detail::is_tensor_v<T>>>
auto tensor_to_btas_tensor(T const& ta_tensor) {
  using value_type = typename T::value_type;
  using range_type = typename T::range_type;

  btas::Tensor<value_type, range_type> result{ta_tensor.range()};
  TA::tensor_to_btas_subtensor(ta_tensor, result);
  return result;
}

template <typename NumericT, typename RangeT, typename... Ts,
          typename = std::enable_if_t<std::is_convertible_v<RangeT, TA::Range>>>
auto btas_tensor_to_tensor(
    btas::Tensor<NumericT, RangeT, Ts...> const& btas_tensor) {
  TA::Tensor<NumericT> result{TA::Range(btas_tensor.range())};
  TA::btas_subtensor_to_tensor(btas_tensor, result);
  return result;
}

///
/// @c einsum_annot pattern example: 'ik,kj->ij'. See tensor_contract function.
///
template <typename T, std::enable_if_t<TA::detail::is_tensor_v<T>, bool> = true>
auto tensor_contract_btas(std::string const& einsum_annot, T const& A,
                          T const& B) {
  using ::Einsum::string::split2;
  auto [ab, aC] = split2(einsum_annot, "->");
  auto [aA, aB] = split2(ab, ",");

  using NumericT = typename T::numeric_type;

  struct {
    btas::Tensor<NumericT, TA::Range> A, B, C;
  } btas_tensor{tensor_to_btas_tensor(A), tensor_to_btas_tensor(B), {}};

  btas::contract(NumericT{1}, btas_tensor.A, aA, btas_tensor.B, aB, NumericT{0},
                 btas_tensor.C, aC);

  return btas_tensor_to_tensor(btas_tensor.C);
}

///
/// \tparam T TA::Tensor type
/// \param einsum_annot see tensor_contract_mult
/// \return True when TA::detail::tensor_contract and btas::contract result the
///         result. Performs bitwise comparison.
///
template <typename T, typename = std::enable_if_t<TA::detail::is_tensor_v<T>>>
auto tensor_contract_equal(std::string const& einsum_annot, T const& A,
                           T const& B) {
  T result_ta = tensor_contract(einsum_annot, A, B);
  T result_btas = tensor_contract_btas(einsum_annot, A, B);
  return result_ta == result_btas;
}

#endif

/*
 *
 * When generating arrays containing tensors of tensors (ToT) we adopt simple
 * algorithms for initialing the values. These algorithms map the outer indices
 * to the values of the inner tensor in such a way that the inner tensors will
 * have differing extents (they must all have the same rank).
 */
struct ToTArrayFixture {
  ToTArrayFixture() : m_world(*GlobalFixture::world) {}
  ~ToTArrayFixture() { GlobalFixture::world->gop.fence(); }

  /* This function returns an std::vector of tiled ranges. The ranges are for a
   * tensor of rank 1 and cover the following scenarios:
   *
   * - A single element in a single tile
   * - Multiple elements in a single tile
   * - Multiple tiles with a single element each
   * - Multiple tiles, one with a single element and one with multiple elements,
   * - Multiple tiles, each with two elements
   */
  template <typename TupleElementType>
  auto vector_tiled_ranges() {
    using trange_type = trange_type<TupleElementType>;
    return std::vector<trange_type>{
        trange_type{{0, 1}}, trange_type{{0, 2}}, trange_type{{0, 1, 2}},
        trange_type{{0, 1, 3}}, trange_type{{0, 2, 4}}};
  }

  /* This function returns an std::vector of tiled ranges. The ranges are for a
   * tensor of rank 2 and cover the following scenarios:
   * - Single tile
   *   - single element
   *   - multiple elements on row/column but single element on column/row
   *   - multiple elements on rows and columns
   * - multiple tiles on rows/columns, but single tile on column/rows
   *   - single element row/column and multiple element column/row
   *   - Multiple elements in both rows and columns
   * - multiple tiles on rows and columns
   */
  template <typename TupleElementType>
  auto matrix_tiled_ranges() {
    using trange_type = trange_type<TupleElementType>;
    return std::vector<trange_type>{
        trange_type{{0, 1}, {0, 1}},      trange_type{{0, 2}, {0, 1}},
        trange_type{{0, 2}, {0, 2}},      trange_type{{0, 1}, {0, 2}},
        trange_type{{0, 1}, {0, 1, 2}},   trange_type{{0, 1, 2}, {0, 1}},
        trange_type{{0, 2}, {0, 1, 2}},   trange_type{{0, 1, 2}, {0, 2}},
        trange_type{{0, 1, 2}, {0, 1, 2}}};
  }

  template <typename TupleElementType, typename Index>
  auto inner_vector_tile(Index&& idx) {
    auto sum = std::accumulate(idx.begin(), idx.end(), 0);
    inner_type<TupleElementType> elem(Range(sum + 1));
    std::iota(elem.begin(), elem.end(), 1);
    return elem;
  }

  template <typename TupleElementType, typename Index>
  auto inner_matrix_tile(Index&& idx) {
    unsigned int row_max = idx[0] + 1;
    unsigned int col_max = std::accumulate(idx.begin(), idx.end(), 0) + 1;
    unsigned int zero = 0;
    inner_type<TupleElementType> elem(Range({zero, zero}, {row_max, col_max}));
    std::iota(elem.begin(), elem.end(), 1);
    return elem;
  }

  template <typename TupleElementType>
  auto tensor_of_vector(const TiledRange& tr) {
    return make_array<tensor_type<TupleElementType>>(
        m_world, tr, [this](tile_type<TupleElementType>& tile, const Range& r) {
          tile_type<TupleElementType> new_tile(r);
          for (auto idx : r) {
            new_tile(idx) = inner_vector_tile<TupleElementType>(idx);
          }
          tile = new_tile;
        });
  }

  template <typename TupleElementType>
  auto tensor_of_matrix(const TiledRange& tr) {
    return make_array<tensor_type<TupleElementType>>(
        m_world, tr, [this](tile_type<TupleElementType>& tile, const Range& r) {
          tile_type<TupleElementType> new_tile(r);
          for (auto idx : r) {
            new_tile(idx) = inner_matrix_tile<TupleElementType>(idx);
          }
          tile = new_tile;
        });
  }

  /* The majority of the unit tests can be written in such a way that all they
   * need is the tiled range, the rank of the inner tensors, and the resulting
   * array. The tiled range plus the inner rank is more or less the correct
   * answer, since the ToTArrayFixture maps it to the values of the array, and
   * the tensor is the thing to test. This function creates a std::vector of
   * tiled range, inner rank, tensor tuples spanning:
   *
   * - vector of vectors
   * - vector of matrices
   * - matrix of vectors
   * - matrix of matrices
   *
   * The unit tests simply loop over the vector testing for all these scenarios
   * with a few lines of code.
   */
  template <typename TupleElementType>
  auto run_all() {
    using trange_type = trange_type<TupleElementType>;
    using tensor_type = tensor_type<TupleElementType>;
    std::vector<std::tuple<trange_type, int, tensor_type>> rv;

    // Make vector of vector
    for (auto tr : vector_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 1, tensor_of_vector<TupleElementType>(tr)));

    // Make vector of matrix
    for (auto tr : vector_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 2, tensor_of_matrix<TupleElementType>(tr)));

    // Make matrix of vector
    for (auto tr : matrix_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 1, tensor_of_vector<TupleElementType>(tr)));

    // Make matrix of matrix
    for (auto tr : matrix_tiled_ranges<TupleElementType>())
      rv.push_back(
          std::make_tuple(tr, 2, tensor_of_matrix<TupleElementType>(tr)));

    // Make sure all the tensors are actually made
    m_world.gop.fence();
    return rv;
  }

  /* This function tests for exact equality between DistArray instances. By
   * exact equality we mean:
   * - Same type
   * - Either both are initialized or both are not initialized
   * - Same MPI context
   * - Same shape (unless the template parameter ShapeCmp is set false)
   * - Same distribution
   * - Same tiling
   * - Components are bit-wise equal (i.e., 3.1400000000 != 3.1400000001)
   *
   * TODO: pmap comparisons
   */
  template <ShapeComp ShapeCompFlag = ShapeComp::True, typename LHSTileType,
            typename LHSPolicy, typename RHSTileType, typename RHSPolicy>
  static bool are_equal(const DistArray<LHSTileType, LHSPolicy>& lhs,
                        const DistArray<RHSTileType, RHSPolicy>& rhs) {
    // Same type
    if constexpr (!std::is_same_v<decltype(lhs), decltype(rhs)>) {
      return false;
    } else {
      // Are initialized?
      if (lhs.is_initialized() != rhs.is_initialized()) return false;
      if (!lhs.is_initialized()) return true;  // both are default constructed

      // Same world instance?
      if (&lhs.world() != &rhs.world()) return false;

      // Same shape?
      if constexpr (ShapeCompFlag == ShapeComp::True)
        if (lhs.shape() != rhs.shape()) return false;

      // Same pmap?
      // if(*lhs.pmap() != *rhs.pmap()) return false;

      // Same tiling?
      if (lhs.trange() != rhs.trange()) return false;

      // Same components? Here we make all ranks check all tiles
      bool are_same = true;
      for (auto idx : lhs.tiles_range()) {
        const auto& lhs_tot = lhs.find(idx).get();
        const auto& rhs_tot = rhs.find(idx).get();
        if (lhs_tot != rhs_tot) {
          are_same = false;
          break;
        }
      }
      lhs.world().gop.fence();
      return are_same;
    }
    abort();  // unreachable
  }
  // The world to use for the test suite
  madness::World& m_world;

};  // TotArrayFixture
#endif
