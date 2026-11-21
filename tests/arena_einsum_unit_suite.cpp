/// Unit tests for arena einsum plans and dispatch.

#include "TiledArray/tensor/arena_einsum.h"

#include "tiledarray.h"
#include "unit_test_config.h"

#include <optional>
#include <random>
#include <vector>

BOOST_AUTO_TEST_SUITE(arena_einsum_unit_suite, TA_UT_LABEL_SERIAL)

namespace TA = TiledArray;

BOOST_AUTO_TEST_CASE(inner_shape_plan_left_range) {
  TA::Tensor<double> l(TA::Range{3, 4});
  TA::Tensor<double> r(TA::Range{3, 4});
  TA::detail::ArenaInnerShapePlan p{
      TA::detail::ArenaInnerShapeKind::left_range, std::nullopt};
  auto out = p.make<TA::Range>(l, r);
  BOOST_CHECK(out == l.range());
}

BOOST_AUTO_TEST_CASE(inner_shape_plan_right_range) {
  TA::Tensor<double> l(TA::Range{3, 4});
  TA::Tensor<double> r(TA::Range{5, 6});
  TA::detail::ArenaInnerShapePlan p{
      TA::detail::ArenaInnerShapeKind::right_range, std::nullopt};
  auto out = p.make<TA::Range>(l, r);
  BOOST_CHECK(out == r.range());
}

BOOST_AUTO_TEST_CASE(inner_shape_plan_gemm_result_range) {
  TA::Tensor<double> l(TA::Range{3, 5});
  TA::Tensor<double> r(TA::Range{5, 4});
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                           TA::math::blas::NoTranspose, 2, 2, 2);
  TA::detail::ArenaInnerShapePlan p{
      TA::detail::ArenaInnerShapeKind::gemm_result_range,
      std::make_optional(gh)};
  auto out = p.make<TA::Range>(l, r);
  BOOST_CHECK_EQUAL(out.volume(), std::size_t{12});
}

BOOST_AUTO_TEST_CASE(is_contraction_arena_tot_v_predicate) {
  using ToT = TA::Tensor<TA::Tensor<double>>;
  static_assert(TA::detail::is_contraction_arena_tot_v<ToT, ToT, ToT>);
  using Plain = TA::Tensor<double>;
  static_assert(!TA::detail::is_contraction_arena_tot_v<Plain, Plain, Plain>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(arena_plan_storage_t_resolves) {
  using ToT = TA::Tensor<TA::Tensor<double>>;
  using Plain = TA::Tensor<double>;
  using ToTStorage = TA::detail::arena_plan_storage_t<ToT, ToT, ToT>;
  using PlainStorage = TA::detail::arena_plan_storage_t<Plain, Plain, Plain>;
  static_assert(!std::is_same_v<ToTStorage, std::monostate>);
  static_assert(std::is_same_v<PlainStorage, std::monostate>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(make_plan_returns_nullopt_when_disabled) {
  using ToT = TA::Tensor<TA::Tensor<double>>;
  TA::detail::arena_disabled() = true;
  auto plan = TA::detail::make_contraction_arena_plan<ToT, ToT, ToT>(
      TA::detail::ArenaInnerShapeKind::left_range, std::nullopt,
      TA::Permutation{});
  BOOST_CHECK(!plan.has_value());
  TA::detail::arena_disabled() = false;
}

BOOST_AUTO_TEST_CASE(make_plan_returns_nullopt_for_plain_tensor) {
  using Plain = TA::Tensor<double>;
  // Non-ToT gating happens inside the function body, not in the return type.
  auto plan = TA::detail::make_contraction_arena_plan<Plain, Plain, Plain>(
      TA::detail::ArenaInnerShapeKind::left_range, std::nullopt,
      TA::Permutation{});
  BOOST_CHECK(!plan.has_value());
}

BOOST_AUTO_TEST_CASE(make_plan_rejects_nonidentity_inner_perm) {
  using ToT = TA::Tensor<TA::Tensor<double>>;
  TA::Permutation perm({1, 0});
  auto plan = TA::detail::make_contraction_arena_plan<ToT, ToT, ToT>(
      TA::detail::ArenaInnerShapeKind::left_range, std::nullopt, perm);
  BOOST_CHECK(!plan.has_value());
}

BOOST_AUTO_TEST_CASE(make_plan_returns_active_for_tot) {
  using ToT = TA::Tensor<TA::Tensor<double>>;
  auto plan = TA::detail::make_contraction_arena_plan<ToT, ToT, ToT>(
      TA::detail::ArenaInnerShapeKind::left_range, std::nullopt,
      TA::Permutation{});
  BOOST_CHECK(plan.has_value());
}

namespace {
using ToT = TA::Tensor<TA::Tensor<double>>;

// Placement-new initializes each ToT inner cell in existing tensor storage.
ToT make_uniform_tot(const TA::Range& outer, const TA::Range& inner,
                     double fill) {
  ToT t(outer);
  const std::size_t vol = outer.volume();
  for (std::size_t i = 0; i < vol; ++i) {
    new (t.data() + i) TA::Tensor<double>(inner, fill);
  }
  return t;
}
}  // namespace

BOOST_AUTO_TEST_CASE(reserve_and_construct_uniform_inner) {
  TA::math::GemmHelper outer_gh(TA::math::blas::NoTranspose,
                                  TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                  TA::math::blas::NoTranspose, 2, 2, 2);
  auto left  = make_uniform_tot(TA::Range{2, 3}, TA::Range{3, 5}, 1.0);
  auto right = make_uniform_tot(TA::Range{3, 4}, TA::Range{5, 4}, 1.0);
  TA::detail::ArenaInnerShapePlan inner_plan{
      TA::detail::ArenaInnerShapeKind::gemm_result_range,
      std::make_optional(inner_gh)};
  TA::detail::ContractionArenaPlan<ToT, ToT, ToT> plan(inner_plan);
  ToT result = plan.reserve_and_construct(left, right, outer_gh);
  BOOST_CHECK_EQUAL(result.range().volume(), std::size_t{8});
  BOOST_CHECK_EQUAL(result.data()[0].range().volume(), std::size_t{12});
}

BOOST_AUTO_TEST_CASE(reserve_and_construct_zero_volume_outer_skips_reserve) {
  TA::math::GemmHelper outer_gh(TA::math::blas::NoTranspose,
                                  TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                  TA::math::blas::NoTranspose, 2, 2, 2);
  auto left  = make_uniform_tot(TA::Range{0, 3}, TA::Range{3, 5}, 1.0);
  auto right = make_uniform_tot(TA::Range{3, 2}, TA::Range{5, 4}, 1.0);
  TA::detail::ArenaInnerShapePlan inner_plan{
      TA::detail::ArenaInnerShapeKind::gemm_result_range,
      std::make_optional(inner_gh)};
  TA::detail::ContractionArenaPlan<ToT, ToT, ToT> plan(inner_plan);
  ToT result = plan.reserve_and_construct(left, right, outer_gh);
  BOOST_CHECK_EQUAL(result.range().volume(), std::size_t{0});
}

BOOST_AUTO_TEST_CASE(reserve_and_construct_jagged_inner_per_cell) {
  // Jagged left cells make first-non-empty K-strip range selection observable.
  TA::math::GemmHelper outer_gh(TA::math::blas::NoTranspose,
                                  TA::math::blas::NoTranspose, 2, 2, 2);
  ToT left(TA::Range{2, 3});
  for (std::size_t m = 0; m < 2; ++m)
    for (std::size_t k = 0; k < 3; ++k) {
      TA::Range r{static_cast<long>(m + 1), static_cast<long>(k + 2)};
      new (left.data() + (m * 3 + k)) TA::Tensor<double>(r, 1.0);
    }
  auto right = make_uniform_tot(TA::Range{3, 2}, TA::Range{2, 2}, 1.0);
  TA::detail::ArenaInnerShapePlan inner_plan{
      TA::detail::ArenaInnerShapeKind::left_range, std::nullopt};
  TA::detail::ContractionArenaPlan<ToT, ToT, ToT> plan(inner_plan);
  ToT result = plan.reserve_and_construct(left, right, outer_gh);
  BOOST_CHECK_EQUAL(result.range().volume(), std::size_t{4});
  BOOST_CHECK_EQUAL(result.data()[0].range().volume(), std::size_t{2});
  BOOST_CHECK_EQUAL(result.data()[1].range().volume(), std::size_t{2});
  BOOST_CHECK_EQUAL(result.data()[2].range().volume(), std::size_t{4});
  BOOST_CHECK_EQUAL(result.data()[3].range().volume(), std::size_t{4});
}

namespace {

/// A ToT operand tile over \p range whose cell \c o is null
/// (`state[o] == 0`), has range \p inner (`1`), or -- owning inner only --
/// is a non-empty cell of the zero-volume range \p zero_inner (`2`).
template <typename Outer>
Outer make_grow_operand(const TA::Range& range, const std::vector<int>& state,
                        const TA::Range& inner, const TA::Range& zero_inner) {
  using Inner = typename Outer::value_type;
  if constexpr (TA::is_arena_tensor_v<Inner>) {
    return TA::detail::arena_outer_init<Outer>(
        range, 1, [&](std::size_t o) -> TA::Range {
          return state[o] == 1 ? inner : TA::Range{};
        });
  } else {
    Outer t(range);
    for (std::size_t o = 0; o != range.volume(); ++o) {
      if (state[o] == 1) t.data()[o] = Inner(inner, 1.0);
      if (state[o] == 2) t.data()[o] = Inner(zero_inner);
    }
    return t;
  }
}

/// Checks `grow_to_cover` against `reserve_and_construct`, which evaluates
/// every cell's range. A result shaped by one random K-panel is grown by a
/// second one; a null cell must become exactly what the second panel alone
/// would shape, a non-null cell must keep its range and data, and a panel
/// that grows nothing must leave the tile's storage alone. The K values
/// straddle the 64-bit words of the non-empty masks.
template <typename Outer>
void check_grow_to_cover(bool zero_volume_cells, unsigned seed) {
  using TA::detail::ArenaInnerShapeKind;
  namespace blas = TA::math::blas;
  std::mt19937 gen(seed);
  auto pick = [&gen](std::size_t n) { return gen() % n; };
  const long Ks[] = {1, 3, 63, 64, 65, 130};
  const blas::Op ops[] = {blas::NoTranspose, blas::Transpose};
  const ArenaInnerShapeKind kinds[] = {ArenaInnerShapeKind::left_range,
                                       ArenaInnerShapeKind::right_range,
                                       ArenaInnerShapeKind::gemm_result_range};
  std::size_t ngrown = 0, nsteady = 0;
  for (int trial = 0; trial != 600; ++trial) {
    const long M = 1 + pick(4), N = 1 + pick(4), K = Ks[pick(6)];
    const auto lop = ops[pick(2)], rop = ops[pick(2)];
    const auto kind = kinds[pick(3)];
    const double density = pick(3) == 0 ? 0.02 : 0.3;
    const TA::math::GemmHelper outer_gh(lop, rop, 2, 2, 2);
    std::optional<TA::math::GemmHelper> inner_gh;
    TA::Range linner{2, 3}, rinner{2, 3};
    const TA::Range lzero{0, 3};
    TA::Range rzero{0, 3};
    if (kind == ArenaInnerShapeKind::gemm_result_range) {
      inner_gh.emplace(blas::NoTranspose, blas::NoTranspose, 2, 2, 2);
      rinner = TA::Range{3, 4};
      rzero = TA::Range{3, 0};
    }
    auto plan = TA::detail::make_contraction_arena_plan<Outer, Outer, Outer>(
        kind, inner_gh, TA::Permutation{});
    BOOST_REQUIRE(plan.has_value());
    const TA::Range lrange =
        lop == blas::NoTranspose ? TA::Range{M, K} : TA::Range{K, M};
    const TA::Range rrange =
        rop == blas::NoTranspose ? TA::Range{K, N} : TA::Range{N, K};
    auto random_state = [&](std::size_t n) {
      std::vector<int> state(n);
      for (auto& s : state) {
        const double u = pick(10000) / 10000.0;
        s = u < density ? 1 : (zero_volume_cells && u < 2 * density ? 2 : 0);
      }
      return state;
    };
    const Outer l1 =
        make_grow_operand<Outer>(lrange, random_state(M * K), linner, lzero);
    const Outer r1 =
        make_grow_operand<Outer>(rrange, random_state(K * N), rinner, rzero);
    const Outer l2 =
        make_grow_operand<Outer>(lrange, random_state(M * K), linner, lzero);
    const Outer r2 =
        make_grow_operand<Outer>(rrange, random_state(K * N), rinner, rzero);

    const Outer ref1 = plan->reserve_and_construct(l1, r1, outer_gh);
    const Outer ref2 = plan->reserve_and_construct(l2, r2, outer_gh);
    Outer result = plan->reserve_and_construct(l1, r1, outer_gh);
    const std::size_t ncells = static_cast<std::size_t>(M * N);
    for (std::size_t o = 0; o != ncells; ++o) {
      auto& cell = result.data()[o];
      if (cell.empty()) continue;
      for (std::size_t i = 0; i != cell.size(); ++i)
        cell.data()[i] = 1.0 + o + 0.01 * i;
    }
    const auto* const data_before = result.data();
    BOOST_REQUIRE_NO_THROW(plan->grow_to_cover(result, l2, r2, outer_gh));

    bool grows = false;
    for (std::size_t o = 0; o != ncells; ++o)
      if (ref1.data()[o].empty() && !ref2.data()[o].empty() &&
          ref2.data()[o].range().volume() != 0)
        grows = true;
    if (!grows) {
      ++nsteady;
      BOOST_CHECK(result.data() == data_before);  // no reallocation
      continue;
    }
    ++ngrown;
    for (std::size_t o = 0; o != ncells; ++o) {
      const auto& got = result.data()[o];
      const auto& want = ref1.data()[o].empty() ? ref2.data()[o]   // grown
                                                : ref1.data()[o];  // kept
      BOOST_REQUIRE_EQUAL(got.empty(), want.empty());
      if (got.empty()) continue;
      BOOST_REQUIRE(got.range() == want.range());
      const bool kept = !ref1.data()[o].empty();
      for (std::size_t i = 0; i != got.size(); ++i)
        BOOST_REQUIRE_EQUAL(got.data()[i], kept ? 1.0 + o + 0.01 * i : 0.0);
    }
  }
  // both the growing and the steady-state branch were exercised
  BOOST_CHECK_GT(ngrown, 0u);
  BOOST_CHECK_GT(nsteady, 0u);
}

}  // namespace

BOOST_AUTO_TEST_CASE(grow_to_cover_matches_reserve_and_construct_arena) {
  check_grow_to_cover<TA::Tensor<TA::ArenaTensor<double, TA::Range>>>(
      /* zero_volume_cells = */ false, 1u);
}

BOOST_AUTO_TEST_CASE(grow_to_cover_matches_reserve_and_construct_owning) {
  check_grow_to_cover<TA::Tensor<TA::Tensor<double>>>(
      /* zero_volume_cells = */ false, 2u);
}

// An owning inner cell can be non-empty with a zero-volume range; such a
// cell grows nothing, so it must not make a panel reallocate the result.
BOOST_AUTO_TEST_CASE(grow_to_cover_zero_volume_operand_cells) {
  check_grow_to_cover<TA::Tensor<TA::Tensor<double>>>(
      /* zero_volume_cells = */ true, 3u);
}

BOOST_AUTO_TEST_CASE(fused_hadamard_inplace_accumulates) {
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> l(TA::Range{4}, 1.0);
  TA::Tensor<double> rr(TA::Range{4}, 2.0);
  TA::detail::fused_hadamard_inplace(r, l, rr);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 2.0, 1e-12);
  TA::detail::fused_hadamard_inplace(r, l, rr);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 4.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_hadamard_scaled_inplace_accumulates) {
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> l(TA::Range{4}, 1.0);
  TA::Tensor<double> rr(TA::Range{4}, 2.0);
  TA::detail::fused_hadamard_scaled_inplace(r, l, rr, 3.0);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 6.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_scale_tot_x_t_inplace_accumulates) {
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> l(TA::Range{4}, 1.5);
  TA::detail::fused_scale_tot_x_t_inplace(r, l, 2.0);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 3.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_scale_t_x_tot_inplace_accumulates) {
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> rr(TA::Range{4}, 2.5);
  TA::detail::fused_scale_t_x_tot_inplace(r, 4.0, rr);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 10.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_contraction_inplace_accumulates) {
  TA::Tensor<double> r(TA::Range{2, 2}, 0.0);
  TA::Tensor<double> l(TA::Range{2, 2}, 1.0);
  TA::Tensor<double> rr(TA::Range{2, 2}, 2.0);
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                           TA::math::blas::NoTranspose, 2, 2, 2);
  TA::detail::fused_contraction_inplace(r, l, rr, 1.0, gh);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 4.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_hadamard_lambda_round_trip) {
  auto fn = TA::detail::make_fused_hadamard_lambda<
      TA::Tensor<double>, TA::Tensor<double>, TA::Tensor<double>>();
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> l(TA::Range{4}, 1.0);
  TA::Tensor<double> rr(TA::Range{4}, 2.0);
  fn(r, l, rr);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 2.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_hadamard_scaled_lambda_round_trip) {
  auto fn = TA::detail::make_fused_hadamard_scaled_lambda<
      TA::Tensor<double>, TA::Tensor<double>, TA::Tensor<double>, double>(3.0);
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> l(TA::Range{4}, 1.0);
  TA::Tensor<double> rr(TA::Range{4}, 2.0);
  fn(r, l, rr);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 6.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_scale_tot_x_t_lambda_round_trip) {
  auto fn = TA::detail::make_fused_scale_tot_x_t_lambda<
      TA::Tensor<double>, TA::Tensor<double>, double>();
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> l(TA::Range{4}, 1.5);
  fn(r, l, 2.0);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 3.0, 1e-12);
}

BOOST_AUTO_TEST_CASE(fused_scale_t_x_tot_lambda_round_trip) {
  auto fn = TA::detail::make_fused_scale_t_x_tot_lambda<
      TA::Tensor<double>, double, TA::Tensor<double>>();
  TA::Tensor<double> r(TA::Range{4}, 0.0);
  TA::Tensor<double> rr(TA::Range{4}, 2.5);
  fn(r, 4.0, rr);
  for (std::size_t i = 0; i < 4; ++i)
    BOOST_CHECK_CLOSE(r.data()[i], 10.0, 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
