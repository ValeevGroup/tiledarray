/// Unit tests for arena einsum plans and dispatch.

#include "TiledArray/tensor/arena_einsum.h"

#include "tiledarray.h"
#include "unit_test_config.h"

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
