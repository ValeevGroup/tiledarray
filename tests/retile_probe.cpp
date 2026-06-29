#include <tiledarray.h>
#include <TiledArray/util/retile_probe.h>
#include "unit_test_config.h"

using TiledArray::detail::RetileBucket;
using TiledArray::detail::RetileTimer;
using TiledArray::detail::PermuteBackScope;

namespace {
// Burn a few microseconds so the wall timer records a strictly positive ns.
inline void burn() {
  volatile std::uint64_t x = 0;
  for (int i = 0; i < 200000; ++i) x += i;
}
inline std::uint64_t calls(const TiledArray::detail::RetileCounters& c,
                           RetileBucket b) {
  return c.calls[static_cast<std::size_t>(b)];
}
inline std::uint64_t ns(const TiledArray::detail::RetileCounters& c,
                        RetileBucket b) {
  return c.ns[static_cast<std::size_t>(b)];
}
}  // namespace

BOOST_AUTO_TEST_SUITE(retile_probe_suite)

BOOST_AUTO_TEST_CASE(accumulates_when_enabled) {
  TiledArray::detail::set_retile_probe_enabled_for_testing(true);
  TiledArray::detail::retile_probe_reset_for_testing();
  {
    RetileTimer t{RetileBucket::Gemm};
    burn();
  }
  auto s = TiledArray::detail::retile_probe_snapshot();
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::Gemm), 1u);
  BOOST_CHECK_GT(ns(s, RetileBucket::Gemm), 0u);
  TiledArray::detail::clear_retile_probe_testing_override();
}

BOOST_AUTO_TEST_CASE(noop_when_disabled) {
  TiledArray::detail::set_retile_probe_enabled_for_testing(false);
  TiledArray::detail::retile_probe_reset_for_testing();
  {
    RetileTimer t{RetileBucket::Gemm};
    burn();
  }
  auto s = TiledArray::detail::retile_probe_snapshot();
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::Gemm), 0u);
  TiledArray::detail::clear_retile_probe_testing_override();
}

BOOST_AUTO_TEST_CASE(extra_gate_false_suppresses) {
  TiledArray::detail::set_retile_probe_enabled_for_testing(true);
  TiledArray::detail::retile_probe_reset_for_testing();
  {
    RetileTimer t{RetileBucket::RepackIn, /*extra_gate=*/false};
    burn();
  }
  auto s = TiledArray::detail::retile_probe_snapshot();
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::RepackIn), 0u);
  TiledArray::detail::clear_retile_probe_testing_override();
}

BOOST_AUTO_TEST_CASE(permute_back_suppresses_nested_permute_in) {
  TiledArray::detail::set_retile_probe_enabled_for_testing(true);
  TiledArray::detail::retile_probe_reset_for_testing();
  {
    PermuteBackScope guard;  // raises tls_permute_back_depth()
    // a nested PermuteIn timer must see depth>0 and stay off
    RetileTimer in{RetileBucket::PermuteIn,
                   TiledArray::detail::tls_permute_back_depth() == 0};
    burn();
  }
  auto s = TiledArray::detail::retile_probe_snapshot();
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::PermuteIn), 0u);
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::PermuteBack), 1u);
  TiledArray::detail::clear_retile_probe_testing_override();
}

BOOST_AUTO_TEST_CASE(gemm_seam_fires_without_retile) {
  auto& w = TA::get_default_world();
  TA::TiledRange tr{{0, 2, 4}, {0, 2, 4}};  // 2x2 tiles each mode
  TA::TArrayD a(w, tr), b(w, tr);
  a.fill(1.0);
  b.fill(1.0);
  w.gop.fence();

  TiledArray::detail::set_retile_probe_enabled_for_testing(true);
  TiledArray::detail::retile_probe_reset_for_testing();

  TA::TArrayD c;
  c("i,j") = a("i,k") * b("k,j");  // plain dense contraction, NO retile
  w.gop.fence();

  auto s = TiledArray::detail::retile_probe_snapshot();
  // GEMM fired even though plan_.active is false (no retile requested).
  BOOST_CHECK_GT(calls(s, RetileBucket::Gemm), 0u);
  // No retile machinery ran.
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::RepackIn), 0u);
  BOOST_CHECK_EQUAL(calls(s, RetileBucket::CarveOut), 0u);
  TiledArray::detail::clear_retile_probe_testing_override();
}

BOOST_AUTO_TEST_SUITE_END()
