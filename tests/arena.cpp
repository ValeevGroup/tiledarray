#include "TiledArray/tensor/arena.h"

#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <vector>

using TiledArray::detail::Arena;
using TiledArray::detail::ArenaPlan;
using TiledArray::detail::ArenaResource;
using TiledArray::detail::plan;

namespace {
// Minimal Range-like shim for plan() tests: supports only volume().
struct FakeRange {
  std::size_t v;
  std::size_t volume() const noexcept { return v; }
};
}

BOOST_AUTO_TEST_SUITE(arena_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(default_arena_is_empty) {
  Arena a;
  BOOST_CHECK_EQUAL(a.capacity(), 0u);
  BOOST_CHECK_EQUAL(a.cursor(), 0u);
  BOOST_CHECK(a.empty());
  BOOST_CHECK(a.resource() != nullptr);
}

BOOST_AUTO_TEST_CASE(reserve_initializes_capacity) {
  Arena a;
  a.reserve(1024);
  BOOST_CHECK_EQUAL(a.capacity(), 1024u);
  BOOST_CHECK_EQUAL(a.cursor(), 0u);
  BOOST_CHECK_EQUAL(a.remaining(), 1024u);
}

BOOST_AUTO_TEST_CASE(reserve_zero_init_clears_slab) {
  Arena a;
  a.reserve(64, /*zero_init=*/true);
  auto h = a.slice<unsigned char>(0, 64);
  for (std::size_t i = 0; i < 64; ++i) BOOST_CHECK_EQUAL(h[i], 0u);
}

BOOST_AUTO_TEST_CASE(slice_random_access_and_aliasing) {
  Arena a;
  a.reserve(1024);
  std::shared_ptr<double[]> p1 = a.slice<double>(0, 4);
  std::shared_ptr<double[]> p2 = a.slice<double>(64, 4);
  for (int i = 0; i < 4; ++i) p1[i] = double(i);
  for (int i = 0; i < 4; ++i) p2[i] = double(10 + i);
  for (int i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(p1[i], double(i));
  for (int i = 0; i < 4; ++i) BOOST_CHECK_EQUAL(p2[i], double(10 + i));
  BOOST_CHECK(static_cast<void*>(&p2[0]) >= static_cast<void*>(&p1[4]));
}

BOOST_AUTO_TEST_CASE(claim_advances_cursor_and_aligns) {
  Arena a;
  a.reserve(1024);
  std::shared_ptr<double[]> h = a.claim<double>(10);
  BOOST_REQUIRE(h.get() != nullptr);
  BOOST_CHECK_EQUAL(reinterpret_cast<std::uintptr_t>(h.get()) % alignof(double),
                    0u);
  BOOST_CHECK(a.cursor() >= 10u * sizeof(double));
}

BOOST_AUTO_TEST_CASE(slab_survives_arena_destruction) {
  std::shared_ptr<int[]> survivor;
  {
    Arena tmp;
    tmp.reserve(256);
    survivor = tmp.claim<int>(10);
    for (int i = 0; i < 10; ++i) survivor[i] = -i;
  }
  for (int i = 0; i < 10; ++i) BOOST_CHECK_EQUAL(survivor[i], -i);
}

BOOST_AUTO_TEST_CASE(plan_uniform_cells) {
  ArenaPlan p = plan(
      /*N_cells=*/6,
      /*shape_fn=*/[](std::size_t /*ord*/) { return FakeRange{10}; },
      /*element_size=*/sizeof(double),
      /*alignment=*/alignof(double));
  BOOST_CHECK_EQUAL(p.total_bytes, 6u * 10u * sizeof(double));
  BOOST_CHECK_EQUAL(p.offsets.size(), 6u);
  BOOST_CHECK_EQUAL(p.offsets[0], 0u);
  BOOST_CHECK_EQUAL(p.offsets[5], 5u * 10u * sizeof(double));
}

BOOST_AUTO_TEST_CASE(plan_variable_cells_match_pivot_doc_example) {
  ArenaPlan p = plan(
      /*N_cells=*/12,
      /*shape_fn=*/[](std::size_t /*ord*/) { return FakeRange{20}; },
      /*element_size=*/sizeof(double),
      /*alignment=*/alignof(double));
  BOOST_CHECK_EQUAL(p.total_bytes, 12u * 20u * sizeof(double));
  BOOST_CHECK_EQUAL(p.offsets[1], 20u * sizeof(double));
}

BOOST_AUTO_TEST_CASE(plan_then_construct_then_read) {
  const std::size_t N = 4;
  std::vector<std::size_t> volumes = {3, 5, 2, 7};
  auto shape_fn = [&volumes](std::size_t ord) { return FakeRange{volumes[ord]}; };
  ArenaPlan p = plan(N, shape_fn, sizeof(double), alignof(double));
  Arena a;
  a.reserve(p.total_bytes);
  std::vector<std::shared_ptr<double[]>> handles(N);
  for (std::size_t ord = 0; ord < N; ++ord) {
    handles[ord] = a.slice<double>(p.offsets[ord], volumes[ord]);
    for (std::size_t i = 0; i < volumes[ord]; ++i)
      handles[ord][i] = double(100 * ord + i);
  }
  for (std::size_t ord = 0; ord < N; ++ord)
    for (std::size_t i = 0; i < volumes[ord]; ++i)
      BOOST_CHECK_EQUAL(handles[ord][i], double(100 * ord + i));
}

BOOST_AUTO_TEST_CASE(arena_resource_is_identity_equal) {
  Arena a;
  a.reserve(64);
  ArenaResource r1(&a);
  ArenaResource r2(&a);
  BOOST_CHECK(r1.is_equal(r1));
  BOOST_CHECK(!r1.is_equal(r2));
}

BOOST_AUTO_TEST_SUITE_END()
