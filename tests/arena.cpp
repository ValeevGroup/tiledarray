#include "TiledArray/tensor/arena.h"

#include "tiledarray.h"
#include "unit_test_config.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <vector>

using TiledArray::detail::Arena;
using TiledArray::detail::ArenaResource;
using TiledArray::detail::kArenaDefaultPageBytes;

namespace {
bool is_aligned(const void* p, std::size_t a) {
  return reinterpret_cast<std::uintptr_t>(p) % a == 0;
}
}  // namespace

BOOST_AUTO_TEST_SUITE(arena_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(default_arena_is_empty) {
  Arena a;
  BOOST_CHECK(a.empty());
  BOOST_CHECK_EQUAL(a.page_count(), 0u);
  BOOST_CHECK_EQUAL(a.bytes_allocated(), 0u);
  BOOST_CHECK_EQUAL(a.bytes_reserved(), 0u);
  BOOST_CHECK_EQUAL(a.page_size(), kArenaDefaultPageBytes);
  BOOST_CHECK(a.resource() != nullptr);
}

BOOST_AUTO_TEST_CASE(reserve_page_lays_down_one_exact_page) {
  Arena a;
  a.reserve_page(1024, 64);
  BOOST_CHECK_EQUAL(a.page_count(), 1u);
  BOOST_CHECK_EQUAL(a.bytes_reserved(), 1024u);
  // nothing claimed yet
  BOOST_CHECK(a.empty());
  BOOST_CHECK_EQUAL(a.bytes_allocated(), 0u);
}

BOOST_AUTO_TEST_CASE(claims_pack_into_the_reserved_page) {
  Arena a;
  a.reserve_page(1024, 128);
  auto h1 = a.claim_bytes(100, 64);
  auto h2 = a.claim_bytes(100, 64);
  auto h3 = a.claim_bytes(100, 64);
  // all three land in the single reserved page
  BOOST_CHECK_EQUAL(a.page_count(), 1u);
  BOOST_CHECK_EQUAL(a.bytes_allocated(), 300u);
  BOOST_CHECK(is_aligned(h1.get(), 64));
  BOOST_CHECK(is_aligned(h2.get(), 64));
  BOOST_CHECK(is_aligned(h3.get(), 64));
  // distinct, non-overlapping
  BOOST_CHECK(h2.get() >= h1.get() + 100);
  BOOST_CHECK(h3.get() >= h2.get() + 100);
}

BOOST_AUTO_TEST_CASE(claim_auto_allocates_a_standard_page) {
  Arena a;  // no reserve_page
  std::shared_ptr<double[]> h = a.claim<double>(10);
  BOOST_REQUIRE(h.get() != nullptr);
  BOOST_CHECK(is_aligned(h.get(), alignof(double)));
  BOOST_CHECK_EQUAL(a.page_count(), 1u);
  BOOST_CHECK_EQUAL(a.bytes_reserved(), kArenaDefaultPageBytes);
  for (int i = 0; i < 10; ++i) h[i] = double(i);
  for (int i = 0; i < 10; ++i) BOOST_CHECK_EQUAL(h[i], double(i));
}

BOOST_AUTO_TEST_CASE(claims_roll_over_to_fresh_pages) {
  Arena a(std::pmr::new_delete_resource(), /*page_size=*/256);
  std::vector<std::shared_ptr<std::byte[]>> handles;
  // 64 B at 64-B alignment => 4 per 256 B page; 10 claims => >= 3 pages
  for (int i = 0; i < 10; ++i) handles.push_back(a.claim_bytes(64, 64));
  BOOST_CHECK_GE(a.page_count(), 3u);
  BOOST_CHECK_EQUAL(a.bytes_allocated(), 10u * 64u);
  // every handle is a distinct, valid, writable region
  for (std::size_t i = 0; i < handles.size(); ++i)
    std::memset(handles[i].get(), int(i), 64);
  for (std::size_t i = 0; i < handles.size(); ++i)
    BOOST_CHECK_EQUAL(static_cast<unsigned char>(handles[i][0]),
                      static_cast<unsigned char>(i));
}

BOOST_AUTO_TEST_CASE(oversized_claim_gets_a_dedicated_page) {
  Arena a(std::pmr::new_delete_resource(), /*page_size=*/256);
  // request larger than a page -> a dedicated, exactly-sized page
  auto big = a.claim_bytes(1024, 64);
  BOOST_REQUIRE(big.get() != nullptr);
  BOOST_CHECK(is_aligned(big.get(), 64));
  BOOST_CHECK_EQUAL(a.page_count(), 1u);
  BOOST_CHECK_EQUAL(a.bytes_reserved(), 1024u);
  // a following normal claim does not reuse the dedicated page; it opens a
  // standard page
  auto small = a.claim_bytes(64, 64);
  BOOST_CHECK_EQUAL(a.page_count(), 2u);
  BOOST_CHECK_EQUAL(a.bytes_reserved(), 1024u + 256u);
  std::memset(big.get(), 1, 1024);
  std::memset(small.get(), 2, 64);
}

BOOST_AUTO_TEST_CASE(single_exact_page_corner_case) {
  // corner case (b): a lone cell -> one exactly-sized page, no waste
  Arena a;
  a.reserve_page(640, 128);
  auto h = a.claim_bytes(640, 128);
  BOOST_CHECK_EQUAL(a.page_count(), 1u);
  BOOST_CHECK_EQUAL(a.bytes_reserved(), 640u);
  BOOST_CHECK_EQUAL(a.bytes_allocated(), 640u);
  BOOST_CHECK(is_aligned(h.get(), 128));
}

BOOST_AUTO_TEST_CASE(zero_init_clears_each_page) {
  Arena a(std::pmr::new_delete_resource(), /*page_size=*/256,
          /*zero_init=*/true);
  auto h = a.claim<unsigned char>(200);
  for (std::size_t i = 0; i < 200; ++i) BOOST_CHECK_EQUAL(h[i], 0u);
}

BOOST_AUTO_TEST_CASE(claimed_memory_survives_arena_destruction) {
  std::shared_ptr<int[]> survivor;
  {
    Arena tmp(std::pmr::new_delete_resource(), /*page_size=*/256);
    survivor = tmp.claim<int>(10);
    for (int i = 0; i < 10; ++i) survivor[i] = -i;
  }
  // the aliasing handle keeps its page alive past the Arena
  for (int i = 0; i < 10; ++i) BOOST_CHECK_EQUAL(survivor[i], -i);
}

BOOST_AUTO_TEST_CASE(arena_resource_is_identity_equal) {
  Arena a;
  ArenaResource r1(&a);
  ArenaResource r2(&a);
  BOOST_CHECK(r1.is_equal(r1));
  BOOST_CHECK(!r1.is_equal(r2));
}

BOOST_AUTO_TEST_SUITE_END()
