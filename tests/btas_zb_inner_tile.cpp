/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026 Virginia Tech
 *
 *  btas_zb_inner_tile.cpp
 *
 *  Sniff tests for using btas::Tensor<T, btas::zb::RangeNd<>, ...> as the
 *  inner tile of a TiledArray Tensor-of-Tensor. This is the entry point for
 *  validating Phase 2 of the inner-tile shrink work (see PR
 *  ValeevGroup/BTAS#187); ops not yet validated for the new range type will
 *  surface as compile errors here.
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_BTAS

#include <TiledArray/external/btas.h>
#include <TiledArray/tensor.h>
#include <btas/zb/range.h>

#include "unit_test_config.h"

using namespace TiledArray;

// Inner tile under test: btas::Tensor with zero-based packed range and the
// BTAS default storage (boost::container::small_vector wrapper). Uses extent
// type int16_t / ordinal type int32_t by default.
using bTensorIzb_storage = btas::DEFAULT::storage<int>;
using bTensorIzb = btas::Tensor<int, btas::zb::RangeNd<>, bTensorIzb_storage>;

// Sanity-check the size claim — fail loudly here if range layout drifts.
static_assert(sizeof(btas::zb::RangeNd<>) == 14,
              "zb::RangeNd default layout must remain 14 bytes");

BOOST_AUTO_TEST_SUITE(btas_zb_inner_tile_suite,
                      *boost::unit_test::label("@serial"))

// 1. Bare btas::Tensor parametrized on zb::RangeNd compiles, constructs, and
//    permits element access via raw storage. Anchor for the whole stack.
BOOST_AUTO_TEST_CASE(btas_tensor_basic) {
  bTensorIzb t(3, 4);
  BOOST_REQUIRE_EQUAL(t.range().rank(), 2u);
  BOOST_REQUIRE_EQUAL(t.range().area(), 12u);
  BOOST_REQUIRE_EQUAL(t.range().extent(0), 3);
  BOOST_REQUIRE_EQUAL(t.range().extent(1), 4);

  // Fill via raw pointer (smallest possible surface).
  auto* p = t.data();
  for (std::size_t i = 0; i < t.range().area(); ++i) p[i] = static_cast<int>(i);
  BOOST_CHECK_EQUAL(p[0], 0);
  BOOST_CHECK_EQUAL(p[5], 5);
  BOOST_CHECK_EQUAL(p[11], 11);
}

// 2. Wrap as the element type of a TA::Tensor (the ToT shape) and verify
//    construction + per-element placement of inner tiles works.
BOOST_AUTO_TEST_CASE(tensor_of_btas_tensor_construct) {
  Tensor<bTensorIzb> outer(Range({2ul, 3ul}));
  BOOST_REQUIRE_EQUAL(outer.range().rank(), 2u);
  BOOST_REQUIRE_EQUAL(outer.range().area(), 6u);

  // Each inner tile has non-uniform extents — the load-bearing property of
  // ToT — but the inner range is zero-based.
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      bTensorIzb inner(static_cast<int>(2 + i), static_cast<int>(3 + j));
      for (std::size_t k = 0; k < inner.range().area(); ++k)
        inner.data()[k] = static_cast<int>((i + 1) * 100 + (j + 1) * 10 + k);
      outer(i, j) = std::move(inner);
    }
  }

  BOOST_CHECK_EQUAL(outer(0, 0).range().rank(), 2u);
  BOOST_CHECK_EQUAL(outer(0, 0).range().area(), 2 * 3);
  BOOST_CHECK_EQUAL(outer(1, 2).range().area(), 3 * 5);
  BOOST_CHECK_EQUAL(outer(0, 0).data()[0], 110);
  BOOST_CHECK_EQUAL(outer(1, 2).data()[3 * 5 - 1], 230 + 14);
}

// 3. Helper: build a Tensor<bTensorIzb> with uniform inner-tile extents (so
//    both operands are congruent, which the binary ops require).
static Tensor<bTensorIzb> make_uniform_TobT_zb(const Range& r, int fill) {
  Tensor<bTensorIzb> tensor(r);
  for (decltype(r.extent(0)) i = 0; i < r.extent(0); ++i) {
    for (decltype(r.extent(1)) j = 0; j < r.extent(1); ++j) {
      bTensorIzb inner(static_cast<int>(4), static_cast<int>(5));
      for (std::size_t k = 0; k < inner.range().area(); ++k)
        inner.data()[k] = fill + static_cast<int>(k);
      tensor(i, j) = std::move(inner);
    }
  }
  return tensor;
}

// 4. Exercise the actual ToT ops we care about — these instantiate the
//    cross-namespace operator infrastructure (btas::operator- via shared
//    body) inside TA::Tensor<bTensorIzb>::{subt,add,scale,neg} lambdas.
//    *This* is the test that proves btas::Tensor with btas::zb::RangeNd is
//    usable as a ToT inner tile end-to-end.
BOOST_AUTO_TEST_CASE(tot_subt_with_zb_inner) {
  auto a = make_uniform_TobT_zb(Range({2ul, 3ul}), 10);
  auto b = make_uniform_TobT_zb(Range({2ul, 3ul}), 1);

  Tensor<bTensorIzb> c;
  BOOST_REQUIRE_NO_THROW(c = a.subt(b));

  BOOST_REQUIRE_EQUAL(c.range().area(), 6u);
  // Inner tile (0,0): each element should be a_inner[k] - b_inner[k] = 9.
  for (std::size_t k = 0; k < c(0, 0).range().area(); ++k)
    BOOST_CHECK_EQUAL(c(0, 0).data()[k], 9);
}

BOOST_AUTO_TEST_CASE(tot_add_with_zb_inner) {
  auto a = make_uniform_TobT_zb(Range({2ul, 3ul}), 10);
  auto b = make_uniform_TobT_zb(Range({2ul, 3ul}), 5);

  Tensor<bTensorIzb> c;
  BOOST_REQUIRE_NO_THROW(c = a.add(b));

  // First inner-tile element: (10 + 0) + (5 + 0) = 15.
  BOOST_CHECK_EQUAL(c(0, 0).data()[0], 15);
}

BOOST_AUTO_TEST_CASE(tot_scale_with_zb_inner) {
  auto a = make_uniform_TobT_zb(Range({2ul, 3ul}), 1);

  Tensor<bTensorIzb> c;
  BOOST_REQUIRE_NO_THROW(c = a.scale(3));

  BOOST_CHECK_EQUAL(c(0, 0).data()[0], 3);
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_BTAS
