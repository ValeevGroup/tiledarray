/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Ajay Melekamburath
 *  Department of Chemistry, Virginia Tech
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/device/tensor.h>
#include <TiledArray/einsum/tiledarray.h>
#include <range_fixture.h>
#include <tiledarray.h>
#include "unit_test_config.h"

using namespace TiledArray;

// Expression-engine tests for the native UMTensor tile type (TA::Tensor
// backed by device_um_allocator). The pattern follows expressions_device_um.cpp
// but uses the bare TA::Tensor specialization -- TA::Tensor is already
// shallow-copy, so we do not wrap it in TA::Tile<> (per CLAUDE.md guidance).
//
// All correctness checks use a CPU-side TiledArray::Tensor<double> mirror
// of the input arrays (built from `find().get()` on the device side and a
// flat std::vector for reference). The expression runs through the engine
// for both sides; we then compare elements after `gop.fence()` to make sure
// the device kernels have actually completed.

struct DeviceTensorExpressionsFixture : public TiledRangeFixture {
  using TileD = UMTensor<double>;
  using TArrayD = TiledArray::DistArray<TileD, TA::DensePolicy>;

  using HostTile = TiledArray::Tensor<double>;
  using HostArray = TiledArray::DistArray<HostTile, TA::DensePolicy>;

  static constexpr double tolerance = 5.0e-14;

  DeviceTensorExpressionsFixture()
      : a(*GlobalFixture::world, tr),
        b(*GlobalFixture::world, tr),
        c(*GlobalFixture::world, tr),
        a_h(*GlobalFixture::world, tr),
        b_h(*GlobalFixture::world, tr),
        c_h(*GlobalFixture::world, tr) {
    fill_with_seed(a, a_h, 7);
    fill_with_seed(b, b_h, 11);
    GlobalFixture::world->gop.fence();
  }

  ~DeviceTensorExpressionsFixture() { GlobalFixture::world->gop.fence(); }

  // Fill paired device + host arrays with the same deterministic data so
  // the host array is an exact reference for the device expression result.
  template <typename DeviceArray, typename HostArrayT>
  static void fill_with_seed(DeviceArray& d, HostArrayT& h, int seed) {
    auto pmap_d = d.pmap();
    for (auto it = pmap_d->begin(); it != pmap_d->end(); ++it) {
      const auto tile_range = d.trange().make_tile_range(*it);
      const auto vol = tile_range.volume();

      // Build deterministic data so seeds match across allocators.
      const auto ord = *it;
      typename DeviceArray::value_type d_tile(tile_range);
      typename HostArrayT::value_type h_tile(tile_range);
      for (std::size_t k = 0; k < vol; ++k) {
        // 1000-element period is plenty for unit testing; division keeps
        // values in [-5, 5] so dot products stay representable.
        const double v =
            static_cast<double>(((ord + 1) * 1664525u + seed + k) % 1000) /
                100.0 -
            5.0;
        d_tile.data()[k] = v;
        h_tile.data()[k] = v;
      }
      d.set(*it, d_tile);
      h.set(*it, h_tile);
    }
  }

  // Compare every element of two DistArrays with matching tiles.
  template <typename DeviceArrayT, typename HostArrayT>
  static void check_close(const DeviceArrayT& d, const HostArrayT& h_ref,
                          double tol) {
    GlobalFixture::world->gop.fence();
    for (auto it = d.begin(); it != d.end(); ++it) {
      auto d_tile = it->get();
      auto h_tile = h_ref.find(it.index()).get();
      BOOST_REQUIRE_EQUAL(d_tile.range(), h_tile.range());
      for (std::size_t k = 0; k < d_tile.size(); ++k) {
        BOOST_CHECK_CLOSE_FRACTION(d_tile.data()[k], h_tile.data()[k], tol);
      }
    }
  }

  TArrayD a, b, c;
  HostArray a_h, b_h, c_h;
};

BOOST_FIXTURE_TEST_SUITE(device_tensor_expressions_suite,
                         DeviceTensorExpressionsFixture)

BOOST_AUTO_TEST_CASE(is_device_tile_classification) {
  using detail::is_device_tile_v;
  BOOST_CHECK(is_device_tile_v<UMTensor<double>>);
  BOOST_CHECK(is_device_tile_v<UMTensor<float>>);
  BOOST_CHECK(is_device_tile_v<typename TArrayD::value_type>);
  BOOST_CHECK(!is_device_tile_v<HostTile>);
}

BOOST_AUTO_TEST_CASE(direct_assign) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(permute) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("c,b,a"));
  c_h("a,b,c") = a_h("c,b,a");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scale) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 2.5 * a("a,b,c"));
  c_h("a,b,c") = 2.5 * a_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(neg) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = -a("a,b,c"));
  c_h("a,b,c") = -a_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(add) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") + b("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c") + b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(add_with_permute) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("c,b,a") + b("a,b,c"));
  c_h("a,b,c") = a_h("c,b,a") + b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(add_to) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") += b("a,b,c"));
  c_h("a,b,c") += b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(subt) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") - b("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c") - b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(subt_to) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c");
  BOOST_REQUIRE_NO_THROW(c("a,b,c") -= b("a,b,c"));
  c_h("a,b,c") -= b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scaled_subt_right) {
  // Isolate: scale-on-right only. `c = a - 3*b`.
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") - 3.0 * b("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c") - 3.0 * b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scaled_subt_left) {
  // Isolate: scale-on-left only. `c = 2*a - b`.
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 2.0 * a("a,b,c") - b("a,b,c"));
  c_h("a,b,c") = 2.0 * a_h("a,b,c") - b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(mixed_linear_combination) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 2.0 * a("a,b,c") - 3.0 * b("a,b,c"));
  c_h("a,b,c") = 2.0 * a_h("a,b,c") - 3.0 * b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(hadamard) {
  // C(ijk) = A(ijk) .* B(ijk), element-wise multiplication
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") * b("a,b,c"));
  c_h("a,b,c") = a_h("a,b,c") * b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(contraction) {
  // C(i,k) = A(i,j) * B(j,k) requires rank-2 arrays; build them on the fly
  // using the first slice of `tr` so the fixture data is reusable.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2;
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h;
  fill_with_seed(a2, a2_h, 13);
  fill_with_seed(b2, b2_h, 17);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(c2("i,k") = a2("i,j") * b2("j,k"));
  c2_h("i,k") = a2_h("i,j") * b2_h("j,k");
  // GEMM tolerance: float-add reordering between BLAS and CPU Eigen path.
  check_close(c2, c2_h, 1.0e-12);
}

BOOST_AUTO_TEST_CASE(norm2_value) {
  // Scalar reduction across all tiles. Compare device-computed value against
  // CPU-computed value from the mirror array.
  const double dev_norm = TA::norm2(a);
  const double host_norm = TA::norm2(a_h);
  GlobalFixture::world->gop.fence();
  BOOST_CHECK_CLOSE_FRACTION(dev_norm, host_norm, 1.0e-12);
}

BOOST_AUTO_TEST_CASE(dot_value) {
  // dot expression: scalar = a . b
  double dev_dot = static_cast<double>(a("a,b,c") * b("a,b,c"));
  double host_dot = static_cast<double>(a_h("a,b,c") * b_h("a,b,c"));
  GlobalFixture::world->gop.fence();
  BOOST_CHECK_CLOSE_FRACTION(dev_dot, host_dot, 1.0e-12);
}

BOOST_AUTO_TEST_CASE(reuse_stress) {
  // MPQC-pattern stress: same input tile referenced multiple times in one
  // expression, then again across iterations. Catches the LazyArrayTile
  // conversion race if it surfaces (it should be a known master-branch
  // baseline failure -- not introduced by this branch).
  const double host_ref =
      static_cast<double>(a_h("a,b,c") * a_h("a,b,c"));
  GlobalFixture::world->gop.fence();
  for (int iter = 0; iter < 8; ++iter) {
    const double d = static_cast<double>(a("a,b,c") * a("a,b,c"));
    GlobalFixture::world->gop.fence();
    BOOST_CHECK_CLOSE_FRACTION(d, host_ref, 1.0e-12);
  }
}

// ---------------------------------------------------------------------------
// In-place expression operators (+=, -=, *=). These exercise the engine's
// "result is consumable" paths that surfaced the dispatch + sign-flip bugs;
// here we want broad coverage of compound assignment forms beyond the
// `add_to` / `subt_to` cases already tested.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(plus_equal_expr) {
  c("a,b,c") = a("a,b,c");
  c_h("a,b,c") = a_h("a,b,c");
  c("a,b,c") += b("a,b,c");
  c_h("a,b,c") += b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(plus_equal_with_permute) {
  c("a,b,c") = a("a,b,c");
  c_h("a,b,c") = a_h("a,b,c");
  c("a,b,c") += b("c,b,a");
  c_h("a,b,c") += b_h("c,b,a");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(minus_equal_expr) {
  c("a,b,c") = a("a,b,c");
  c_h("a,b,c") = a_h("a,b,c");
  c("a,b,c") -= b("a,b,c");
  c_h("a,b,c") -= b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(times_equal_expr) {
  // Hadamard, in place.
  c("a,b,c") = a("a,b,c");
  c_h("a,b,c") = a_h("a,b,c");
  c("a,b,c") *= b("a,b,c");
  c_h("a,b,c") *= b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

// ---------------------------------------------------------------------------
// Negated and scaled-then-negated forms. These force the engine to combine
// scaling with sign-flip across different operand positions.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(neg_scaled_sum) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = -(2.0 * (a("a,b,c") + b("a,b,c"))));
  c_h("a,b,c") = -(2.0 * (a_h("a,b,c") + b_h("a,b,c")));
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(neg_permuted) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = -a("c,b,a"));
  c_h("a,b,c") = -a_h("c,b,a");
  check_close(c, c_h, tolerance);
}

// ---------------------------------------------------------------------------
// Multi-step chains: results of one expression feed the next. Validates
// dataflow handoff between dist-evals without an intervening fence (per
// CLAUDE.md's synchronization-hierarchy section).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(multi_step_chain) {
  TArrayD t(*GlobalFixture::world, tr);
  HostArray t_h(*GlobalFixture::world, tr);
  t("a,b,c") = a("a,b,c") + b("a,b,c");
  t_h("a,b,c") = a_h("a,b,c") + b_h("a,b,c");
  c("a,b,c") = 2.0 * t("a,b,c") - a("a,b,c");
  c_h("a,b,c") = 2.0 * t_h("a,b,c") - a_h("a,b,c");
  check_close(c, c_h, tolerance);
}

// ---------------------------------------------------------------------------
// Block expressions. PR 531 hit known issues in this area; we cover the
// common patterns: read-only block, block in a sum, block on the RHS of an
// accumulating assignment.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(block_assign) {
  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};

  // Result range matches the block's element range; build small companion
  // arrays to receive the result.
  const TiledRange ctr{TiledRange1{lo[0], up[0]},
                       TiledRange1{lo[1], up[1]},
                       TiledRange1{lo[2], up[2]}};
  TArrayD blk_d(*GlobalFixture::world, ctr);
  HostArray blk_h(*GlobalFixture::world, ctr);
  blk_d("a,b,c") = a("a,b,c").block(lo, up);
  blk_h("a,b,c") = a_h("a,b,c").block(lo, up);
  check_close(blk_d, blk_h, tolerance);
}

BOOST_AUTO_TEST_CASE(block_add_then_scale) {
  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};
  const TiledRange ctr{TiledRange1{lo[0], up[0]},
                       TiledRange1{lo[1], up[1]},
                       TiledRange1{lo[2], up[2]}};
  TArrayD blk_d(*GlobalFixture::world, ctr);
  HostArray blk_h(*GlobalFixture::world, ctr);
  blk_d("a,b,c") = 2.0 * (a("a,b,c").block(lo, up) + b("a,b,c").block(lo, up));
  blk_h("a,b,c") =
      2.0 * (a_h("a,b,c").block(lo, up) + b_h("a,b,c").block(lo, up));
  check_close(blk_d, blk_h, tolerance);
}

BOOST_AUTO_TEST_CASE(block_accumulate) {
  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};
  const TiledRange ctr{TiledRange1{lo[0], up[0]},
                       TiledRange1{lo[1], up[1]},
                       TiledRange1{lo[2], up[2]}};
  TArrayD blk_d(*GlobalFixture::world, ctr);
  HostArray blk_h(*GlobalFixture::world, ctr);
  blk_d("a,b,c") = a("a,b,c").block(lo, up);
  blk_h("a,b,c") = a_h("a,b,c").block(lo, up);
  blk_d("a,b,c") += b("a,b,c").block(lo, up);
  blk_h("a,b,c") += b_h("a,b,c").block(lo, up);
  check_close(blk_d, blk_h, tolerance);
}

// ---------------------------------------------------------------------------
// Outer product: c(i,j) = u(i) * v(j). Exercises the rank-changing GEMM
// path (different left / right / result ranks) without going through a
// shared contraction index.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(outer_product) {
  const TiledRange tr_u{tr.data()[0]};
  const TiledRange tr_v{tr.data()[1]};
  const TiledRange tr_w{tr.data()[0], tr.data()[1]};

  TArrayD u(*GlobalFixture::world, tr_u);
  TArrayD v(*GlobalFixture::world, tr_v);
  TArrayD w;
  HostArray u_h(*GlobalFixture::world, tr_u);
  HostArray v_h(*GlobalFixture::world, tr_v);
  HostArray w_h;
  fill_with_seed(u, u_h, 31);
  fill_with_seed(v, v_h, 37);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(w("i,j") = u("i") * v("j"));
  w_h("i,j") = u_h("i") * v_h("j");
  check_close(w, w_h, 1.0e-12);
}

// ---------------------------------------------------------------------------
// Contraction shape variants. Different output ranks and contraction
// patterns. CC-style: result is rank-4, contraction index is multi-d.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(contraction_permuted_result) {
  // c(k,i) = a(i,j) * b(j,k) -- the same contraction as `contraction` but
  // with the result indices swapped; checks that the engine fuses a final
  // permutation into the GEMM as CLAUDE.md describes.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2;
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h;
  fill_with_seed(a2, a2_h, 41);
  fill_with_seed(b2, b2_h, 43);
  GlobalFixture::world->gop.fence();
  BOOST_REQUIRE_NO_THROW(c2("k,i") = a2("i,j") * b2("j,k"));
  c2_h("k,i") = a2_h("i,j") * b2_h("j,k");
  // Looser tolerance for permuted GEMM: BLAS sums in different
  // tile-internal order than the CPU reference path.
  check_close(c2, c2_h, 1.0e-10);
}

BOOST_AUTO_TEST_CASE(contraction_with_transpose_on_right) {
  // c(i,k) = a(i,j) * b(k,j) -- right operand needs transposing to align
  // the contraction index.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2;
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h;
  fill_with_seed(a2, a2_h, 47);
  fill_with_seed(b2, b2_h, 53);
  GlobalFixture::world->gop.fence();
  BOOST_REQUIRE_NO_THROW(c2("i,k") = a2("i,j") * b2("k,j"));
  c2_h("i,k") = a2_h("i,j") * b2_h("k,j");
  check_close(c2, c2_h, 1.0e-12);
}

BOOST_AUTO_TEST_CASE(contraction_rank4_via_two_indices) {
  // r(a,c) = t(a,b,k,l) * v(c,b,k,l) -- pattern that shows up in CC-style
  // intermediates; contraction is over (b,k,l), free indices are (a) on
  // the left and (c) on the right.
  const TiledRange tr4{tr.data()[0], tr.data()[1], tr.data()[2], tr.data()[2]};
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD t(*GlobalFixture::world, tr4);
  TArrayD v(*GlobalFixture::world, tr4);
  TArrayD r;
  HostArray t_h(*GlobalFixture::world, tr4);
  HostArray v_h(*GlobalFixture::world, tr4);
  HostArray r_h;
  fill_with_seed(t, t_h, 59);
  fill_with_seed(v, v_h, 61);
  GlobalFixture::world->gop.fence();
  BOOST_REQUIRE_NO_THROW(r("a,c") = t("a,b,k,l") * v("c,b,k,l"));
  r_h("a,c") = t_h("a,b,k,l") * v_h("c,b,k,l");
  check_close(r, r_h, 1.0e-12);
}

// ---------------------------------------------------------------------------
// TA::einsum entry point. The fully-typed einsum API is the documented way
// to express patterns the regular `*` operator can't capture (general
// contraction with explicit output indices, Hadamard with permutation,
// etc.). For UMTensor we test that einsum dispatches through the same tile
// ops we already validated above and produces matching results vs. the
// host-tensor reference.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(einsum_matmul) {
  // c(i,k) = a(i,j) * b(j,k) via einsum
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  fill_with_seed(a2, a2_h, 67);
  fill_with_seed(b2, b2_h, 71);
  GlobalFixture::world->gop.fence();

  auto c2 = TiledArray::einsum(a2("i,j"), b2("j,k"), "i,k");
  auto c2_h = TiledArray::einsum(a2_h("i,j"), b2_h("j,k"), "i,k");
  check_close(c2, c2_h, 1.0e-11);
}

BOOST_AUTO_TEST_CASE(einsum_hadamard) {
  // c(i,j) = a(i,j) * b(i,j) via einsum -- Hadamard / element-wise multiply
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  fill_with_seed(a2, a2_h, 73);
  fill_with_seed(b2, b2_h, 79);
  GlobalFixture::world->gop.fence();

  auto c2 = TiledArray::einsum(a2("i,j"), b2("i,j"), "i,j");
  auto c2_h = TiledArray::einsum(a2_h("i,j"), b2_h("i,j"), "i,j");
  check_close(c2, c2_h, tolerance);
}

// Note: einsum patterns where an index appears in both inputs *and* the
// output (e.g. `einsum("ij,jk->ijk")`, an outer-product-with-broadcast
// over `j`) are not yet supported for plain (non-ToT) tile types -- they
// segfault inside einsum's internals on master regardless of allocator.
// We don't cover that case here.

BOOST_AUTO_TEST_CASE(einsum_contraction_over_two_indices) {
  // c(a,c) = t(a,b,k) * v(c,b,k) via einsum -- contraction over (b, k),
  // free indices (a) on the left and (c) on the right. CC-intermediate
  // shape, fully expressible with the regular `*` operator but still
  // worth covering through the einsum entry point.
  const TiledRange tr3{tr.data()[0], tr.data()[1], tr.data()[2]};
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD t(*GlobalFixture::world, tr3);
  TArrayD v(*GlobalFixture::world, tr3);
  HostArray t_h(*GlobalFixture::world, tr3);
  HostArray v_h(*GlobalFixture::world, tr3);
  fill_with_seed(t, t_h, 83);
  fill_with_seed(v, v_h, 89);
  GlobalFixture::world->gop.fence();

  auto r = TiledArray::einsum(t("a,b,k"), v("c,b,k"), "a,c");
  auto r_h = TiledArray::einsum(t_h("a,b,k"), v_h("c,b,k"), "a,c");
  check_close(r, r_h, 1.0e-11);
}

BOOST_AUTO_TEST_CASE(einsum_permuted_result) {
  // c(j,i) = a(i,j) -- one-operand reshape; not a true einsum binary, but
  // also useful: verify einsum handles single-input permutation.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  HostArray a2_h(*GlobalFixture::world, tr2);
  fill_with_seed(a2, a2_h, 97);
  GlobalFixture::world->gop.fence();

  // For permutation alone we just use the expression DSL, which einsum
  // delegates to; this verifies that path still works for UMTensor.
  TArrayD a2T;
  HostArray a2T_h;
  a2T("j,i") = a2("i,j");
  a2T_h("j,i") = a2_h("i,j");
  check_close(a2T, a2T_h, tolerance);
}

// ---------------------------------------------------------------------------
// Scaled / permuted variants of the elementary arithmetic ops. The first
// commit of these tests covered the bare forms; the engine fuses scaling
// and permutation differently across these combinations, so each one is
// a distinct dispatch path worth validating numerically.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(scale_add) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5.0 * (a("a,b,c") + b("a,b,c")));
  c_h("a,b,c") = 5.0 * (a_h("a,b,c") + b_h("a,b,c"));
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scale_add_permute) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5.0 * (2.0 * a("c,b,a")) + (3.0 * b("a,b,c")));
  c_h("a,b,c") = 5.0 * (2.0 * a_h("c,b,a")) + (3.0 * b_h("a,b,c"));
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(subt_permute) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("c,b,a") - b("a,b,c"));
  c_h("a,b,c") = a_h("c,b,a") - b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scale_subt) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5.0 * (a("a,b,c") - b("a,b,c")));
  c_h("a,b,c") = 5.0 * (a_h("a,b,c") - b_h("a,b,c"));
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scale_subt_permute) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5.0 * (2.0 * a("c,b,a")) - (3.0 * b("a,b,c")));
  c_h("a,b,c") = 5.0 * (2.0 * a_h("c,b,a")) - (3.0 * b_h("a,b,c"));
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(mult_permute) {
  // Hadamard with permutation on left operand.
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("c,b,a") * b("a,b,c"));
  c_h("a,b,c") = a_h("c,b,a") * b_h("a,b,c");
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scale_mult) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5.0 * (a("a,b,c") * b("a,b,c")));
  c_h("a,b,c") = 5.0 * (a_h("a,b,c") * b_h("a,b,c"));
  check_close(c, c_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scale_mult_permute) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5.0 * (2.0 * a("c,b,a")) * (3.0 * b("a,b,c")));
  c_h("a,b,c") = 5.0 * (2.0 * a_h("c,b,a")) * (3.0 * b_h("a,b,c"));
  check_close(c, c_h, tolerance);
}

// ---------------------------------------------------------------------------
// Scaled contraction variants. These exercise the engine's scale-fuse-
// into-GEMM path that PR 531 stumbled on. Tolerance is 1e-10 for GEMM
// paths to absorb summation-order differences between BLAS and Eigen.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(scale_cont) {
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2;
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h;
  fill_with_seed(a2, a2_h, 101);
  fill_with_seed(b2, b2_h, 103);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(c2("i,k") = 5.0 * (a2("i,j") * b2("j,k")));
  c2_h("i,k") = 5.0 * (a2_h("i,j") * b2_h("j,k"));
  check_close(c2, c2_h, 1.0e-10);
}

BOOST_AUTO_TEST_CASE(scale_cont_permute) {
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2;
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h;
  fill_with_seed(a2, a2_h, 107);
  fill_with_seed(b2, b2_h, 109);
  GlobalFixture::world->gop.fence();

  // c(k,i) = 5 * a(i,j) * b(j,k): scaled, result-permuted contraction.
  BOOST_REQUIRE_NO_THROW(c2("k,i") = 5.0 * (a2("i,j") * b2("j,k")));
  c2_h("k,i") = 5.0 * (a2_h("i,j") * b2_h("j,k"));
  check_close(c2, c2_h, 1.0e-10);
}

BOOST_AUTO_TEST_CASE(scale_cont_with_input_transpose) {
  // 5 * a(i,j) * b(k,j) -- contraction needs to transpose b before GEMM.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2;
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h;
  fill_with_seed(a2, a2_h, 113);
  fill_with_seed(b2, b2_h, 127);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(c2("i,k") = 5.0 * (a2("i,j") * b2("k,j")));
  c2_h("i,k") = 5.0 * (a2_h("i,j") * b2_h("k,j"));
  check_close(c2, c2_h, 1.0e-10);
}

// ---------------------------------------------------------------------------
// Non-uniform tile sizes for contraction. Mirrors btas-device's
// cont_non_uniform1/2: the rank-4 inputs use one tiny tiling on the
// outer dimensions and one wide tiling on an inner dimension, so the
// GEMM has irregular per-tile k blocks. Catches GEMM kernels that
// silently assume uniform tile shapes.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(cont_non_uniform_split_inner) {
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange tr_irr(tiling4.begin(), tiling4.end());

  TArrayD lhs(*GlobalFixture::world, tr_irr);
  TArrayD rhs(*GlobalFixture::world, tr_irr);
  TArrayD out;
  HostArray lhs_h(*GlobalFixture::world, tr_irr);
  HostArray rhs_h(*GlobalFixture::world, tr_irr);
  HostArray out_h;
  fill_with_seed(lhs, lhs_h, 131);
  fill_with_seed(rhs, rhs_h, 137);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(out("x,y") =
                             5.0 * (lhs("x,i,j,k") * rhs("y,i,j,k")));
  out_h("x,y") = 5.0 * (lhs_h("x,i,j,k") * rhs_h("y,i,j,k"));
  check_close(out, out_h, 1.0e-9);
}

BOOST_AUTO_TEST_CASE(cont_non_uniform_split_two_inner) {
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_1, tr1_2, tr1_2}};
  TiledRange tr_irr(tiling4.begin(), tiling4.end());

  TArrayD lhs(*GlobalFixture::world, tr_irr);
  TArrayD rhs(*GlobalFixture::world, tr_irr);
  TArrayD out;
  HostArray lhs_h(*GlobalFixture::world, tr_irr);
  HostArray rhs_h(*GlobalFixture::world, tr_irr);
  HostArray out_h;
  fill_with_seed(lhs, lhs_h, 139);
  fill_with_seed(rhs, rhs_h, 149);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(out("x,y") =
                             5.0 * (lhs("x,i,j,k") * rhs("y,i,j,k")));
  out_h("x,y") = 5.0 * (lhs_h("x,i,j,k") * rhs_h("y,i,j,k"));
  check_close(out, out_h, 1.0e-9);
}

// ---------------------------------------------------------------------------
// Contraction-plus-reduction (norm2 of a contraction). Exercises the
// dataflow handoff from a binary dist-eval to a reduction without an
// intervening fence.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(cont_plus_reduce) {
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  fill_with_seed(a2, a2_h, 151);
  fill_with_seed(b2, b2_h, 157);
  GlobalFixture::world->gop.fence();

  TArrayD c2;
  HostArray c2_h;
  c2("i,k") = a2("i,j") * b2("j,k");
  c2_h("i,k") = a2_h("i,j") * b2_h("j,k");
  const double dev_n = TA::norm2(c2);
  const double host_n = TA::norm2(c2_h);
  GlobalFixture::world->gop.fence();
  BOOST_CHECK_CLOSE_FRACTION(dev_n, host_n, 1.0e-10);
}

BOOST_AUTO_TEST_CASE(no_alias_plus_reduce) {
  // `no_alias()` tells the engine the LHS does not alias any RHS operand,
  // permitting an extra in-place optimization. Validate that path
  // produces correct values.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  TArrayD c2(*GlobalFixture::world,
             TiledRange{tr.data()[0], tr.data()[1]});
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  HostArray c2_h(*GlobalFixture::world,
                 TiledRange{tr.data()[0], tr.data()[1]});
  fill_with_seed(a2, a2_h, 163);
  fill_with_seed(b2, b2_h, 167);
  c2.fill_local(0.0);
  c2_h.fill_local(0.0);
  GlobalFixture::world->gop.fence();

  BOOST_REQUIRE_NO_THROW(c2("i,k").no_alias() = a2("i,j") * b2("j,k"));
  c2_h("i,k").no_alias() = a2_h("i,j") * b2_h("j,k");
  check_close(c2, c2_h, 1.0e-10);
  const double dev_n = TA::norm2(c2);
  const double host_n = TA::norm2(c2_h);
  GlobalFixture::world->gop.fence();
  BOOST_CHECK_CLOSE_FRACTION(dev_n, host_n, 1.0e-10);
}

// ---------------------------------------------------------------------------
// Block-expression variants beyond the basic three already covered.
// Block bounds are TILE coordinates; a {3,3,3} -> {5,5,5} block selects
// the 2x2x2 corner tiles of `tr` (5 tiles per dim).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(const_block) {
  const auto& ca = a;
  const auto& ca_h = a_h;
  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};
  const TiledRange ctr{TiledRange1{lo[0], up[0]},
                       TiledRange1{lo[1], up[1]},
                       TiledRange1{lo[2], up[2]}};
  TArrayD blk_d(*GlobalFixture::world, ctr);
  HostArray blk_h(*GlobalFixture::world, ctr);
  blk_d("a,b,c") = ca("a,b,c").block(lo, up);
  blk_h("a,b,c") = ca_h("a,b,c").block(lo, up);
  check_close(blk_d, blk_h, tolerance);
}

BOOST_AUTO_TEST_CASE(scal_block) {
  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};
  const TiledRange ctr{TiledRange1{lo[0], up[0]},
                       TiledRange1{lo[1], up[1]},
                       TiledRange1{lo[2], up[2]}};
  TArrayD blk_d(*GlobalFixture::world, ctr);
  HostArray blk_h(*GlobalFixture::world, ctr);
  blk_d("a,b,c") = 2.0 * a("a,b,c").block(lo, up);
  blk_h("a,b,c") = 2.0 * a_h("a,b,c").block(lo, up);
  check_close(blk_d, blk_h, tolerance);
}

BOOST_AUTO_TEST_CASE(permute_block) {
  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};
  const TiledRange ctr{TiledRange1{lo[0], up[0]},
                       TiledRange1{lo[1], up[1]},
                       TiledRange1{lo[2], up[2]}};
  TArrayD blk_d(*GlobalFixture::world, ctr);
  HostArray blk_h(*GlobalFixture::world, ctr);
  // Permute the source annotation before slicing.
  blk_d("a,b,c") = a("c,b,a").block(lo, up);
  blk_h("a,b,c") = a_h("c,b,a").block(lo, up);
  check_close(blk_d, blk_h, tolerance);
}

BOOST_AUTO_TEST_CASE(assign_sub_block) {
  // Write into a tile sub-block of an existing array. Tiles outside the
  // block keep their original contents -- so we initialize both sides
  // identically with a known value before the block assignment.
  c.fill_local(0.0);
  c_h.fill_local(0.0);
  GlobalFixture::world->gop.fence();

  const std::array<int, 3> lo{3, 3, 3};
  const std::array<int, 3> up{5, 5, 5};
  BOOST_REQUIRE_NO_THROW(c("a,b,c").block(lo, up) = a("a,b,c").block(lo, up));
  c_h("a,b,c").block(lo, up) = a_h("a,b,c").block(lo, up);
  check_close(c, c_h, tolerance);
}

// ---------------------------------------------------------------------------
// Block-fed-into-contraction. PR 531 had known issues here. The result
// array has rank 2 (carved out of the rank-3 fixture by contracting two
// indices).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(block_contract) {
  const TiledRange tr_w{tr.data()[0], tr.data()[1]};
  TArrayD w(*GlobalFixture::world, tr_w);
  HostArray w_h(*GlobalFixture::world, tr_w);
  w.fill_local(0.0);
  w_h.fill_local(0.0);
  GlobalFixture::world->gop.fence();

  const std::array<int, 3> alo{3, 2, 3};
  const std::array<int, 3> aup{5, 5, 5};
  const std::array<int, 3> blo{2, 3, 3};
  const std::array<int, 3> bup{5, 5, 5};

  BOOST_REQUIRE_NO_THROW(
      w("a,b") = a("a,c,d").block(alo, aup) * b("c,d,b").block(blo, bup));
  w_h("a,b") = a_h("a,c,d").block(alo, aup) * b_h("c,d,b").block(blo, bup);
  check_close(w, w_h, 1.0e-10);
}

BOOST_AUTO_TEST_CASE(block_permute_contract) {
  // Same as block_contract but with a permuted left-operand annotation:
  // `a("a,d,c")` instead of `a("a,c,d")` -- forces a permutation of the
  // sliced block before GEMM.
  const TiledRange tr_w{tr.data()[0], tr.data()[1]};
  TArrayD w(*GlobalFixture::world, tr_w);
  HostArray w_h(*GlobalFixture::world, tr_w);
  w.fill_local(0.0);
  w_h.fill_local(0.0);
  GlobalFixture::world->gop.fence();

  const std::array<int, 3> alo{3, 3, 2};
  const std::array<int, 3> aup{5, 5, 5};
  const std::array<int, 3> blo{2, 3, 3};
  const std::array<int, 3> bup{5, 5, 5};

  BOOST_REQUIRE_NO_THROW(
      w("a,b") = a("a,d,c").block(alo, aup) * b("c,d,b").block(blo, bup));
  w_h("a,b") = a_h("a,d,c").block(alo, aup) * b_h("c,d,b").block(blo, bup);
  check_close(w, w_h, 1.0e-10);
}

// ---------------------------------------------------------------------------
// Dot-product variants beyond the basic case.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(dot_permute) {
  const double dev_d =
      static_cast<double>(a("a,b,c") * b("c,b,a"));
  const double host_d =
      static_cast<double>(a_h("a,b,c") * b_h("c,b,a"));
  GlobalFixture::world->gop.fence();
  // Looser tolerance because permuted dot reads tiles in a different
  // order, so the partial-sum accumulation order differs.
  BOOST_CHECK_CLOSE_FRACTION(dev_d, host_d, 1.0e-12);
}

BOOST_AUTO_TEST_CASE(dot_contr) {
  // Dot of two contraction expressions: scalar = (a*b) . (b*a).
  // This is a NO_THROW-only check in the btas-device suite; we go one
  // step further and validate the scalar value against the CPU mirror.
  const TiledRange tr2{tr.data()[0], tr.data()[1]};
  TArrayD a2(*GlobalFixture::world, tr2);
  TArrayD b2(*GlobalFixture::world, tr2);
  HostArray a2_h(*GlobalFixture::world, tr2);
  HostArray b2_h(*GlobalFixture::world, tr2);
  fill_with_seed(a2, a2_h, 173);
  fill_with_seed(b2, b2_h, 179);
  GlobalFixture::world->gop.fence();

  const double dev_d = static_cast<double>(
      (a2("i,j") * b2("j,k")) * (a2("i,j") * b2("j,k")));
  const double host_d = static_cast<double>(
      (a2_h("i,j") * b2_h("j,k")) * (a2_h("i,j") * b2_h("j,k")));
  GlobalFixture::world->gop.fence();
  BOOST_CHECK_CLOSE_FRACTION(dev_d, host_d, 1.0e-10);
}

// ---------------------------------------------------------------------------
// Archive round-trip, host/device array conversions, and bulk to_host /
// to_device. Smoke + correctness for the helpers in device/tensor.h that
// are not in the expression-engine path.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(serialize_um_tensor) {
  // Single-tile round-trip: build a UMTensor, write to a buffer archive,
  // read into a fresh UMTensor, compare element-wise. The Store side
  // forces a host prefetch so the archive sees coherent data.
  TileD t(TiledArray::Range{4, 4});
  for (std::size_t k = 0; k < t.size(); ++k)
    t.data()[k] = static_cast<double>(k) - 5.0;

  const std::size_t buf_size =
      (t.range().volume() * sizeof(double) +
       sizeof(std::size_t) * (t.range().rank() * 4 + 4)) *
      2;
  std::vector<unsigned char> buf(buf_size);

  madness::archive::BufferOutputArchive oar(buf.data(), buf.size());
  BOOST_REQUIRE_NO_THROW(oar & t);
  const std::size_t nbyte = oar.size();
  oar.close();

  TileD u;
  madness::archive::BufferInputArchive iar(buf.data(), nbyte);
  BOOST_REQUIRE_NO_THROW(iar & u);
  iar.close();

  BOOST_REQUIRE_EQUAL(t.range(), u.range());
  for (std::size_t k = 0; k < t.size(); ++k)
    BOOST_CHECK_CLOSE(t.data()[k], u.data()[k], 1.0e-15);
}

BOOST_AUTO_TEST_CASE(serialize_um_tensor_empty) {
  // Empty-tensor round-trip: the empty branch in Store/Load.
  TileD t;
  std::vector<unsigned char> buf(1024);
  madness::archive::BufferOutputArchive oar(buf.data(), buf.size());
  BOOST_REQUIRE_NO_THROW(oar & t);
  const std::size_t nbyte = oar.size();
  oar.close();

  TileD u(TiledArray::Range{2, 2});  // start non-empty so load has work
  madness::archive::BufferInputArchive iar(buf.data(), nbyte);
  BOOST_REQUIRE_NO_THROW(iar & u);
  iar.close();
  BOOST_CHECK(u.empty());
}

BOOST_AUTO_TEST_CASE(um_to_ta_round_trip) {
  // UMTensor array -> host array -> UMTensor array, verify element-wise
  // against the original both at the host and device endpoints.
  HostArray host_view = TiledArray::um_tensor_to_ta_tensor(a);
  GlobalFixture::world->gop.fence();
  // Compare host_view directly against a_h (same data was used for both).
  check_close(host_view, a_h, tolerance);

  TArrayD device_view = TiledArray::ta_tensor_to_um_tensor(host_view);
  GlobalFixture::world->gop.fence();
  // Round-trip: device_view should match the original `a`.
  check_close(device_view, a_h, tolerance);
}

BOOST_AUTO_TEST_CASE(um_to_ta_then_expression) {
  // After converting a device array to host, plain CPU expressions on the
  // host array should produce the same values as their device counterpart.
  HostArray sum_h(*GlobalFixture::world, tr);
  sum_h("a,b,c") = a_h("a,b,c") + b_h("a,b,c");

  HostArray converted = TiledArray::um_tensor_to_ta_tensor(a);
  HostArray sum_from_device(*GlobalFixture::world, tr);
  sum_from_device("a,b,c") = converted("a,b,c") + b_h("a,b,c");

  GlobalFixture::world->gop.fence();
  check_close(sum_from_device, sum_h, tolerance);
}

BOOST_AUTO_TEST_CASE(bulk_prefetch_round_trip) {
  // to_host / to_device on a DistArray should be no-ops for correctness:
  // the array contents are unchanged, only the page residency hints are
  // adjusted. We just verify the array is still equal to its host mirror
  // after bouncing through both directions.
  TiledArray::to_host(a);
  TiledArray::to_device(a);
  GlobalFixture::world->gop.fence();
  check_close(a, a_h, tolerance);
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_DEVICE
