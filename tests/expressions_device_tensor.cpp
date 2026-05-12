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

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_DEVICE
