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

// Dense matrix-multiply benchmark using the native UMTensor tile type
// (TA::Tensor backed by device_um_allocator). Companion to the btas-based
// ta_dense_device.cpp; same shape + reporting, but the tile is bare
// `UMTensor<T>` -- no `TA::Tile<>` wrapper -- and the data flows through
// the device tile-op overloads in src/TiledArray/device/tensor.h.
//
// Usage:
//   ta_dense_um_tensor Nm Bm Nn Bn Nk Bk [nrepeat=5]
//
// Computes c(Nm,Nn) = a(Nm,Nk) * b(Nk,Nn) with each dimension blocked by
// Bm/Bn/Bk. Default scalar type is double; nrepeat iterations are timed
// for an average GFLOPS reading.

#include <TiledArray/device/tensor.h>
#include <tiledarray.h>

#ifdef TILEDARRAY_HAS_CUDA
#include <cuda_profiler_api.h>
#endif

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

template <typename T>
void run(TiledArray::World &world, long Nm, long Bm, long Nn, long Bn, long Nk,
         long Bk, long nrepeat) {
  using TA::DistArray;
  using TA::TiledRange;
  using TA::TiledRange1;
  using TA::UMTensor;
  using TileT = UMTensor<T>;
  using ArrayT = DistArray<TileT, TA::DensePolicy>;

  constexpr bool complex_T = TA::detail::is_complex_v<T>;
  // GEMM flops: 2 * M * N * K (8 * for complex).
  const std::int64_t nflops =
      (complex_T ? 8 : 2) * static_cast<std::int64_t>(Nm) *
      static_cast<std::int64_t>(Nn) * static_cast<std::int64_t>(Nk);

  auto blocking = [](long N, long B) {
    std::vector<unsigned int> v;
    for (long i = 0; i <= N; i += B) v.push_back(static_cast<unsigned int>(i));
    return v;
  };
  auto blk_m = blocking(Nm, Bm);
  auto blk_n = blocking(Nn, Bn);
  auto blk_k = blocking(Nk, Bk);

  TiledRange trange_a({TiledRange1(blk_m.begin(), blk_m.end()),
                       TiledRange1(blk_k.begin(), blk_k.end())});
  TiledRange trange_b({TiledRange1(blk_k.begin(), blk_k.end()),
                       TiledRange1(blk_n.begin(), blk_n.end())});
  TiledRange trange_c({TiledRange1(blk_m.begin(), blk_m.end()),
                       TiledRange1(blk_n.begin(), blk_n.end())});

  if (world.rank() == 0)
    std::cout << "TiledArray UMTensor dense matrix multiply\n"
              << "  Nodes        = " << world.size() << "\n"
              << "  A            = " << Nm << " x " << Nk << " ("
              << double(Nm * Nk * sizeof(T)) / 1.0e9 << " GB)\n"
              << "  B            = " << Nk << " x " << Nn << " ("
              << double(Nk * Nn * sizeof(T)) / 1.0e9 << " GB)\n"
              << "  C            = " << Nm << " x " << Nn << " ("
              << double(Nm * Nn * sizeof(T)) / 1.0e9 << " GB)\n"
              << "  Tile A,B,C   = " << Bm << "x" << Bk << ", " << Bk << "x"
              << Bn << ", " << Bm << "x" << Bn << "\n"
              << "  Iterations   = " << nrepeat << "\n";

  ArrayT a(world, trange_a);
  ArrayT b(world, trange_b);
  ArrayT c(world, trange_c);

  const T val_a = T(0.03);
  const T val_b = T(0.02);
  a.fill(val_a);
  b.fill(val_b);
  world.gop.fence();

  // Prefetch inputs to the device once before the timed loop -- the per-tile
  // ops will also prefetch lazily, but doing it up front keeps the timing
  // focused on the GEMM kernel cost.
  TA::to_device(a);
  TA::to_device(b);

#ifdef TILEDARRAY_HAS_CUDA
  cudaProfilerStart();
#endif

  double total_time = 0.0;
  double total_gflops = 0.0;
  for (long i = 0; i < nrepeat; ++i) {
    const double t0 = madness::wall_time();
    c("m,n") = a("m,k") * b("k,n");
    world.gop.fence();
    const double t1 = madness::wall_time();
    const double dt = t1 - t0;
    const double gflops = static_cast<double>(nflops) / (dt * 1.0e9);
    total_time += dt;
    total_gflops += gflops;
    if (world.rank() == 0)
      std::cout << "  iter " << (i + 1) << "  time=" << dt
                << " s  gflops=" << gflops << "\n";
  }

#ifdef TILEDARRAY_HAS_CUDA
  cudaProfilerStop();
#endif

  if (world.rank() == 0)
    std::cout << "  Average time   = " << (total_time / double(nrepeat))
              << " s\n  Average gflops = " << (total_gflops / double(nrepeat))
              << "\n";

  // Verify: every result element should be Nk * val_a * val_b.
  const T expected = T(Nk) * val_a * val_b;
  const auto eps = std::numeric_limits<TA::detail::scalar_t<T>>::epsilon();
  const auto tolerance = std::abs(expected) * static_cast<decltype(eps)>(Nk) *
                         static_cast<decltype(eps)>(8) * eps;
  TA::to_host(c);
  bool ok = true;
  for (auto it = c.begin(); it != c.end(); ++it) {
    const auto tile = it->get();
    for (std::size_t k = 0; k < tile.size(); ++k) {
      if (std::abs(tile.data()[k] - expected) > tolerance) {
        ok = false;
        if (world.rank() == 0)
          std::cout << "  MISMATCH at tile " << it.index() << " element " << k
                    << ": got " << tile.data()[k] << " expected " << expected
                    << "\n";
        break;
      }
    }
    if (!ok) break;
  }
  if (world.rank() == 0)
    std::cout << (ok ? "  Verification PASSED\n" : "  Verification FAILED\n");
}

}  // namespace

int try_main(int argc, char **argv) {
  TiledArray::World &world = TA_SCOPED_INITIALIZE(argc, argv);

  if (argc < 7) {
    if (world.rank() == 0)
      std::cerr
          << "Usage: " << argv[0] << " Nm Bm Nn Bn Nk Bk [nrepeat=5]\n"
          << "  Computes c(Nm,Nn) = a(Nm,Nk) * b(Nk,Nn) with UMTensor tiles\n";
    return 1;
  }
  const long Nm = std::atol(argv[1]);
  const long Bm = std::atol(argv[2]);
  const long Nn = std::atol(argv[3]);
  const long Bn = std::atol(argv[4]);
  const long Nk = std::atol(argv[5]);
  const long Bk = std::atol(argv[6]);
  const long nrepeat = (argc >= 8 ? std::atol(argv[7]) : 5);
  if (Nm <= 0 || Nn <= 0 || Nk <= 0 || Bm <= 0 || Bn <= 0 || Bk <= 0 ||
      nrepeat <= 0) {
    if (world.rank() == 0)
      std::cerr << "All sizes / blocks / nrepeat must be positive\n";
    return 1;
  }

  run<double>(world, Nm, Bm, Nn, Bn, Nk, Bk, nrepeat);
  return 0;
}

int main(int argc, char **argv) {
  try {
    return try_main(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "unknown exception\n";
    return 1;
  }
}
