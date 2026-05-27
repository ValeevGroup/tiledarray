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

// Element-wise vector-op benchmarks (add, scale, permute, Hadamard) using
// the native UMTensor tile type. Companion to ta_vector_device.cpp.
//
// Usage:
//   ta_vector_um_tensor Nm Bm Nn Bn [nrepeat=5]
//
// Times each op for nrepeat iterations and reports the average wall time
// and effective bandwidth (counting one read + one write per element for
// in-place ops, two reads + one write for binary ops).

#include <TiledArray/device/tensor.h>
#include <tiledarray.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

template <typename T>
void run(TiledArray::World &world, long Nm, long Bm, long Nn, long Bn,
         long nrepeat) {
  using TA::DistArray;
  using TA::TiledRange;
  using TA::TiledRange1;
  using TA::UMTensor;
  using TileT = UMTensor<T>;
  using ArrayT = DistArray<TileT, TA::DensePolicy>;

  auto blocking = [](long N, long B) {
    std::vector<unsigned int> v;
    for (long i = 0; i <= N; i += B) v.push_back(static_cast<unsigned int>(i));
    return v;
  };
  auto blk_m = blocking(Nm, Bm);
  auto blk_n = blocking(Nn, Bn);

  TiledRange trange({TiledRange1(blk_m.begin(), blk_m.end()),
                     TiledRange1(blk_n.begin(), blk_n.end())});
  TiledRange trange_T({TiledRange1(blk_n.begin(), blk_n.end()),
                       TiledRange1(blk_m.begin(), blk_m.end())});

  if (world.rank() == 0)
    std::cout << "TiledArray UMTensor vector-op benchmark\n"
              << "  Nodes        = " << world.size() << "\n"
              << "  Matrix       = " << Nm << " x " << Nn << " ("
              << double(Nm * Nn * sizeof(T)) / 1.0e9 << " GB)\n"
              << "  Tile         = " << Bm << " x " << Bn << "\n"
              << "  Iterations   = " << nrepeat << "\n";

  ArrayT a(world, trange);
  ArrayT b(world, trange);
  ArrayT c(world, trange);
  ArrayT t(world, trange_T);  // transposed-shape result for permute test

  a.fill(T(0.03));
  b.fill(T(0.02));
  c.fill(T(0.0));
  t.fill(T(0.0));
  world.gop.fence();
  TA::to_device(a);
  TA::to_device(b);

  const double bytes_per_elem = static_cast<double>(sizeof(T));
  const double n_elems = static_cast<double>(Nm) * static_cast<double>(Nn);

  auto bench = [&](const char *name, double bytes_per_iter, auto &&op) {
    double total_time = 0.0;
    for (long i = 0; i < nrepeat; ++i) {
      const double t0 = madness::wall_time();
      op();
      world.gop.fence();
      const double t1 = madness::wall_time();
      total_time += t1 - t0;
    }
    const double avg = total_time / static_cast<double>(nrepeat);
    const double bw_gbs = bytes_per_iter / (avg * 1.0e9);
    if (world.rank() == 0)
      std::cout << "  " << name << ":  avg=" << avg << " s  bw=" << bw_gbs
                << " GB/s\n";
  };

  // Binary read-read-write: 3 element accesses per element.
  const double rw3_bytes = 3.0 * n_elems * bytes_per_elem;
  // Unary read-write: 2 element accesses per element.
  const double rw2_bytes = 2.0 * n_elems * bytes_per_elem;

  bench("add(c=a+b)", rw3_bytes, [&] { c("m,n") = a("m,n") + b("m,n"); });
  bench("subt(c=a-b)", rw3_bytes, [&] { c("m,n") = a("m,n") - b("m,n"); });
  bench("scale(c=2*a)", rw2_bytes, [&] { c("m,n") = 2.0 * a("m,n"); });
  bench("hadamard(c=a*b)", rw3_bytes, [&] { c("m,n") = a("m,n") * b("m,n"); });
  bench("permute(t=a^T)", rw2_bytes, [&] { t("n,m") = a("m,n"); });
  bench("axpy(c+=a)", rw3_bytes, [&] { c("m,n") += a("m,n"); });

  world.gop.fence();
}

}  // namespace

int try_main(int argc, char **argv) {
  TiledArray::World &world = TA_SCOPED_INITIALIZE(argc, argv);

  if (argc < 5) {
    if (world.rank() == 0)
      std::cerr
          << "Usage: " << argv[0] << " Nm Bm Nn Bn [nrepeat=5]\n"
          << "  Times element-wise vector ops on Nm x Nn UMTensor matrices\n";
    return 1;
  }
  const long Nm = std::atol(argv[1]);
  const long Bm = std::atol(argv[2]);
  const long Nn = std::atol(argv[3]);
  const long Bn = std::atol(argv[4]);
  const long nrepeat = (argc >= 6 ? std::atol(argv[5]) : 5);
  if (Nm <= 0 || Nn <= 0 || Bm <= 0 || Bn <= 0 || nrepeat <= 0) {
    if (world.rank() == 0)
      std::cerr << "All sizes / blocks / nrepeat must be positive\n";
    return 1;
  }

  run<double>(world, Nm, Bm, Nn, Bn, nrepeat);
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
