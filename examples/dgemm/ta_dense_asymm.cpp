/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <TiledArray/external/btas.h>
#include <tiledarray.h>
#include <iostream>

int main(int argc, char** argv) {
  // Initialize runtime
  TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  // Get command line arguments
  if (argc < 6) {
    std::cout << "multiplies A(Nm,Nk) * B(Nk,Nn), with dimensions m, n, and k "
                 "blocked by Bm, Bn, and Bk, respectively"
              << std::endl
              << "Usage: " << argv[0]
              << " Nm Bm Nn Bn Nk Bk [repetitions=5] [real=double]\n";
    return 0;
  }
  const long Nm = atol(argv[1]);
  const long Bm = atol(argv[2]);
  const long Nn = atol(argv[3]);
  const long Bn = atol(argv[4]);
  const long Nk = atol(argv[5]);
  const long Bk = atol(argv[6]);
  if (Nm <= 0 || Nn <= 0 || Nk <= 0) {
    std::cerr << "Error: dimensions must be greater than zero.\n";
    return 1;
  }
  if (Bm <= 0 || Bn <= 0 || Bk <= 0) {
    std::cerr << "Error: block sizes must be greater than zero.\n";
    return 1;
  }
  if ((Nm % Bm) != 0ul || Nn % Bn != 0ul || Nk % Bk != 0ul) {
    std::cerr
        << "Error: dimension size must be evenly divisible by block size.\n";
    return 1;
  }
  const long repeat = (argc >= 8 ? atol(argv[7]) : 5);
  if (repeat <= 0) {
    std::cerr << "Error: number of repetitions must be greater than zero.\n";
    return 1;
  }

  const std::string real_type_str = (argc >= 9 ? argv[8] : "double");
  if (real_type_str != "double" && real_type_str != "float") {
    std::cerr << "Error: invalid real type " << real_type_str << ".\n";
    return 1;
  }

  const std::size_t Tm = Nm / Bm;
  const std::size_t Tn = Nn / Bn;
  const std::size_t Tk = Nk / Bk;

  if (world.rank() == 0)
    std::cout << "TiledArray: dense matrix multiply test...\n"
              << "Number of nodes     = " << world.size()
              << "\nSize of A         = " << Nm << "x" << Nk << " ("
              << double(Nm * Nk * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of A block   = " << Bm << "x" << Bk
              << "\nSize of B         = " << Nk << "x" << Nn << " ("
              << double(Nk * Nn * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of B block   = " << Bk << "x" << Bn
              << "\nSize of C         = " << Nm << "x" << Nn << " ("
              << double(Nm * Nn * sizeof(double)) / 1.0e9 << " GB)"
              << "\nSize of C block   = " << Bm << "x" << Bn
              << "\n# of blocks of C  = " << Tm * Tn
              << "\nAverage # of blocks of C/node = "
              << double(Tm * Tn) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> blocking_m;
  blocking_m.reserve(Tm + 1);
  for (long i = 0l; i <= Nm; i += Bm) blocking_m.push_back(i);

  std::vector<unsigned int> blocking_n;
  blocking_n.reserve(Tn + 1);
  for (long i = 0l; i <= Nn; i += Bn) blocking_n.push_back(i);

  std::vector<unsigned int> blocking_k;
  blocking_k.reserve(Tk + 1);
  for (long i = 0l; i <= Nk; i += Bk) blocking_k.push_back(i);

  // Structure of c
  std::vector<TiledArray::TiledRange1> blocking_C;
  blocking_C.reserve(2);
  blocking_C.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking_C.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  // Structure of a
  std::vector<TiledArray::TiledRange1> blocking_A;
  blocking_A.reserve(2);
  blocking_A.push_back(
      TiledArray::TiledRange1(blocking_m.begin(), blocking_m.end()));
  blocking_A.push_back(
      TiledArray::TiledRange1(blocking_k.begin(), blocking_k.end()));

  // Structure of b
  std::vector<TiledArray::TiledRange1> blocking_B;
  blocking_B.reserve(2);
  blocking_B.push_back(
      TiledArray::TiledRange1(blocking_k.begin(), blocking_k.end()));
  blocking_B.push_back(
      TiledArray::TiledRange1(blocking_n.begin(), blocking_n.end()));

  TiledArray::TiledRange  // TRange for c
      trange_c(blocking_C.begin(), blocking_C.end());

  TiledArray::TiledRange  // TRange for a
      trange_a(blocking_A.begin(), blocking_A.end());

  TiledArray::TiledRange  // TRange for b
      trange_b(blocking_B.begin(), blocking_B.end());

  auto run = [&](auto* tarray_ptr) {
    using Array = std::decay_t<std::remove_pointer_t<decltype(tarray_ptr)>>;

    // Construct and initialize arrays
    Array a(world, trange_a);
    Array b(world, trange_b);
    Array c(world, trange_c);
    a.fill(1.0);
    b.fill(1.0);

    // Start clock
    world.gop.fence();
    const double wall_time_start = madness::wall_time();

    // Do matrix multiplication
    for (int i = 0; i < repeat; ++i) {
      c("m,n") = a("m,k") * b("k,n");
      world.gop.fence();
      if (world.rank() == 0) std::cout << "Iteration " << i + 1 << "\n";
    }

    // Stop clock
    const double wall_time_stop = madness::wall_time();

    if (world.rank() == 0)
      std::cout << "Average wall time   = "
                << (wall_time_stop - wall_time_start) / double(repeat)
                << " sec\nAverage GFLOPS      = "
                << double(repeat) * 2.0 * double(Nn * Nm * Nk) /
                       (wall_time_stop - wall_time_start) / 1.0e9
                << "\n";
  };

  // by default use TiledArray tensors
  constexpr bool use_btas = false;
  // btas::Tensor instead
  if (real_type_str == "double") {
    if constexpr (!use_btas)
      run(static_cast<TiledArray::TArrayD*>(nullptr));
    else
      run(static_cast<TiledArray::DistArray<
              TiledArray::Tile<btas::Tensor<double, TiledArray::Range>>>*>(
          nullptr));
  } else {
    if constexpr (!use_btas)
      run(static_cast<TiledArray::TArrayF*>(nullptr));
    else
      run(static_cast<TiledArray::DistArray<
              TiledArray::Tile<btas::Tensor<float, TiledArray::Range>>>*>(
          nullptr));
  }

  return 0;
}
