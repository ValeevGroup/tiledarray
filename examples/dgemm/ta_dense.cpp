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

#include <TiledArray/version.h>
#include <madness/world/worldmem.h>
#include <tiledarray.h>
#include <iostream>

bool to_bool(const char* str) {
  if (not strcmp(str, "0") || not strcmp(str, "no") || not strcmp(str, "false"))
    return false;
  if (not strcmp(str, "1") || not strcmp(str, "yes") || not strcmp(str, "true"))
    return true;
  throw std::runtime_error("unrecognized string specification of bool");
}

// Leave as underscore for now since without it is broken on gcc 11/03/2015 Drew
template <typename T>
void gemm_(TiledArray::World& world, const TiledArray::TiledRange& trange,
           long repeat);

int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

    // Get command line arguments
    if (argc < 3) {
      std::cout << "Usage: " << argv[0]
                << " matrix_size block_size [repetitions] [use_complex]\n";
      return 0;
    }
    const long matrix_size = atol(argv[1]);
    const long block_size = atol(argv[2]);
    if (matrix_size <= 0) {
      std::cerr << "Error: matrix size must be greater than zero.\n";
      return 1;
    }
    if (block_size <= 0) {
      std::cerr << "Error: block size must be greater than zero.\n";
      return 1;
    }
    if ((matrix_size % block_size) != 0ul) {
      std::cerr
          << "Error: matrix size must be evenly divisible by block size.\n";
      return 1;
    }
    const long repeat = (argc >= 4 ? atol(argv[3]) : 5);
    if (repeat <= 0) {
      std::cerr << "Error: number of repetitions must be greater than zero.\n";
      return 1;
    }
    const bool use_complex = (argc >= 5 ? to_bool(argv[4]) : false);

    const std::size_t num_blocks = matrix_size / block_size;
    const std::size_t block_count = num_blocks * num_blocks;

    if (world.rank() == 0)
      std::cout << "TiledArray: dense matrix multiply test..."
                << "\nGit description: " << TiledArray::git_description()
                << "\nNumber of nodes     = " << world.size()
                << "\nMatrix size         = " << matrix_size << "x"
                << matrix_size << "\nBlock size          = " << block_size
                << "x" << block_size << "\nMemory per matrix   = "
                << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
                << " GB\nNumber of blocks    = " << block_count
                << "\nAverage blocks/node = "
                << double(block_count) / double(world.size())
                << "\nComplex             = "
                << (use_complex ? "true" : "false") << "\n";

    // Construct TiledRange
    std::vector<unsigned int> blocking;
    blocking.reserve(num_blocks + 1);
    for (long i = 0l; i <= matrix_size; i += block_size) blocking.push_back(i);

    std::vector<TiledArray::TiledRange1> blocking2(
        2, TiledArray::TiledRange1(blocking.begin(), blocking.end()));

    TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());

    if (use_complex)
      gemm_<std::complex<double>>(world, trange, repeat);
    else
      gemm_<double>(world, trange, repeat);

  } catch (TiledArray::Exception& e) {
    std::cerr << "!! TiledArray exception: " << e.what() << "\n";
    rc = 1;
  } catch (madness::MadnessException& e) {
    std::cerr << "!! MADNESS exception: " << e.what() << "\n";
    rc = 1;
  } catch (SafeMPI::Exception& e) {
    std::cerr << "!! SafeMPI exception: " << e.what() << "\n";
    rc = 1;
  } catch (std::exception& e) {
    std::cerr << "!! std exception: " << e.what() << "\n";
    rc = 1;
  } catch (...) {
    std::cerr << "!! exception: unknown exception\n";
    rc = 1;
  }

  return rc;
}

template <typename T>
void gemm_(TiledArray::World& world, const TiledArray::TiledRange& trange,
           long repeat) {
  const bool do_memtrace = false;

  const auto n = trange.elements_range().extent()[0];
  const auto complex_T = TiledArray::detail::is_complex<T>::value;
  const double gflop =
      (complex_T ? 8 : 2)  // 1 multiply takes 6/1 flops for complex/real
                           // 1 add takes 2/1 flops for complex/real
      * double(n * n * n) / 1.0e9;

  auto memtrace = [do_memtrace, &world](const std::string& str) -> void {
    if (do_memtrace) {
      world.gop.fence();
      madness::print_meminfo(world.rank(), str);
    }
#ifdef TA_TENSOR_MEM_PROFILE
    {
      world.gop.fence();
      std::cout << str << ": TA::Tensor allocated "
                << TA::hostEnv::instance()->host_allocator().getHighWatermark()
                << " bytes" << std::endl;
    }
#endif
  };

  memtrace("start");
  {  // array lifetime scope
    // Construct and initialize arrays
    TiledArray::TArray<T> a(world, trange);
    TiledArray::TArray<T> b(world, trange);
    TiledArray::TArray<T> c(world, trange);
    a.fill(1.0);
    b.fill(1.0);
    memtrace("allocated a and b");

    // Start clock
    world.gop.fence();
    if (world.rank() == 0)
      std::cout << "Starting iterations: "
                << "\n";

    double total_time = 0.0;
    double total_gflop_rate = 0.0;

    // Do matrix multiplication
    for (int i = 0; i < repeat; ++i) {
      const double start = madness::wall_time();
      c("m,n") = a("m,k") * b("k,n");
      memtrace("c=a*b");
      const double time = madness::wall_time() - start;
      total_time += time;
      const double gflop_rate = gflop / time;
      total_gflop_rate += gflop_rate;
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << "   time=" << time
                  << "   GFLOPS=" << gflop_rate << "\n";
    }

    // Print results
    if (world.rank() == 0)
      std::cout << "Average wall time   = " << total_time / double(repeat)
                << " sec\nAverage GFLOPS      = "
                << total_gflop_rate / double(repeat) << "\n";

  }  // array lifetime scope
  memtrace("stop");
}
