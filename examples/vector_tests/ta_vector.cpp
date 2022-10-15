/*
 * This file is a part of TiledArray.
 * Copyright (C) 2017  Virginia Tech
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

template <typename T>
void vector_test(TiledArray::World& world, const TiledArray::TiledRange& trange,
                 long repeat);

int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

    // Get command line arguments
    if (argc < 3) {
      std::cout << "Usage: ta_vector matrix_size block_size [repetitions]\n";
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

    const std::size_t num_blocks = matrix_size / block_size;
    const std::size_t block_count = num_blocks * num_blocks;

    if (world.rank() == 0)
      std::cout << "TiledArray: vector ops test..."
                << "\nGit HASH: " << TiledArray::revision()
                << "\nNumber of nodes     = " << world.size()
                << "\nMatrix size         = " << matrix_size << "x"
                << matrix_size << "\nBlock size          = " << block_size
                << "x" << block_size << "\nMemory per matrix   = "
                << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
                << " GB\nNumber of blocks    = " << block_count
                << "\nAverage blocks/node = "
                << double(block_count) / double(world.size()) << "\n";

    // Construct TiledRange
    std::vector<unsigned int> blocking;
    blocking.reserve(num_blocks + 1);
    for (long i = 0l; i <= matrix_size; i += block_size) blocking.push_back(i);

    std::vector<TiledArray::TiledRange1> blocking2(
        2, TiledArray::TiledRange1(blocking.begin(), blocking.end()));

    TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());

    vector_test<double>(world, trange, repeat);

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
void vector_test(TiledArray::World& world, const TiledArray::TiledRange& trange,
                 long repeat) {
  const bool do_memtrace = false;

  auto memtrace = [do_memtrace, &world](const std::string& str) -> void {
    if (do_memtrace) {
      world.gop.fence();
      madness::print_meminfo(world.rank(), str);
    }
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

    double start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      c("i,j") = a("i,j") + b("i,j");
    }
    double stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Sum: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      c("i,j") = 3.0 * c("i,j");
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Scale: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      c("i,j") = 3.0 * (a("i,j") + b("i,j"));
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Scale Sum: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      c("i,j") = a("i,j") * b("i,j");
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Multiply: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      c("i,j") = 3.0 * (a("i,j") * b("i,j"));
    }
    stop = madness::wall_time();
    if (world.rank() == 0)
      std::cout << "Scale Multiply: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      c("i,j") = a("i,j") * a("i,j") + b("i,j") * b("i,j");
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Power Sum: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      T x = a("i,j").abs_max();
      (void)x;  // to prevent unused var warning
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Max abs: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      T x = a("i,j").sum();
      (void)x;  // to prevent unused var warning
    }
    stop = madness::wall_time();
    if (world.rank() == 0)
      std::cout << "Reduce Sum: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      T x = a("i,j").norm();
      (void)x;  // to prevent unused var warning
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Norm: " << stop - start << " s\n";

    start = madness::wall_time();
    for (int i = 0; i < repeat; ++i) {
      T x = a("i,j").dot(b("i,j"));
      (void)x;  // to prevent unused var warning
    }
    stop = madness::wall_time();
    if (world.rank() == 0) std::cout << "Dot: " << stop - start << " s\n";

  }  // array lifetime scope
  memtrace("stop");
}
