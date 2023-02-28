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
#include <tiledarray.h>
#include <iostream>

int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

    // Get command line arguments
    if (argc < 2) {
      std::cout << "Usage: " << argv[0]
                << " matrix_size block_size [repetitions]\n";
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

    const long num_blocks = matrix_size / block_size;
    const long block_count = num_blocks * num_blocks;

    const double flop =
        2.0 * double(matrix_size * matrix_size * matrix_size) / 1.0e9;

    // Construct TiledRange
    std::vector<unsigned long> blocking[3];
    world.srand(42);
    unsigned long min = std::numeric_limits<unsigned long>::max(), max = 0;
    for (long n = 0l; n < 3l; ++n) {
      blocking[n].resize(num_blocks + 1, 1);

      blocking[n][0] = 0;
      for (long i = num_blocks; i < matrix_size; ++i)
        ++(blocking[n][(world.rand() % num_blocks) + 1]);

      for (long i = 1l; i <= num_blocks; ++i) {
        min = std::min(blocking[n][i], min);
        max = std::max(blocking[n][i], max);
        blocking[n][i] += blocking[n][i - 1l];
      }
    }

    if (world.rank() == 0)
      std::cout << "TiledArray: dense-nonuniform matrix multiply test..."
                << "\nGit description: " << TiledArray::git_description()
                << "\nNumber of nodes     = " << world.size()
                << "\nMatrix size         = " << matrix_size << "x"
                << matrix_size << "\nAverage block size  = " << block_size
                << "x" << block_size << "\nMaximum block size  = " << max
                << "\nMinimum block size  = " << min
                << "\nMemory per matrix   = "
                << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
                << " GB\nNumber of blocks    = " << block_count
                << "\nAverage blocks/node = "
                << double(block_count) / double(world.size()) << "\n";

    std::vector<TiledArray::TiledRange1> blockingA, blockingB;
    blockingA.reserve(2);
    blockingA.push_back(
        TiledArray::TiledRange1(blocking[0].begin(), blocking[0].end()));
    blockingA.push_back(
        TiledArray::TiledRange1(blocking[1].begin(), blocking[1].end()));
    blockingB.reserve(2);
    blockingB.push_back(
        TiledArray::TiledRange1(blocking[1].begin(), blocking[1].end()));
    blockingB.push_back(
        TiledArray::TiledRange1(blocking[2].begin(), blocking[2].end()));

    TiledArray::TiledRange trangeA(blockingA.begin(), blockingA.end()),
        trangeB(blockingB.begin(), blockingB.end());

    // Construct and initialize arrays
    TiledArray::TArrayD a(world, trangeA);
    TiledArray::TArrayD b(world, trangeB);
    TiledArray::TArrayD c;
    a.fill(1.0);
    b.fill(1.0);

    // Start clock
    world.gop.fence();
    if (world.rank() == 0)
      std::cout << "Starting iterations: "
                << "\n";

    double total_time = 0.0;

    // Do matrix multiplication
    for (int i = 0; i < repeat; ++i) {
      const double start = madness::wall_time();
      c("m,n") = a("m,k") * b("k,n");
      //      world.gop.fence();
      const double time = madness::wall_time() - start;
      total_time += time;
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << "   time=" << time
                  << "   GFLOPS=" << flop / time << "\n";
    }

    // Print results
    if (world.rank() == 0)
      std::cout << "Average wall time   = " << total_time / double(repeat)
                << " sec\nAverage GFLOPS      = "
                << double(repeat) * flop / total_time << "\n";

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
