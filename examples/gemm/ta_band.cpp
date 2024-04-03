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

#include <TiledArray/util/time.h>
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
                << " matrix_size block_size band_width [repetitions]\n";
      return 0;
    }
    const long matrix_size = atol(argv[1]);
    const long block_size = atol(argv[2]);
    const long band_width = atol(argv[3]);
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
    if (band_width < 0) {
      std::cerr << "Error: Band width must be greater than zero.\n";
      return 1;
    }
    if (band_width > (matrix_size / 2)) {
      std::cerr
          << "Error: Band width must be less than half the matrix size.\n";
      return 1;
    }
    const long repeat = (argc >= 5 ? atol(argv[4]) : 5);
    if (repeat <= 0) {
      std::cerr << "Error: number of repetitions must be greater than zero.\n";
      return 1;
    }

    const long num_blocks = matrix_size / block_size;
    std::size_t block_count = 0;
    for (int i = -band_width + 1; i < band_width; ++i) {
      block_count += num_blocks - (2 * std::abs(i));
    }

    if (world.rank() == 0)
      std::cout << "TiledArray: block-banded matrix multiply test...\n"
                << "Number of nodes    = " << world.size()
                << "\nMatrix size        = " << matrix_size << "x"
                << matrix_size << "\nBlock size         = " << block_size << "x"
                << block_size << "\nMemory per matrix  = "
                << double(block_count * block_size * block_size *
                          sizeof(double)) /
                       1.0e9
                << " GB\nNumber of blocks   = " << block_count
                << "\nAverage blocks/node = " << block_count / world.size()
                << "\n";

    // Construct TiledRange
    std::vector<unsigned int> blocking;
    blocking.reserve(num_blocks + 1);
    for (long i = 0l; i <= matrix_size; i += block_size) blocking.push_back(i);

    std::vector<TiledArray::TiledRange1> blocking2(
        2, TiledArray::TiledRange1(blocking.begin(), blocking.end()));

    TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());

    TiledArray::SparseShape<float>::threshold(0.5);

    // Construct shape
    TiledArray::Tensor<float> shape_tensor(trange.tiles_range(), 0.0f);
    for (long i = 0l; i < num_blocks; ++i) {
      long j = std::max<long>(i - band_width + 1, 0);
      const long j_end = std::min<long>(i + band_width - 1, num_blocks);
      long ij = i * num_blocks + j;
      for (; j < j_end; ++j, ++ij) shape_tensor[ij] = 1.0;
    }

    TiledArray::SparseShape<float> shape(
        shape_tensor, trange, /* per_element_norms_already = */ true);

    // Construct and initialize arrays
    TiledArray::TSpArrayD a(world, trange, shape);
    TiledArray::TSpArrayD b(world, trange, shape);
    TiledArray::TSpArrayD c;
    a.fill(1.0);
    b.fill(1.0);

    // Do matrix multiplication
    world.gop.fence();
    for (int i = 0; i < repeat; ++i) {
      TA_RECORD_DURATION(c("m,n") = a("m,k") * b("k,n"); world.gop.fence();)
      if (world.rank() == 0) std::cout << "Iteration " << i + 1 << "\n";
    }

    // Print results
    const auto gflops_per_call = 2.0 * c("m,n").sum().get() / 1.e9;
    if (world.rank() == 0) {
      auto durations = TiledArray::duration_statistics();
      std::cout << "Average wall time   = " << durations.mean
                << " s\nAverage GFLOPS      = "
                << gflops_per_call * durations.mean_reciprocal
                << "\nMedian wall time   = " << durations.median
                << " s\nMedian GFLOPS      = "
                << gflops_per_call / durations.median << "\n";
    }

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
