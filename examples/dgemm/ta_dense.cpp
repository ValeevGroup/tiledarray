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

#include <iostream>
#include <tiledarray.h>

int main(int argc, char** argv) {
  int rc = 0;

  try {

    // Initialize runtime
    madness::World& world = madness::initialize(argc, argv);

    // Get command line arguments
    if(argc < 2) {
      std::cout << "Usage: ta_dense matrix_size block_size [repetitions]\n";
      return 0;
    }
    const long matrix_size = atol(argv[1]);
    const long block_size = atol(argv[2]);
    if (matrix_size <= 0) {
      std::cerr << "Error: matrix size must greater than zero.\n";
      return 1;
    }
    if (block_size <= 0) {
      std::cerr << "Error: block size must greater than zero.\n";
      return 1;
    }
    if((matrix_size % block_size) != 0ul) {
      std::cerr << "Error: matrix size must be evenly divisible by block size.\n";
      return 1;
    }
    const long repeat = (argc >= 4 ? atol(argv[3]) : 5);
    if (repeat <= 0) {
      std::cerr << "Error: number of repetitions must greater than zero.\n";
      return 1;
    }

    const std::size_t num_blocks = matrix_size / block_size;
    const std::size_t block_count = num_blocks * num_blocks;

    if(world.rank() == 0)
      std::cout << "TiledArray: dense matrix multiply test..."
                << "\nGit HASH: " << TILEDARRAY_REVISION
                << "\nNumber of nodes     = " << world.size()
                << "\nMatrix size         = " << matrix_size << "x" << matrix_size
                << "\nBlock size          = " << block_size << "x" << block_size
                << "\nMemory per matrix   = " << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
                << " GB\nNumber of blocks    = " << block_count
                << "\nAverage blocks/node = " << double(block_count) / double(world.size()) << "\n";

    const double flop = 2.0 * double(matrix_size * matrix_size * matrix_size) / 1.0e9;

    // Construct TiledRange
    std::vector<unsigned int> blocking;
    blocking.reserve(num_blocks + 1);
    for(long i = 0l; i <= matrix_size; i += block_size)
      blocking.push_back(i);

    std::vector<TiledArray::TiledRange1> blocking2(2,
        TiledArray::TiledRange1(blocking.begin(), blocking.end()));

    TiledArray::TiledRange
      trange(blocking2.begin(), blocking2.end());

    // Construct and initialize arrays
    TiledArray::Array<double, 2> a(world, trange);
    TiledArray::Array<double, 2> b(world, trange);
    TiledArray::Array<double, 2> c(world, trange);
    a.set_all_local(1.0);
    b.set_all_local(1.0);

    // Start clock
    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Starting iterations: " << "\n";

    double total_time = 0.0;

    // Do matrix multiplication
    for(int i = 0; i < repeat; ++i) {
      const double start = madness::wall_time();
      c("m,n") = a("m,k") * b("k,n");
//      world.gop.fence();
      const double time = madness::wall_time() - start;
      total_time += time;
      if(world.rank() == 0)
        std::cout << "Iteration " << i + 1 << "   time=" << time << "   GFLOPS="
            << flop / time <<"\n";
    }

    // Print results
    if(world.rank() == 0)
      std::cout << "Average wall time   = " << total_time / double(repeat)
          << " sec\nAverage GFLOPS      = " << double(repeat) * flop / total_time << "\n";

    madness::finalize();

  } catch(TiledArray::Exception& e) {
    std::cerr << "!!ERROR TiledArray: " << e.what() << "\n";
    rc = 1;
  } catch(madness::MadnessException& e) {
    std::cerr << "!!ERROR MADNESS: " << e.what() << "\n";
    rc = 1;
  } catch(SafeMPI::Exception& e) {
    std::cerr << "!!ERROR SafeMPI: " << e.what() << "\n";
    rc = 1;
  } catch(std::exception& e) {
    std::cerr << "!!ERROR std: " << e.what() << "\n";
    rc = 1;
  } catch(...) {
    std::cerr << "!!ERROR: unknown exception\n";
    rc = 1;
  }

  return rc;
}
