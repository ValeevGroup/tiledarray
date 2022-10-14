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

using Tile_t = TiledArray::Tile<TiledArray::Tensor<double>>;
using Array_t = TiledArray::DistArray<Tile_t>;

void set_tiles(double val, Array_t& a) {
  auto const& trange = a.trange();

  auto pmap = a.pmap();
  const auto end = pmap->end();
  for (auto it = pmap->begin(); it != end; ++it) {
    auto range = trange.make_tile_range(*it);
    a.set(*it, Tile_t(TiledArray::Tensor<double>(range, val)));
  }
}

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
      std::cerr << "Error: matrix size must be evenly divisible by block "
                   "size.\n";
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
      std::cout << "TiledArray: dense matrix multiply test..."
                << "\nGit HASH: " << TILEDARRAY_REVISION
                << "\nNumber of nodes     = " << world.size()
                << "\nMatrix size         = " << matrix_size << "x"
                << matrix_size << "\nBlock size          = " << block_size
                << "x" << block_size << "\nMemory per matrix   = "
                << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
                << " GB\nNumber of blocks    = " << block_count
                << "\nAverage blocks/node = "
                << double(block_count) / double(world.size()) << "\n";

    const double flop =
        2.0 * double(matrix_size * matrix_size * matrix_size) / 1.0e9;

    // Construct TiledRange
    std::vector<unsigned int> blocking;
    blocking.reserve(num_blocks + 1);
    for (long i = 0l; i <= matrix_size; i += block_size) blocking.push_back(i);

    std::vector<TiledArray::TiledRange1> blocking2(
        2, TiledArray::TiledRange1(blocking.begin(), blocking.end()));

    TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());

    // Construct and initialize arrays
    Array_t a(world, trange);
    Array_t b(world, trange);
    Array_t c(world, trange);
    set_tiles(1.0, a);
    set_tiles(1.0, b);

    TiledArray::TArrayD a_check(world, trange);
    TiledArray::TArrayD b_check(world, trange);
    TiledArray::TArrayD c_check(world, trange);
    a_check.fill(1.0);
    b_check.fill(1.0);

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
      c_check("m,n") = a_check("m,k") * b_check("k,n");
      //      world.gop.fence();
      const double time = madness::wall_time() - start;
      total_time += time;
      if (world.rank() == 0)
        std::cout << "Iteration " << i + 1 << "   time=" << time
                  << "   GFLOPS=" << flop / time << "\n";
      auto check_it = c_check.begin();
      for (auto it = c.begin(); it != c.end() && check_it != c_check.end();
           ++it, ++check_it) {
        auto tile_diff = it->get().tensor().subt(check_it->get()).norm();
        if (tile_diff >= 1e-15) {
          std::cout << "Tile " << it.ordinal() << " failed test "
                    << " with norm diff " << tile_diff << std::endl;
          assert(false);
        }
      }
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
