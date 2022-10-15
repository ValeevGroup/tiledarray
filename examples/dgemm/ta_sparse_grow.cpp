/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  sparce.cpp
 *  Feb 5, 2015
 *
 */

#include <TiledArray/version.h>
#include <tiledarray.h>
#include <iomanip>
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
    const long repeat = (argc >= 4 ? atol(argv[3]) : 4);
    if (repeat <= 0) {
      std::cerr << "Error: number of repetitions must be greater than zero.\n";
      return 1;
    }

    // Print information about the test
    const std::size_t num_blocks = matrix_size / block_size;
    const double app_flop = 2.0 * matrix_size * matrix_size * matrix_size;
    const float tile_norm = std::sqrt(float(block_size * block_size));
    std::vector<double> speeds, times, app_speeds, real_sparsity;

    if (world.rank() == 0)
      std::cout << "TiledArray: growing, block-sparse matrix multiply test..."
                << "\nGit HASH: " << TiledArray::revision()
                << "\nNumber of nodes     = " << world.size()
                << "\nBlock size          = " << block_size << "x" << block_size
                << "\nMemory per matrix   = "
                << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
                << " GB"
                << "\nAverage blocks/node = "
                << double(num_blocks * num_blocks) / double(world.size())
                << "\n";

    for (unsigned int sparsity = 100u; sparsity > 0u; sparsity -= 10u) {
      TiledArray::TSpArrayD::wait_for_lazy_cleanup(world);

      // Compute the number of blocks and matrix size for the sparse matrix
      const double sparse_fraction = double(sparsity) / 100.0;
      const long sparse_num_blocks =
          std::sqrt(double(num_blocks * num_blocks) / sparse_fraction);
      const long sparse_matrix_size = sparse_num_blocks * block_size;
      const long sparse_block_count =
          sparse_fraction * double(sparse_num_blocks * sparse_num_blocks);

      if (world.rank() == 0)
        std::cout << "\nSparsity = " << sparsity << "%"
                  << "\nMatrix size = " << sparse_matrix_size << "x"
                  << sparse_matrix_size << "\n";

      // Construct TiledRange
      std::vector<unsigned int> blocking;
      blocking.reserve(sparse_num_blocks + 1);
      for (long i = 0l; i <= sparse_matrix_size; i += block_size)
        blocking.push_back(i);

      const std::vector<TiledArray::TiledRange1> blocking2(
          2, TiledArray::TiledRange1(blocking.begin(), blocking.end()));

      const TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());

      // Construct tile norm tensors
      TiledArray::Tensor<float> a_tile_norms(trange.tiles_range(), 0.0f),
          b_tile_norms(trange.tiles_range(), 0.0f);

      // Fill tile norm tensors
      if (world.rank() == 0) {
        if (sparsity == 100u) {
          std::fill(a_tile_norms.begin(), a_tile_norms.end(), tile_norm);
          std::fill(b_tile_norms.begin(), b_tile_norms.end(), tile_norm);
        } else {
          world.srand(time(NULL));
          for (long count = 0l; count < sparse_block_count; ++count) {
            // Find a new zero tile index.
            std::size_t index = 0ul;
            do {
              index = world.rand() % trange.tiles_range().volume();
            } while (a_tile_norms[index] >
                     TiledArray::SparseShape<float>::threshold());

            // Set index tile of matrix matrix a.
            a_tile_norms[index] = tile_norm;

            // Find a new zero tile index.
            do {
              index = world.rand() % trange.tiles_range().volume();
            } while (b_tile_norms[index] >
                     TiledArray::SparseShape<float>::threshold());

            b_tile_norms[index] = tile_norm;
          }
        }
      }

      // Construct the argument shapes
      TiledArray::SparseShape<float> a_shape(world, a_tile_norms, trange),
          b_shape(world, b_tile_norms, trange);

      // Construct and initialize arrays
      TiledArray::TSpArrayD a(world, trange, a_shape);
      TiledArray::TSpArrayD b(world, trange, b_shape);
      TiledArray::TSpArrayD c;
      a.fill_local(1.0);
      b.fill_local(1.0);

      // Start clock
      if (world.rank() == 0) std::cout << "Starting iterations:\n";

      double total_time = 0.0, flop = 0.0;

      // Do matrix multiplication
      for (int i = 0; i < repeat; ++i) {
        const double start = madness::wall_time();
        c("m,n") = a("m,k") * b("k,n");
        const double time = madness::wall_time() - start;
        total_time += time;
        if (flop < 1.0) flop = 2.0 * c("m,n").sum();
        if (world.rank() == 0)
          std::cout << "Iteration " << i + 1 << "   time=" << time
                    << " s,   speed=" << flop / time / 1.0e9
                    << " GFLOPS,   apparent speed=" << app_flop / time / 1.0e9
                    << " GFLOPS\n";
      }

      // Compute results
      speeds.push_back(double(repeat) * flop / total_time / 1.0e9);
      times.push_back(total_time / repeat);
      app_speeds.push_back(double(repeat) * app_flop / total_time / 1.0e9);
      real_sparsity.push_back(100.0 * double(sparse_block_count) /
                              double(sparse_num_blocks * sparse_num_blocks));

      // Print results for this iteration
      if (world.rank() == 0) {
        std::cout << "\nSparsity               = " << real_sparsity.back()
                  << "%\n"
                  << "Average wall time      = " << times.back() << " s\n"
                  << "Average speed          = " << speeds.back() << " GFLOPS\n"
                  << "Average apparent speed = " << app_speeds.back()
                  << " GFLOPS\n";
      }
    }

    // Print out comma separated list of all results
    if (world.rank() == 0) {
      std::cout << "\n\nResults:\n"
                   "sparsity (%), time (s), speed (GFLOPS), apparent speed "
                   "(GFLOPS)\n";
      for (unsigned int i = 0; i < 10; ++i) {
        std::cout << real_sparsity[i] << ", " << times[i] << ", " << speeds[i]
                  << ", " << app_speeds[i] << "\n";
      }
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
