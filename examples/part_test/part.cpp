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

#define TILEDARRAY_ENABLE_TEST_PROC_GRID
#include <iostream>
#include <tiledarray.h>

using namespace TiledArray;

int main(int argc, char** argv) {
  int rc = 0;

  try {

    // Initialize runtime
    TiledArray::World& world = TiledArray::initialize(argc, argv);

    // Get command line arguments
    if(argc < 4) {
      std::cout << "Usage: ta_dense matrix_size block_size procs\n";
      return 0;
    }
    const long matrix_size = atol(argv[1]);
    const long block_size = atol(argv[2]);
    const long num_procs = atol(argv[3]);
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

    const std::size_t num_blocks = matrix_size / block_size;
    const std::size_t block_count = num_blocks * num_blocks;
    const float tile_norm = std::sqrt(float(block_size * block_size));

    std::cout << "TiledArray: partition test..."
              << "\nGit HASH: " << TILEDARRAY_REVISION
              << "\nNumper of processes = " << num_procs
              << "\nMatrix size         = " << matrix_size << "x" << matrix_size
              << "\nBlock size          = " << block_size << "x" << block_size << "\n";


    math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, 2u, 2u);

    TiledArray::detail::ProcGrid
    proc_grid(world, 0, num_procs, num_blocks, num_blocks, matrix_size, matrix_size);

    for(unsigned int sparsity = 50u; sparsity > 0u; sparsity -= 2u) {

      // Compute the number of blocks and matrix size for the sparse matrix
      const double sparse_fraction = double(sparsity) / 100.0;
      const long sparse_num_blocks =
          std::sqrt(double(num_blocks * num_blocks) / sparse_fraction);
      const long sparse_matrix_size = sparse_num_blocks * block_size;
      const long sparse_block_count = sparse_fraction * double(sparse_num_blocks * sparse_num_blocks);

      std::cout << "\nSparsity = " << sparsity << "%"
          << "\nMatrix size = " << sparse_matrix_size << "x" << sparse_matrix_size << "\n";

      // Construct TiledRange
      std::vector<unsigned int> blocking;
      blocking.reserve(sparse_num_blocks + 1);
      for(long i = 0l; i <= sparse_matrix_size; i += block_size)
        blocking.push_back(i);

      const std::vector<TiledArray::TiledRange1> blocking2(2,
          TiledArray::TiledRange1(blocking.begin(), blocking.end()));

      const TiledArray::TiledRange
        trange(blocking2.begin(), blocking2.end());

      // Construct tile norm tensors
      TiledArray::Tensor<float>
          a_tile_norms(trange.tiles(), 0.0f),
          b_tile_norms(trange.tiles(), 0.0f);

      // Fill tile norm tensors
      if(sparsity == 100u) {
        std::fill(a_tile_norms.begin(), a_tile_norms.end(), tile_norm);
        std::fill(b_tile_norms.begin(), b_tile_norms.end(), tile_norm);
      } else {
        world.srand(time(NULL));
        for(long count = 0l; count < sparse_block_count; ++count) {
          // Find a new zero tile index.
          std::size_t index = 0ul;
          do {
            index = world.rand() % trange.tiles().volume();
          } while(a_tile_norms[index] > TiledArray::SparseShape<float>::threshold());

          // Set index tile of matrix matrix a.
          a_tile_norms[index] = tile_norm;

          // Find a new zero tile index.
          do {
            index = world.rand() % trange.tiles().volume();
          } while(b_tile_norms[index] > TiledArray::SparseShape<float>::threshold());

          b_tile_norms[index] = tile_norm;
        }
      }

      // Construct the argument shapes
      TiledArray::SparseShape<float>
          a_shape(world, a_tile_norms, trange),
          b_shape(world, b_tile_norms, trange);

      std::cout << "\n\n******** Row partition ********\n\n";

      TiledArray::detail::HyperGraph row_hgraph = a_shape.make_row_hypergraph(b_shape, gemm_helper);

      const long init_row_cutset = row_hgraph.init_cut_set(proc_grid.proc_rows());

      const double start_row_part = madness::wall_time();
      row_hgraph.partition(proc_grid.proc_rows(), 0l, 0.1);
      const double finish_row_part = madness::wall_time();

      const long final_row_cutset = row_hgraph.cut_set();


      row_hgraph.verify_lambdas();
      std::cout << "\n\nInitial cutset = " << init_row_cutset << "\n"
          << "Final cutset   = " << final_row_cutset << "\n"
          << "Comm decrease  = " << 100.0 * (1.0 - double(final_row_cutset) / double(init_row_cutset)) << "%\n"
          << "Partition time = " << finish_row_part - start_row_part << " s\n";


      std::cout << "\n\n******** Column partition ********\n\n";

      TiledArray::detail::HyperGraph col_hgraph = b_shape.make_col_hypergraph(a_shape, gemm_helper);

      const long init_col_cutset = col_hgraph.init_cut_set(proc_grid.proc_cols());

      const double start_col_part = madness::wall_time();
      col_hgraph.partition(proc_grid.proc_cols(), 0l, 0.1);
      const double finish_col_part = madness::wall_time();

      const long final_col_cutset = col_hgraph.cut_set();


      col_hgraph.verify_lambdas();
      std::cout << "\n\nInitial cutset = " << init_col_cutset << "\n"
          << "Final cutset   = " << final_col_cutset << "\n"
          << "Comm decrease  = " << 100.0 * (1.0 - double(final_col_cutset) / double(init_col_cutset)) << "%\n"
          << "Partition time = " << finish_col_part - start_col_part << " s\n";
    }

    TiledArray::finalize();
  } catch(TiledArray::Exception& e) {
    std::cerr << "!! TiledArray exception: " << e.what() << "\n";
    rc = 1;
  } catch(madness::MadnessException& e) {
    std::cerr << "!! MADNESS exception: " << e.what() << "\n";
    rc = 1;
  } catch(SafeMPI::Exception& e) {
    std::cerr << "!! SafeMPI exception: " << e.what() << "\n";
    rc = 1;
  } catch(std::exception& e) {
    std::cerr << "!! std exception: " << e.what() << "\n";
    rc = 1;
  } catch(...) {
    std::cerr << "!! exception: unknown exception\n";
    rc = 1;
  }

  return rc;
}
