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
#include <time.h> // for time
#include <tiled_array.h>

int main(int argc, char** argv) {
  madness::initialize(argc,argv);
  madness::World world(SafeMPI::COMM_WORLD);

  // Get command line arguments
  if(argc < 2) {
    std::cout << "Usage: ta_sparse matrix_size block_size sparsity [repetitions]\n";
    return 0;
  }
  const long matrix_size = atol(argv[1]);
  const long block_size = atol(argv[2]);
  const long sparsity = atol(argv[3]);
  if(matrix_size <= 0) {
    std::cerr << "Error: matrix size must greater than zero.\n";
    return 1;
  }
  if(block_size <= 0) {
    std::cerr << "Error: block size must greater than zero.\n";
    return 1;
  }
  if((matrix_size % block_size) != 0ul) {
    std::cerr << "Error: matrix size must be evenly divisible by block size.\n";
    return 1;
  }
  if(sparsity < 0) {
    std::cerr << "Error: Sparsity must be greater than zero.\n";
    return 1;
  }
  if(sparsity > (matrix_size / 2)) {
    std::cerr << "Error: Sparsity must be less than 100.\n";
    return 1;
  }
  const long repeat = (argc >= 4 ? atol(argv[4]) : 5);
  if(repeat <= 0) {
    std::cerr << "Error: number of repititions must greater than zero.\n";
    return 1;
  }

  // Print information about the test
  const std::size_t num_blocks = matrix_size / block_size;
  const long block_count = (double(sparsity) / 100.0) * double(num_blocks * num_blocks);
  if(world.rank() == 0)
    std::cout << "TiledArray: block-sparse matrix multiply test...\n"
              << "Number of nodes    = " << world.size()
              << "\nMatrix size        = " << matrix_size << "x" << matrix_size
              << "\nBlock size         = " << block_size << "x" << block_size
              << "\nMemory per matrix  = " << double(block_count * block_size * block_size * sizeof(double)) / 1.0e9
              << " GB\nNumber of blocks   = " << block_count
              << "\nAverage blocks/node = " << double(block_count) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> blocking;
  blocking.reserve(num_blocks + 1);
  for(std::size_t i = 0; i <= matrix_size; i += block_size)
    blocking.push_back(i);

  std::vector<TiledArray::TiledRange1> blocking2(2,
      TiledArray::TiledRange1(blocking.begin(), blocking.end()));

  TiledArray::TiledRange
    trange(blocking2.begin(), blocking2.end());

  // Construct shape
  std::vector<std::size_t> a_shape, b_shape;
  if(world.rank() == 0) {
    a_shape.reserve(block_count);
    world.srand(time(NULL));
    for(long i = 0; i < block_count; ++i)
      a_shape.push_back(world.rand() % trange.tiles().volume());

    b_shape.reserve(block_count);
    for(long i = 0; i < block_count; ++i)
      b_shape.push_back(world.rand() % trange.tiles().volume());
  }

  // Construct and initialize arrays
  TiledArray::Array<double, 2> a(world, trange, a_shape.begin(), a_shape.end());
  TiledArray::Array<double, 2> b(world, trange, b_shape.begin(), b_shape.end());
  TiledArray::Array<double, 2> c(world, trange);
  a.set_all_local(1.0);
  b.set_all_local(1.0);

  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do matrix multiplication
  for(int i = 0; i < repeat; ++i) {
    c("m,n") = a("m,k") * b("k,n");
    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration " << i + 1 << "\n";
  }

  // Stop clock
  const double wall_time_stop = madness::wall_time();

  // Print results
  const long flop = 2.0 * TiledArray::expressions::sum(c("m,n"));
  if(world.rank() == 0) {
    std::cout << "Average wall time = " << (wall_time_stop - wall_time_start) / double(repeat)
        << "\nAverage GFLOPS = " << double(repeat) * double(flop) / (wall_time_stop - wall_time_start) / 1.0e9 << "\n";
  }


  madness::finalize();
  return 0;
}
