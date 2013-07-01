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
#include <tiled_array.h>

int main(int argc, char** argv) {
  // Initialize runtime
  TiledArray::Runtime ta_runtime(argc,argv);
  madness::World& world = ta_runtime.get_world();

  // Get command line arguments
  if(argc < 2) {
    std::cout << "Usage: ta_band matrix_size block_size band_width [repetitions]\n";
    return 0;
  }
  const long matrix_size = atol(argv[1]);
  const long block_size = atol(argv[2]);
  const long band_width = atol(argv[3]);
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
  if(band_width < 0) {
    std::cerr << "Error: Band width must be greater than zero.\n";
    return 1;
  }
  if(band_width > (matrix_size / 2)) {
    std::cerr << "Error: Band width must be less than half the matrix size.\n";
    return 1;
  }
  const long repeat = (argc >= 4 ? atol(argv[4]) : 5);
  if(repeat <= 0) {
    std::cerr << "Error: number of repititions must greater than zero.\n";
    return 1;
  }

  const std::size_t num_blocks = matrix_size / block_size;
  std::size_t block_count = 0;
  for (int i = -band_width + 1; i < band_width; ++i) {
    block_count += num_blocks - (2 * std::abs(i));
  }

  if(world.rank() == 0)
    std::cout << "TiledArray: block-banded matrix multiply test...\n"
              << "Number of nodes    = " << world.size()
              << "\nMatrix size        = " << matrix_size << "x" << matrix_size
              << "\nBlock size         = " << block_size << "x" << block_size
              << "\nMemory per matrix  = " << double(block_count * block_size * block_size * sizeof(double)) / 1.0e9
              << " GB\nNumber of blocks   = " << block_count
              << "\nAverage blocks/node = " << block_count / world.size() << "\n";

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
  TiledArray::detail::Bitset<> shape(trange.tiles().volume());
  for(long i = 0; i < num_blocks; ++i) {
    long j = std::max<long>(i - band_width + 1, 0);
    const long j_end = std::min<long>(i + band_width - 1, num_blocks);
    long ij = i * num_blocks + j;
    for(; j < j_end; ++j, ++ij)
      shape.set(ij);
  }

  // Construct and initialize arrays
  TiledArray::Array<double, 2> a(world, trange, shape);
  TiledArray::Array<double, 2> b(world, trange, shape);
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

  return 0;
}
