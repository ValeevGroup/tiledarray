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
  // Initialize runtime
  madness::World& world = madness::initialize(argc, argv);

  // Get command line arguments
  if(argc < 2) {
    std::cout << "Usage: ta_dense row_size row_block_size col_size col_block_size [repetitions]\n";
    return 0;
  }
  const long row_size = atol(argv[1]);
  const long row_block_size = atol(argv[2]);
  const long col_size = atol(argv[3]);
  const long col_block_size = atol(argv[4]);
  if (row_size <= 0 || col_size <= 0) {
    std::cerr << "Error: dimensions must greater than zero.\n";
    return 1;
  }
  if (row_block_size <= 0 || col_block_size <= 0) {
    std::cerr << "Error: block sizes must greater than zero.\n";
    return 1;
  }
  if((row_size % row_block_size) != 0ul || col_size % col_block_size !=0ul) {
    std::cerr << "Error: diminsion size must be evenly divisible by block size.\n";
    return 1;
  }
  const long repeat = (argc >= 6 ? atol(argv[5]) : 5);
  if (repeat <= 0) {
    std::cerr << "Error: number of repetitions must greater than zero.\n";
    return 1;
  }

  const std::size_t row_blocks = row_size / row_block_size;
  const std::size_t col_blocks = col_size / col_block_size;
  const std::size_t block_count = row_blocks * col_blocks;
//  const std::size_t matrix_size = row_size * col_size;

  if(world.rank() == 0)
    std::cout << "TiledArray: dense matrix multiply test...\n"
              << "Number of nodes     = " << world.size()
              << "\nMatrix size         = " << row_size << "x" << col_size
              << "\nBlock size          = " << row_block_size << "x" << col_block_size
              << "\nMemory per matrix   = " << double(row_size * col_size * sizeof(double)) / 1.0e9
              << " GB\nNumber of blocks    = " << block_count
              << "\nAverage blocks/node = " << double(block_count) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> blocking_row;
  blocking_row.reserve(row_blocks + 1);
  for(long i = 0l; i <= row_size; i += row_block_size)
    blocking_row.push_back(i);

  std::vector<unsigned int> blocking_col;
  blocking_col.reserve(col_blocks + 1);
  for(long i = 0l; i <= col_size; i += col_block_size)
    blocking_col.push_back(i);

  // Stucture of c
  std::vector<TiledArray::TiledRange1> blocking_result;
  blocking_result.reserve(2);
  blocking_result.push_back(TiledArray::TiledRange1(blocking_row.begin(),blocking_row.end()));
  blocking_result.push_back(TiledArray::TiledRange1(blocking_row.begin(), blocking_row.end()));

  // Strucure of a
  std::vector<TiledArray::TiledRange1> blocking_rowxcol;
  blocking_rowxcol.reserve(2);
  blocking_rowxcol.push_back(TiledArray::TiledRange1(blocking_row.begin(),blocking_row.end()));
  blocking_rowxcol.push_back(TiledArray::TiledRange1(blocking_col.begin(), blocking_col.end()));

  // Structure of b
  std::vector<TiledArray::TiledRange1> blocking_colxrow;
  blocking_colxrow.reserve(2);
  blocking_rowxcol.push_back(TiledArray::TiledRange1(blocking_col.begin(),blocking_col.end()));
  blocking_rowxcol.push_back(TiledArray::TiledRange1(blocking_row.begin(), blocking_row.end()));

  TiledArray::TiledRange // TRange for c
    trange(blocking_result.begin(), blocking_result.end());

  TiledArray::TiledRange // TRange for a
    trange_a(blocking_rowxcol.begin(), blocking_rowxcol.end());

  TiledArray::TiledRange // TRange for b
    trange_b(blocking_colxrow.begin(), blocking_colxrow.end());

  // Construct and initialize arrays
  TiledArray::Array<double, 2> a(world, trange_a);
  TiledArray::Array<double, 2> b(world, trange_b);
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

  if(world.rank() == 0)
    std::cout << "Average wall time   = " << (wall_time_stop - wall_time_start) / double(repeat)
        << " sec\nAverage GFLOPS      = " << double(repeat) * 2.0 * double(col_size *
            row_size * row_size) / (wall_time_stop - wall_time_start) / 1.0e9 << "\n";

  madness::finalize();
  return 0;
}
