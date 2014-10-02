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

template <typename it, typename tiletype>
void random_tile_task(it iter, tiletype tile){
  std::size_t size = tile.size();
  std::generate(tile.data(), tile.data()+size, []{return std::rand()%100;});
  *iter = tile;
}

TiledArray::Array<double, 2>
make_random_array(madness::World &world, TiledArray::TiledRange &trange){
  TiledArray::Array<double, 2> array(world, trange);
  typename TiledArray::Array<double, 2>::iterator it = array.begin();
  for(; it != array.end(); ++it){
    typename TiledArray::Array<double, 2>::value_type tile(
                                array.trange().make_tile_range(it.ordinal()));
    world.taskq.add(&random_tile_task<decltype(it), decltype(tile)>, it, tile);
  }
  return array;
}

int main(int argc, char** argv) {
  // Initialize runtime
  madness::World& world = madness::initialize(argc, argv);
  elem::Grid grid(elem::DefaultGrid().Comm());

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
    std::cout << "TiledArray: dense matrix multiply test...\n"
              << "Number of nodes     = " << world.size()
              << "\nMatrix size         = " << matrix_size << "x" << matrix_size
              << "\nBlock size          = " << block_size << "x" << block_size
              << "\nMemory per matrix   = " << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
              << " GB\nNumber of blocks    = " << block_count
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

  // Construct and initialize arrays
  TiledArray::Array<double, 2> a = make_random_array(world, trange);
  TiledArray::Array<double, 2> b = make_random_array(world, trange);
  TiledArray::Array<double, 2> c(world, trange);
  if(world.rank() == 0 && matrix_size < 11){
    std::cout << "a = \n" << a << std::endl;
    std::cout << "b = \n" << b << std::endl;
  }

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

  if(world.rank() == 0){
    std::cout << "Average wall time   = " << (wall_time_stop - wall_time_start) / double(repeat)
        << " sec\nAverage GFLOPS      = " << double(repeat) * 2.0 * double(matrix_size *
            matrix_size * matrix_size) / (wall_time_stop - wall_time_start) / 1.0e9 << "\n" << std::endl;
  }

  // Copying matrices to elemental
  elem::DistMatrix<double> a_elem = array_to_elem(a,grid);
  elem::DistMatrix<double> b_elem = array_to_elem(b,grid);
  elem::mpi::Barrier(grid.Comm());
  if(matrix_size < 11){
    Print(a_elem, "a from elem");
    Print(b_elem, "b from elem");
  }

  // Timed copy
  const double wall_time_copy0 = madness::wall_time();
  int j = 0;
  while(j++ < repeat){
    a_elem = array_to_elem(a,grid);
    b_elem = array_to_elem(b,grid);
    elem::mpi::Barrier(grid.Comm());
  }
  const double wall_time_copy1 = madness::wall_time();

  // How long the copy took
  if(world.rank() == 0){
    std::cout << "Spent " <<
      (wall_time_copy1 - wall_time_copy0)/(2.0 * double(repeat)) <<
      " s for an array copy to elemental on average.\n" << std::endl;
  }

  // Make the data output array
  elem::DistMatrix<double> c_elem(matrix_size, matrix_size, grid);
  elem::Zero(c_elem);
  elem::mpi::Barrier(grid.Comm());

  // Do the multiply
  const double wt_elem_start = madness::wall_time();
  for(std::size_t i = 0; i < repeat; ++i){
    elem::Gemm(elem::NORMAL, elem::NORMAL, 1., a_elem, b_elem, 0., c_elem);
    elem::mpi::Barrier(grid.Comm());
    if(grid.Rank() == 0){
      std::cout << "Elem Iteration " << i + 1 << "\n";
    }
  }
  const double wt_elem_end = madness::wall_time();

  // Time elemental
  if(world.rank() == 0){
    std::cout << "Average Elemental wall time   = " << (wt_elem_end - wt_elem_start) / double(repeat)
        << " sec\nAverage GFLOPS      = " << double(repeat) * 2.0 * double(matrix_size *
            matrix_size * matrix_size) / (wt_elem_end - wt_elem_start) / 1.0e9 << "\n";
  }

  // copy back to ta
  int i = 0;
  const double e_to_t_start = madness::wall_time();
  while(i++ < repeat){
    TiledArray::elem_to_array(c, c_elem);
    elem::mpi::Barrier(grid.Comm());
  }
  const double e_to_t_end = madness::wall_time();

  if(world.rank() == 0){
    std::cout << "Copying to TA from Elemental took " << (e_to_t_end - e_to_t_start)/(double(repeat)) << " s on average." << std::endl;
  }

  madness::finalize();
  return 0;
}
