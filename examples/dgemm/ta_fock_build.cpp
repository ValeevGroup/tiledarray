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
  madness::World& world = madness::initialize(argc, argv);

  // Get command line arguments
  if(argc < 2) {
    std::cout << "Usage: fock_build matrix_size block_size df_size df_block_size [repetitions]\n";
    return 0;
  }
  const long matrix_size = atol(argv[1]);
  const long block_size = atol(argv[2]);
  const long df_size = atol(argv[3]);
  const long df_block_size = atol(argv[4]);
  if (matrix_size <= 0) {
    std::cerr << "Error: matrix size must greater than zero.\n";
    return 1;
  }
  if (df_size <= 0) {
    std::cerr << "Error: third rank size must greater than zero.\n";
    return 1;
  }
  if (block_size <= 0 || df_block_size <= 0) {
    std::cerr << "Error: block size must greater than zero.\n";
    return 1;
  }
  if(matrix_size % block_size != 0ul && df_size % df_block_size != 0ul) {
    std::cerr << "Error: tensor size must be evenly divisible by block size.\n";
    return 1;
  }
  const long repeat = (argc >= 6 ? atol(argv[5]) : 5);
  if (repeat <= 0) {
    std::cerr << "Error: number of repititions must greater than zero.\n";
    return 1;
  }

  const std::size_t num_blocks = matrix_size / block_size;
  const std::size_t df_num_blocks = df_size / df_block_size;
  const std::size_t block_count = num_blocks * num_blocks;
  const std::size_t df_block_count = df_num_blocks * num_blocks * num_blocks;

  if(world.rank() == 0)
    std::cout << "TiledArray: Fock Build Test ...\n"
              << "Number of nodes     = " << world.size()
              << "\nMatrix size         = " << matrix_size << "x" << matrix_size
              << "\nTensor size         = " << matrix_size << "x" << matrix_size << "x" << df_size
              << "\nBlock size          = " << block_size << "x" << block_size << "x" << df_block_size
              << "\nMemory per matrix   = " << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
              << " GB\nMemory per tensor   = " << double(matrix_size * matrix_size * df_size * sizeof(double)) / 1.0e9
              << " GB\nNumber of matrix blocks    = " << block_count
              << "\nNumber of tensor blocks    = " << df_block_count
              << "\nAverage blocks/node matrix = " << double(block_count) / double(world.size())
              << "\nAverage blocks/node tensor = " << double(df_block_count) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> blocking;
  blocking.reserve(num_blocks + 1);
  for(std::size_t i = 0; i <= matrix_size; i += block_size)
    blocking.push_back(i);

  std::vector<unsigned int> df_blocking;
  blocking.reserve(df_num_blocks + 1);
  for(std::size_t i = 0; i <= df_size; i += df_block_size)
    df_blocking.push_back(i);

  std::vector<TiledArray::TiledRange1> blocking2(2,
      TiledArray::TiledRange1(blocking.begin(), blocking.end()));

  std::vector<TiledArray::TiledRange1> blocking3 = {
      TiledArray::TiledRange1(blocking.begin(), blocking.end()),
      TiledArray::TiledRange1(blocking.begin(), blocking.end()),
      TiledArray::TiledRange1(df_blocking.begin(), df_blocking.end()) };


  TiledArray::TiledRange trange(blocking2.begin(), blocking2.end());
  TiledArray::TiledRange df_trange(blocking3.begin(), blocking3.end());

  // Construct and initialize arrays
  TiledArray::Array<double, 2> D(world, trange);
  TiledArray::Array<double, 2> DL(world, trange);
  TiledArray::Array<double, 2> F(world, trange);
  TiledArray::Array<double, 2> G(world, trange);
  TiledArray::Array<double, 2> H(world, trange);
  TiledArray::Array<double, 3> TCInts(world, df_trange);
  TiledArray::Array<double, 3> ExchTemp(world, df_trange);
  D.set_all_local(1.0);
  DL.set_all_local(1.0);
  H.set_all_local(2.0);
  TCInts.set_all_local(3.0);

  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do fock build
  for(int i = 0; i < repeat; ++i) {
      // Assume we have the cholesky decompositon of the density matrix
      ExchTemp("s,j,P") = DL("s,n") * TCInts("n,j,P");
      // Compute coulomb and exchange
      G("i,j") = 2.0 * TCInts("i,j,P") * ( D("n,m") * TCInts("n,m,P") ) -
                       ExchTemp("s,i,P") * ExchTemp("s,j,P");
      F("i,j") = G("i,j") + H("i,j");
      world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration " << i + 1 << "\n";
  }

  // Stop clock
  const double wall_time_stop = madness::wall_time();

  if(world.rank() == 0){
    std::cout << "Average wall time   = " << (wall_time_stop - wall_time_start) / double(repeat)
        << " sec\nAverage GFLOPS      = " << double(repeat) *
        (double(4.0 * matrix_size * matrix_size * df_size) + // Coulomb flops
        double(4.0 * matrix_size * matrix_size * matrix_size * df_size)) // Exchange flops
        / (wall_time_stop - wall_time_start) / 1.0e9 << "\n";
  }

  madness::finalize();
  return 0;
}
