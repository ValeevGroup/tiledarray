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
  TiledArray::World& world = TiledArray::initialize(argc, argv);

  // Get command line arguments
  if(argc < 2) {
    std::cout << "Usage: fock_build matrix_size block_size coeff_size coeff_block_size df_size df_block_size [repetitions]\n";
    return 0;
  }
  const long matrix_size = atol(argv[1]);
  const long block_size = atol(argv[2]);
  const long coeff_size = atol(argv[3]);
  const long coeff_block_size = atol(argv[4]);
  const long df_size = atol(argv[5]);
  const long df_block_size = atol(argv[6]);
  if (matrix_size <= 0) {
    std::cerr << "Error: matrix size must greater than zero.\n";
    return 1;
  }
  if (df_size <= 0) {
    std::cerr << "Error: third rank size must greater than zero.\n";
    return 1;
  }
  if (coeff_size <= 0) {
    std::cerr << "Error: coeff size must greater than zero.\n";
    return 1;
  }
  if (block_size <= 0 || df_block_size <= 0 || coeff_block_size <= 0) {
    std::cerr << "Error: block sizes must greater than zero.\n";
    return 1;
  }
  if(matrix_size % block_size != 0ul && df_size % df_block_size != 0ul && coeff_size % coeff_block_size != 0) {
    std::cerr << "Error: tensor sizes must be evenly divisible by block sizes.\n";
    return 1;
  }
  const long repeat = (argc >= 8 ? atol(argv[7]) : 5);
  if (repeat <= 0) {
    std::cerr << "Error: number of repetitions must greater than zero.\n";
    return 1;
  }

  const std::size_t num_blocks = matrix_size / block_size;
  const std::size_t coeff_num_blocks = coeff_size / coeff_block_size;
  const std::size_t df_num_blocks = df_size / df_block_size;
  const std::size_t block_count = num_blocks * num_blocks;
  const std::size_t coeff_block_count = coeff_num_blocks * coeff_num_blocks;
  const std::size_t df_block_count = df_num_blocks * num_blocks * num_blocks;

  // Memory used
  double matrix_memory = double(matrix_size * matrix_size * sizeof(double))/1e9;
  double coeff_memory = double(matrix_size * coeff_size * sizeof(double))/1e9;
  double tensor_memory = double(matrix_size * matrix_size * df_size * sizeof(double))/1e9;
  double co_tensor_memory = double(coeff_size * matrix_size * df_size * sizeof(double))/1e9;

  if(world.rank() == 0)
    std::cout << "TiledArray: Fock Build Test ...\n"
              << "Number of nodes     = " << world.size()
              << "\nMatrix size         = " << matrix_size << "x" << matrix_size
              << "\nCoeff size         = " << coeff_size << "x" << matrix_size
              << "\nTensor size         = " << matrix_size << "x" << matrix_size << "x" << df_size
              << "\nBlock size          = " << block_size << "x" << block_size << "x" << df_block_size
              << "\nMemory per matrix   = " << matrix_memory
              << " GB\nMemory per coeff   = " << coeff_memory
              << " GB\nMemory per tensor   = " << tensor_memory
              << " GB\nMemory per temp_tensor   = " << co_tensor_memory
              << " GB\nNumber of matrix blocks    = " << block_count
              << "\nNumber of coeff blocks    = " << coeff_block_count
              << "\nNumber of tensor blocks    = " << df_block_count
              << "\nAverage blocks/node matrix = " << double(block_count) / double(world.size())
              << "\nAverage blocks/node coeff = " << double(coeff_block_count) / double(world.size())
              << "\nAverage blocks/node tensor = " << double(df_block_count) / double(world.size()) << "\n";

  // Construct TiledRange
  std::vector<unsigned int> matrix_blocking;
  matrix_blocking.reserve(num_blocks + 1);
  for(long i = 0; i <= matrix_size; i += block_size)
    matrix_blocking.push_back(i);

  std::vector<unsigned int> coeff_blocking;
  coeff_blocking.reserve(coeff_num_blocks + 1);
  for(long i = 0; i <= coeff_size; i += coeff_block_size)
    coeff_blocking.push_back(i);

  std::vector<unsigned int> df_blocking;
  df_blocking.reserve(df_num_blocks + 1);
  for(long i = 0; i <= df_size; i += df_block_size)
    df_blocking.push_back(i);

  std::vector<TiledArray::TiledRange1> matrix_blocking2(
    2, TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end())
  );

  // Create M^-1 blocking
  std::vector<TiledArray::TiledRange1> df_matrix_blocking2(
    2, TiledArray::TiledRange1(df_blocking.begin(), df_blocking.end())
  );

  // Create C^T blocking
  std::vector<TiledArray::TiledRange1> coeff_blocking2;
  coeff_blocking2.reserve(2);
  coeff_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  coeff_blocking2.push_back(TiledArray::TiledRange1(coeff_blocking.begin(), coeff_blocking.end()));

  std::vector<TiledArray::TiledRange1> df_blocking2;
  df_blocking2.reserve(3);
  df_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  df_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  df_blocking2.push_back(TiledArray::TiledRange1(df_blocking.begin(), df_blocking.end()));

  std::vector<TiledArray::TiledRange1> temp_blocking2;
  temp_blocking2.reserve(3);
  temp_blocking2.push_back(TiledArray::TiledRange1(coeff_blocking.begin(), coeff_blocking.end()));
  temp_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  temp_blocking2.push_back(TiledArray::TiledRange1(df_blocking.begin(), df_blocking.end()));

  TiledArray::TiledRange matrix_trange(matrix_blocking2.begin(), matrix_blocking2.end());
  TiledArray::TiledRange df_matrix_trange(df_matrix_blocking2.begin(), df_matrix_blocking2.end());
  TiledArray::TiledRange coeff_trange(coeff_blocking2.begin(), coeff_blocking2.end());
  TiledArray::TiledRange df_trange(df_blocking2.begin(), df_blocking2.end());
  TiledArray::TiledRange temp_trange(temp_blocking2.begin(), temp_blocking2.end());

  // Construct and initialize arrays
  TiledArray::TArrayD C(world, coeff_trange);
  TiledArray::TArrayD K(world, matrix_trange);
  TiledArray::TArrayD M_oh_inv(world, df_matrix_trange);
  TiledArray::TArrayD Eri(world, df_trange);
  TiledArray::TArrayD K_temp(world, temp_trange);
  C.fill(1.0);
  M_oh_inv.fill(1.0);
  K.fill(1.0);
  Eri.fill(1.0);
  world.gop.fence();


  // Time first part of exchange build
  if(world.rank() == 0){
    std::cout << "\nStarting K1" << std::endl; 
  }
  world.gop.fence();
  const double k1_time_start = madness::wall_time();
  
  // Do K build
  for(int i = 0; i < repeat; ++i) {
  
    K_temp("j,Z,P") = C("m,Z") * Eri("m,j,P");
    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration: "  << i + 1 << "   " << "\r" << std::flush;
  }
  std::cout << std::endl;
  const double k1_time_stop = madness::wall_time();

  double k1_time = k1_time_stop - k1_time_start;
  double k1_gflops = 2.0 * double(coeff_size * matrix_size * matrix_size * df_size); // C("Z,m") * Eri("m,n,P") = K_temp("Z,n,P")
  k1_gflops *= repeat;
  k1_gflops /= (1e9 * k1_time);
  
  if(world.rank() == 0){
    std::cout << "Average K1 time = " << double(k1_time) / double(repeat) << std::endl;
    std::cout << "K1 GFlops = " << k1_gflops << std::endl;
  }

  // Starting K2 
  if(world.rank() == 0){
    std::cout << "\nStarting K2" << std::endl; 
  }
  world.gop.fence();
  const double k2_time_start = madness::wall_time();
  
  // Do K build
  for(int i = 0; i < repeat; ++i) {
  
    K_temp("j,Z,P") = K_temp("j, Z, X") * M_oh_inv("X,P");
    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration: "  << i + 1 << "   " << "\r" << std::flush;
  }
  std::cout << std::endl;
  const double k2_time_stop = madness::wall_time();

  double k2_time = k2_time_stop - k2_time_start;
  double k2_gflops = 2.0 * double(df_size * df_size * matrix_size * coeff_size); // K_temp("j,Z,P") = K_temp("j, Z, X") * M_oh_size("X,P")
  k2_gflops *= repeat;
  k2_gflops /= (1e9 * k2_time);
  
  if(world.rank() == 0){
    std::cout << "Average K2 time = " << double(k2_time) / double(repeat) << std::endl;
    std::cout << "K2 GFlops = " << k2_gflops << std::endl;
  }

  // STARTING K2 
  if(world.rank() == 0){
    std::cout << "\nStarting K3" << std::endl; 
  }
  world.gop.fence();
  const double k3_time_start = madness::wall_time();
  
  // Do K build
  for(int i = 0; i < repeat; ++i) {
  
    K("i,j") = K_temp("i,Z,P") * K_temp("j,Z,P");

    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration: "  << i + 1 << "   " << "\r" << std::flush;
  }
  std::cout << std::endl;
  const double k3_time_stop = madness::wall_time();

  double k3_time = k3_time_stop - k3_time_start;
  double k3_gflops = 2.0 * double(coeff_size * matrix_size * matrix_size * df_size); // C("Z,m") * Eri("m,n,P") = K_temp("Z,n,P")
  k3_gflops *= repeat;
  k3_gflops /= (1e9 * k3_time);

  world.gop.fence();
  
  if(world.rank() == 0){
    std::cout << "Average K3 time = " << double(k3_time) / double(repeat) << std::endl;
    std::cout << "K3 GFlops = " << k3_gflops << std::endl;
  }

  TiledArray::finalize();
  return 0;
}
