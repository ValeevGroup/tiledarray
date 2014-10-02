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
  TiledArray::TiledRange coeff_trange(coeff_blocking2.begin(), coeff_blocking2.end());
  TiledArray::TiledRange df_trange(df_blocking2.begin(), df_blocking2.end());
  TiledArray::TiledRange temp_trange(temp_blocking2.begin(), temp_blocking2.end());

  // Construct and initialize arrays
  TiledArray::Array<double, 2> C(world, coeff_trange);
  TiledArray::Array<double, 2> G(world, matrix_trange);
  TiledArray::Array<double, 2> H(world, matrix_trange);
  TiledArray::Array<double, 2> D(world, matrix_trange);
  TiledArray::Array<double, 2> F(world, matrix_trange);
  TiledArray::Array<double, 3> Eri(world, df_trange);
  TiledArray::Array<double, 3> K_temp(world, temp_trange);
  C.set_all_local(1.0);
  D.set_all_local(1.0);
  H.set_all_local(1.0);
  F.set_all_local(1.0);
  G.set_all_local(1.0);
  Eri.set_all_local(1.0);
  world.gop.fence();


  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do fock build
  for(int i = 0; i < repeat; ++i) {

    K_temp("j,Z,P") = C("m,Z") * Eri("m,j,P");

    // Compute coulomb and exchange
    G("i,j") = 2.0 * ( Eri("i,j,P") * ( C("m,Z") * K_temp("m,Z,P") ) )
                   - ( K_temp("i,Z,P") * K_temp("j,Z,P") );
    D("mu,nu") = C("mu,i") * C("nu,i");

    F("i,j") = G("i,j") + H("i,j");

    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration " << i + 1 << "\n";
  }

  // Stop clock
  const double wall_time_stop = madness::wall_time();

  const double total_time = wall_time_stop - wall_time_start;

  double gflops = 2.0 * double(coeff_size * matrix_size * matrix_size * df_size); // C("Z,m") * Eri("m,n,P") = K_temp("Z,n,P")
  gflops += 2.0 * double(coeff_size * matrix_size * df_size); // C("Z,n") * K_temp("Z,n,P") = temp("P")
  gflops += 2.0 * double(matrix_size * matrix_size * df_size); // Eri("i,j,P") * temp("P") = Final("i,j")
  gflops += 2.0 * double(coeff_size * matrix_size * matrix_size * df_size); // K_temp("Z,i,P") * K_temp("Z,j,P")
  gflops += 1.0 * double(matrix_size * matrix_size); // 2 * J("i,j") - K("i,j")
  gflops += 2.0 * double(coeff_size * matrix_size * matrix_size); // C("Z,mu") * C("Z,nu")
  gflops += double(matrix_size); // G("i,j") + H("i,j")
  gflops = double(repeat * gflops)/(1e9 * total_time);

  if(world.rank() == 0){
    std::cout << "Average wall time = " << (wall_time_stop - wall_time_start) / double(repeat) << std::endl;
    std::cout << "Memory needed (not including undeclared temporaries) = " <<
            4 * matrix_memory + coeff_memory + tensor_memory + co_tensor_memory << " GB" << std::endl;
    std::cout << "GFlops = " << gflops << std::endl;
  }

  madness::finalize();
  return 0;
}
