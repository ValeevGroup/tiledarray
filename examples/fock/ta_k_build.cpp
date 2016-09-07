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
#include <madness/world/worldmem.h>

namespace TA = TiledArray;
using array_type = TA::TSpArrayD;

int main(int argc, char** argv) {
  // Initialize runtime
  TA::World& world = TA::initialize(argc, argv);
  madness::print_meminfo(world.rank(), "t=0");

  // Get command line arguments
  if(argc < 2) {
    std::cout << "Usage: ta_k_build ao_size ao_blk_size occ_size occ_block_size df_size df_blk_size [repetitions]\n";
    return 0;
  }
  const long ao_size = atol(argv[1]);
  const long ao_blk_size = atol(argv[2]);
  const long occ_size = atol(argv[3]);
  const long occ_blk_size = atol(argv[4]);
  const long df_size = atol(argv[5]);
  const long df_blk_size = atol(argv[6]);
  if (ao_size <= 0) {
    std::cerr << "Error: ao size must greater than zero.\n";
    return 1;
  }
  if (occ_size <= 0) {
    std::cerr << "Error: occ size must greater than zero.\n";
    return 1;
  }
  if (df_size <= 0) {
    std::cerr << "Error: df size must greater than zero.\n";
    return 1;
  }
  if (ao_blk_size <= 0 || df_blk_size <= 0 || occ_blk_size <= 0) {
    std::cerr << "Error: block sizes must greater than zero.\n";
    return 1;
  }
  if(ao_size % ao_blk_size != 0ul && df_size % df_blk_size != 0ul && occ_size % occ_blk_size != 0) {
    std::cerr << "Error: tensor sizes must be evenly divisible by block sizes.\n";
    return 1;
  }
  const long repeat = (argc >= 8 ? atol(argv[7]) : 5);
  if (repeat <= 0) {
    std::cerr << "Error: number of repetitions must greater than zero.\n";
    return 1;
  }

  const std::size_t num_blocks = ao_size / ao_blk_size;
  const std::size_t coeff_num_blocks = occ_size / occ_blk_size;
  const std::size_t df_num_blocks = df_size / df_blk_size;
  const std::size_t block_count = num_blocks * num_blocks;
  const std::size_t coeff_block_count = coeff_num_blocks * coeff_num_blocks;
  const std::size_t df_block_count = df_num_blocks * num_blocks * num_blocks;

  // Memory used
  double tensor_memory = double(ao_size * ao_size * df_size * sizeof(double))/1e9;
  double co_tensor_memory = double(occ_size * ao_size * df_size * sizeof(double))/1e9;

  if(world.rank() == 0)
    std::cout << "TiledArray: Fock Build Test ...\n"
              << "Number of nodes     = " << world.size()
              << "\nAO size         = " << ao_size << "x" << ao_size
              << "\nocc size         = " << occ_size << "x" << ao_size
              << "\(ao ao|df) tensor ranks         = " << ao_size << "x" << ao_size << "x" << df_size
              << "\(ao ao|df) tensor block ranks          = " << ao_blk_size << "x" << ao_blk_size << "x" << df_blk_size
              << "\n(ao ao|df) tensor storage = " << tensor_memory
              << " GB\n(ao occ|df) tensor storage   = " << co_tensor_memory
              << " GB\n";

  // Construct TiledRange
  std::vector<unsigned int> ao_blocking;
  ao_blocking.reserve(num_blocks + 1);
  for(long i = 0; i <= ao_size; i += ao_blk_size)
    ao_blocking.push_back(i);

  std::vector<unsigned int> occ_blocking;
  occ_blocking.reserve(coeff_num_blocks + 1);
  for(long i = 0; i <= occ_size; i += occ_blk_size)
    occ_blocking.push_back(i);

  std::vector<unsigned int> df_blocking;
  df_blocking.reserve(df_num_blocks + 1);
  for(long i = 0; i <= df_size; i += df_blk_size)
    df_blocking.push_back(i);

  std::vector<TA::TiledRange1> ao_blocking2(
    2, TA::TiledRange1(ao_blocking.begin(), ao_blocking.end())
  );

  // Create M^-1 blocking
  std::vector<TA::TiledRange1> df_blocking2(
    2, TA::TiledRange1(df_blocking.begin(), df_blocking.end())
  );

  // Create C^T blocking
  std::vector<TA::TiledRange1> coeff_blocking;
  coeff_blocking.reserve(2);
  coeff_blocking.push_back(TA::TiledRange1(ao_blocking.begin(), ao_blocking.end()));
  coeff_blocking.push_back(TA::TiledRange1(occ_blocking.begin(), occ_blocking.end()));

  std::vector<TA::TiledRange1> aad_blocking;
  aad_blocking.reserve(3);
  aad_blocking.push_back(TA::TiledRange1(ao_blocking.begin(), ao_blocking.end()));
  aad_blocking.push_back(TA::TiledRange1(ao_blocking.begin(), ao_blocking.end()));
  aad_blocking.push_back(TA::TiledRange1(df_blocking.begin(), df_blocking.end()));

  std::vector<TA::TiledRange1> oad_blocking;
  oad_blocking.reserve(3);
  oad_blocking.push_back(TA::TiledRange1(occ_blocking.begin(), occ_blocking.end()));
  oad_blocking.push_back(TA::TiledRange1(ao_blocking.begin(), ao_blocking.end()));
  oad_blocking.push_back(TA::TiledRange1(df_blocking.begin(), df_blocking.end()));

  TA::TiledRange ao_matrix_trange(ao_blocking2.begin(), ao_blocking2.end());
  TA::TiledRange df_matrix_trange(df_blocking2.begin(), df_blocking2.end());
  TA::TiledRange coeff_trange(coeff_blocking.begin(), coeff_blocking.end());
  TA::TiledRange aad_trange(aad_blocking.begin(), aad_blocking.end());
  TA::TiledRange oad_trange(oad_blocking.begin(), oad_blocking.end());

  // make shapes
  auto make_shape = [](const TA::TiledRange& trange) -> TA::SparseShape<float> {
    TA::Tensor<float> tile_norms(trange.tiles());
    for(auto& tile_norm: tile_norms) tile_norm = 1.e20;
    //std::cout << "shape_norms = " << tile_norms << std::endl;
    TA::SparseShape<float> result(tile_norms, trange);
    //std::cout << "result.data() = " << result.data() << std::endl;
    return result;
  };
  auto coeff_shape = make_shape(coeff_trange);
  //std::cout << "coeff_shape.data() = " << coeff_shape.data() << std::endl;
  auto ao_matrix_shape = make_shape(ao_matrix_trange);
  //std::cout << "ao_matrix_shape.data() = " << ao_matrix_shape.data() << std::endl;
  auto df_matrix_shape = make_shape(df_matrix_trange);
  //std::cout << "df_matrix_shape.data() = " << df_matrix_shape.data() << std::endl;
  auto aad_shape = make_shape(aad_trange);
  //std::cout << "aad_shape.data() = " << aad_shape.data() << std::endl;
  auto oad_shape = make_shape(oad_trange);
  //std::cout << "oad_shape.data() = " << oad_shape.data() << std::endl;

  // Construct and initialize arrays
  madness::print_meminfo(world.rank(), "before allocation");
  array_type C(world, coeff_trange, coeff_shape);
  C.fill(1.0);
  world.gop.fence();
  madness::print_meminfo(world.rank(), "made C");
  array_type K(world, ao_matrix_trange, ao_matrix_shape);
  K.fill(1.0);
  world.gop.fence();
  madness::print_meminfo(world.rank(), "made K");
  array_type M_oh_inv(world, df_matrix_trange, df_matrix_shape);
  M_oh_inv.fill(1.0);
  world.gop.fence();
  madness::print_meminfo(world.rank(), "made M_oh_inv");
  array_type Eri(world, aad_trange, aad_shape);
  Eri.fill(1.0);
  world.gop.fence();
  madness::print_meminfo(world.rank(), "made Eri");
  array_type K_temp(world, oad_trange, oad_shape);
  world.gop.fence();

  // Time first part of exchange build
  if(world.rank() == 0){
    std::cout << "\nStarting K1" << std::endl; 
  }
  world.gop.fence();
  double k1_time_start = madness::wall_time();
  
  // Do K build
  // NB 1 extra iteration to warm up
  for(int i = 0; i < repeat+1; ++i) {
    if (i == 1)
      k1_time_start = madness::wall_time();
  
    K_temp("j,Z,P") = C("m,Z") * Eri("m,j,P");
    world.gop.fence();
    madness::print_meminfo(world.rank(), "made K1");
    if(world.rank() == 0) {
      if (i == 0)
        std::cout << "Warmup ... ready to work now" << std::endl;
      else
        std::cout << "Iteration: "  << i << "   " << "\r" << std::flush;
    }
  }
  std::cout << std::endl;
  const double k1_time_stop = madness::wall_time();

  double k1_time = k1_time_stop - k1_time_start;
  double k1_gflops = 2.0 * double(occ_size * ao_size * ao_size * df_size); // C("Z,m") * Eri("m,n,P") = K_temp("Z,n,P")
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
    madness::print_meminfo(world.rank(), "made K2");
    if(world.rank() == 0)
      std::cout << "Iteration: "  << i + 1 << "   " << "\r" << std::flush;
  }
  std::cout << std::endl;
  const double k2_time_stop = madness::wall_time();

  double k2_time = k2_time_stop - k2_time_start;
  double k2_gflops = 2.0 * double(df_size * df_size * ao_size * occ_size); // K_temp("j,Z,P") = K_temp("j, Z, X") * M_oh_size("X,P")
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
    madness::print_meminfo(world.rank(), "made K3");
    if(world.rank() == 0)
      std::cout << "Iteration: "  << i + 1 << "   " << "\r" << std::flush;
  }
  std::cout << std::endl;
  const double k3_time_stop = madness::wall_time();

  double k3_time = k3_time_stop - k3_time_start;
  double k3_gflops = 2.0 * double(occ_size * ao_size * ao_size * df_size); // C("Z,m") * Eri("m,n,P") = K_temp("Z,n,P")
  k3_gflops *= repeat;
  k3_gflops /= (1e9 * k3_time);

  world.gop.fence();
  
  if(world.rank() == 0){
    std::cout << "Average K3 time = " << double(k3_time) / double(repeat) << std::endl;
    std::cout << "K3 GFlops = " << k3_gflops << std::endl;
  }

  // build the whole exchange as in MPQC4
  {
    auto compute_G = [&]() -> array_type {
      array_type W;
      W("X, rho, i") = M_oh_inv("X,Y") * (Eri("rho, sig, Y") * C("sig, i"));

      // Make J
      array_type J;
      J("mu, nu") = Eri("mu, nu, Z")
                      * (M_oh_inv("X, Z") * (W("X, rho, i") * C("rho, i")));

      // Permute W
      W("X, i, rho") = W("X, rho, i");

      array_type K;
      K("mu, nu") = W("X, i, mu") * W("X, i, nu");
      array_type G;
      G("mu, nu") = 2 * J("mu, nu") - K("mu, nu");
      world.gop.fence();
      madness::print_meminfo(world.rank(), "made J+K in 1 shot");
      return G;
    };

    array_type G = compute_G();
    for(int i = 1; i < repeat; ++i)
      G("i,j") += compute_G()("i,j");
  }
  world.gop.fence();
  madness::print_meminfo(world.rank(), "made J+K in 1 shot");

  TA::finalize();
  return 0;
}
