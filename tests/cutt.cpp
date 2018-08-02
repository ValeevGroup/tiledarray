/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Chong Peng
 *  Aug 2, 2018
 *
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include "unit_test_config.h"
#include <TiledArray/external/cuda.h>
#include <TiledArray/tensor/cuda/btas_um_tensor.h>

struct cuTTFixture{

  cuTTFixture() : N(1000), rank(2), extent({1000,1000}), perm({1,0}) {}

  std::size_t N;
  int rank;
  std::vector<int> extent;
  std::vector<int> perm;

};

BOOST_FIXTURE_TEST_SUITE(cutt_suite, cuTTFixture);

BOOST_AUTO_TEST_CASE( cutt_gpu_mem ) {

  int* a_host = (int*)std::malloc(N * N * sizeof(int));
  int* b_host = (int*)std::malloc(N * N * sizeof(int));
  int iter = 0;
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < N; j++) {
      a_host[iter] = iter;
      iter++;
    }
  }
  int* a_device;
  cudaMalloc(&a_device, N * N * sizeof(int));
  int* b_device;
  cudaMalloc(&b_device, N * N * sizeof(int));

  cudaMemcpy(a_device, a_host, N * N * sizeof(int), cudaMemcpyHostToDevice);

  cuttHandle plan;
  cuttResult_t status;

  status = cuttPlan(&plan, 2, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);
  cuttDestroy(plan);

  cudaMemcpy(b_host, b_device, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  iter = 0;
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < N; j++) {
      BOOST_CHECK(b_host[j * N + i] == iter);
      iter++;
    }
  }

}

BOOST_AUTO_TEST_CASE(cutt_unified_mem){
  int* a_um;
  cudaMallocManaged(&a_um, N * N * sizeof(int));

  int* b_um;
  cudaMallocManaged(&b_um, N * N * sizeof(int));

  int iter = 0;
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < N; j++) {
      a_um[iter] = iter;
      iter++;
    }
  }

  cuttHandle plan;
  cuttResult_t status;

  status = cuttPlan(&plan, 2, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_um, b_um);

  BOOST_CHECK(status == CUTT_SUCCESS);

  cudaDeviceSynchronize();

  cuttDestroy(plan);

  iter = 0;
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < N; j++) {
      BOOST_CHECK(b_um[j*N + i] == iter);
      iter++;
    }
  }
}

BOOST_AUTO_TEST_CASE( cutt_um_tensor ){

  TiledArray::Range range(std::vector<std::size_t>({N, N}));

  using Tile = btasUMTensorVarray<double, TiledArray::Range>;

  auto a = Tile(range);

  std::size_t iter = 0;

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < N; j++) {
      a[iter] = iter;
      iter++;
    }
  }

  TiledArray::Permutation permutation(perm);

  auto b = permute(a, permutation);

  cudaDeviceSynchronize();

  iter = 0;
  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < N; j++) {
      BOOST_CHECK(b[j * N + i] == iter);
      iter++;
    }
  }
}


BOOST_AUTO_TEST_SUITE_END()
#endif

