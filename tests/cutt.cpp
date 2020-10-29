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

#include <TiledArray/cuda/btas_um_tensor.h>
#include "unit_test_config.h"

struct cuTTFixture {
  //  cuTTFixture()
  //      : A(100),
  //        B(50),
  //        C(20),
  //        rank(2),
  //        extent({100, 100}),
  //        extent_nonsym({100, 50}),
  //        perm({1, 0}) {}
  cuTTFixture() : A(10), B(5), C(2) {}

  int A;
  int B;
  int C;
};

BOOST_FIXTURE_TEST_SUITE(cutt_suite, cuTTFixture, TA_UT_SKIP_IF_DISTRIBUTED);

BOOST_AUTO_TEST_CASE(cutt_gpu_mem) {
  int* a_host = (int*)std::malloc(A * A * sizeof(int));
  int* b_host = (int*)std::malloc(A * A * sizeof(int));
  int iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < A; j++) {
      a_host[iter] = iter;
      iter++;
    }
  }
  int* a_device;
  cudaMalloc(&a_device, A * A * sizeof(int));
  int* b_device;
  cudaMalloc(&b_device, A * A * sizeof(int));

  cudaMemcpy(a_device, a_host, A * A * sizeof(int), cudaMemcpyHostToDevice);

  std::vector<int> extent({A, A});
  TiledArray::extent_to_col_major(extent);

  std::vector<int> perm({1, 0});
  TiledArray::permutation_to_col_major(perm);

  cuttHandle plan;
  cuttResult_t status;

  status = cuttPlan(&plan, 2, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);
  cuttDestroy(plan);

  cudaMemcpy(b_host, b_device, A * A * sizeof(int), cudaMemcpyDeviceToHost);

  iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < A; j++) {
      BOOST_CHECK(b_host[j * A + i] == iter);
      iter++;
    }
  }

  free(a_host);
  free(b_host);

  cudaFree(a_device);
  cudaFree(b_device);
}

BOOST_AUTO_TEST_CASE(cutt_gpu_mem_nonsym) {
  int* a_host = (int*)std::malloc(A * B * sizeof(int));
  int* b_host = (int*)std::malloc(A * B * sizeof(int));
  int iter = 0;
  for (std::size_t i = 0; i < B; i++) {
    for (std::size_t j = 0; j < A; j++) {
      a_host[iter] = iter;
      iter++;
    }
  }

  int* a_device;
  cudaMalloc(&a_device, A * B * sizeof(int));
  int* b_device;
  cudaMalloc(&b_device, A * B * sizeof(int));

  cudaMemcpy(a_device, a_host, A * B * sizeof(int), cudaMemcpyHostToDevice);

  cuttHandle plan;
  cuttResult_t status;

  std::vector<int> extent({B, A});
  TiledArray::extent_to_col_major(extent);

  std::vector<int> perm({1, 0});
  TiledArray::permutation_to_col_major(perm);

  status = cuttPlan(&plan, 2, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);
  cuttDestroy(plan);

  cudaMemcpy(b_host, b_device, A * B * sizeof(int), cudaMemcpyDeviceToHost);

  iter = 0;
  for (std::size_t i = 0; i < B; i++) {
    for (std::size_t j = 0; j < A; j++) {
      BOOST_CHECK(b_host[j * B + i] == iter);
      iter++;
    }
  }

  free(a_host);
  free(b_host);

  cudaFree(a_device);
  cudaFree(b_device);
}

BOOST_AUTO_TEST_CASE(cutt_gpu_mem_nonsym_rank_three_column_major) {
  int* a_host = (int*)std::malloc(A * B * C * sizeof(int));
  int* b_host = (int*)std::malloc(A * B * C * sizeof(int));
  int iter = 0;
  for (std::size_t k = 0; k < C; k++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t i = 0; i < A; i++) {
        a_host[k * A * B + j * A + i] = iter;
        iter++;
      }
    }
  }

  int* a_device;
  cudaMalloc(&a_device, A * B * C * sizeof(int));
  int* b_device;
  cudaMalloc(&b_device, A * B * C * sizeof(int));

  cudaMemcpy(a_device, a_host, A * B * C * sizeof(int), cudaMemcpyHostToDevice);

  // b(j,i,k) = a(i,j,k)

  cuttHandle plan;
  cuttResult_t status;

  std::vector<int> extent3{int(A), int(B), int(C)};

  std::vector<int> perm3{1, 0, 2};
  //  std::vector<int> perm3{0, 2, 1};

  status = cuttPlanMeasure(&plan, 3, extent3.data(), perm3.data(), sizeof(int),
                           0, a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);

  cudaMemcpy(b_host, b_device, A * B * C * sizeof(int), cudaMemcpyDeviceToHost);

  status = cuttDestroy(plan);

  BOOST_CHECK(status == CUTT_SUCCESS);

  iter = 0;
  for (std::size_t k = 0; k < C; k++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t i = 0; i < A; i++) {
        BOOST_CHECK_EQUAL(b_host[k * A * B + i * B + j], iter);
        iter++;
      }
    }
  }

  free(a_host);
  free(b_host);

  cudaFree(a_device);
  cudaFree(b_device);
}

BOOST_AUTO_TEST_CASE(cutt_gpu_mem_nonsym_rank_three_row_major) {
  int* a_host = (int*)std::malloc(A * B * C * sizeof(int));
  int* b_host = (int*)std::malloc(A * B * C * sizeof(int));
  int iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t k = 0; k < C; k++) {
        a_host[i * C * B + j * C + k] = iter;
        iter++;
      }
    }
  }

  int* a_device;
  cudaMalloc(&a_device, A * B * C * sizeof(int));
  int* b_device;
  cudaMalloc(&b_device, A * B * C * sizeof(int));

  cudaMemcpy(a_device, a_host, A * B * C * sizeof(int), cudaMemcpyHostToDevice);

  // b(j,i,k) = a(i,j,k)

  cuttHandle plan;
  cuttResult_t status;

  std::vector<int> extent({A, B, C});
  TiledArray::extent_to_col_major(extent);

  std::vector<int> perm({1, 0, 2});
  TiledArray::permutation_to_col_major(perm);

  status = cuttPlanMeasure(&plan, 3, extent.data(), perm.data(), sizeof(int), 0,
                           a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_device, b_device);

  BOOST_CHECK(status == CUTT_SUCCESS);

  cudaMemcpy(b_host, b_device, A * B * C * sizeof(int), cudaMemcpyDeviceToHost);

  status = cuttDestroy(plan);

  BOOST_CHECK(status == CUTT_SUCCESS);

  iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t k = 0; k < C; k++) {
        BOOST_CHECK_EQUAL(b_host[j * A * C + i * C + k], iter);
        iter++;
      }
    }
  }

  free(a_host);
  free(b_host);

  cudaFree(a_device);
  cudaFree(b_device);
}

BOOST_AUTO_TEST_CASE(cutt_unified_mem) {
  int* a_um;
  cudaMallocManaged(&a_um, A * A * sizeof(int));

  int* b_um;
  cudaMallocManaged(&b_um, A * A * sizeof(int));

  int iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < A; j++) {
      a_um[iter] = iter;
      iter++;
    }
  }

  cuttHandle plan;
  cuttResult_t status;

  std::vector<int> extent({A, A});
  TiledArray::extent_to_col_major(extent);

  std::vector<int> perm({1, 0});
  TiledArray::permutation_to_col_major(perm);

  status = cuttPlan(&plan, 2, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_um, b_um);

  BOOST_CHECK(status == CUTT_SUCCESS);

  cuttDestroy(plan);

  cudaDeviceSynchronize();

  iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < A; j++) {
      BOOST_CHECK(b_um[j * A + i] == iter);
      iter++;
    }
  }

  cudaFree(a_um);
  cudaFree(b_um);
}

BOOST_AUTO_TEST_CASE(cutt_unified_mem_nonsym) {
  int* a_um;
  cudaMallocManaged(&a_um, A * B * sizeof(int));

  int* b_um;
  cudaMallocManaged(&b_um, A * B * sizeof(int));

  int iter = 0;
  for (std::size_t i = 0; i < B; i++) {
    for (std::size_t j = 0; j < A; j++) {
      a_um[iter] = iter;
      iter++;
    }
  }

  cuttHandle plan;
  cuttResult_t status;

  std::vector<int> extent({B, A});
  TiledArray::extent_to_col_major(extent);

  std::vector<int> perm({1, 0});
  TiledArray::permutation_to_col_major(perm);

  status = cuttPlan(&plan, 2, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_um, b_um);

  BOOST_CHECK(status == CUTT_SUCCESS);

  cuttDestroy(plan);
  cudaDeviceSynchronize();

  iter = 0;
  for (std::size_t i = 0; i < B; i++) {
    for (std::size_t j = 0; j < A; j++) {
      BOOST_CHECK(b_um[j * B + i] == iter);
      iter++;
    }
  }
  cudaFree(a_um);
  cudaFree(b_um);
}

BOOST_AUTO_TEST_CASE(cutt_unified_mem_rank_three) {
  int* a_um;
  cudaMallocManaged(&a_um, A * B * C * sizeof(int));

  int* b_um;
  cudaMallocManaged(&b_um, A * B * C * sizeof(int));

  int iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t k = 0; k < C; k++) {
        a_um[iter] = iter;
        iter++;
      }
    }
  }

  cuttHandle plan;
  cuttResult_t status;

  // b(k,i,j) = a(i,j,k)

  std::vector<int> extent({A, B, C});
  TiledArray::extent_to_col_major(extent);

  std::vector<int> perm({2, 0, 1});
  TiledArray::permutation_to_col_major(perm);

  status = cuttPlan(&plan, 3, extent.data(), perm.data(), sizeof(int), 0);

  BOOST_CHECK(status == CUTT_SUCCESS);

  status = cuttExecute(plan, a_um, b_um);

  BOOST_CHECK(status == CUTT_SUCCESS);

  cuttDestroy(plan);
  cudaDeviceSynchronize();

  iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t k = 0; k < C; k++) {
        BOOST_CHECK(b_um[k * A * B + i * B + j] == iter);
        iter++;
      }
    }
  }
  cudaFree(a_um);
  cudaFree(b_um);
}

BOOST_AUTO_TEST_CASE(cutt_um_tensor) {
  TiledArray::Range range{A, A};

  using Tile = TiledArray::btasUMTensorVarray<int, TiledArray::Range>;

  auto a = Tile(range);

  std::size_t iter = 0;

  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < A; j++) {
      a[iter] = iter;
      iter++;
    }
  }

  std::vector<int> perm({1, 0});

  TiledArray::Permutation permutation(perm);

  auto b = permute(a, permutation);

  cudaDeviceSynchronize();
  iter = 0;
  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < A; j++) {
      BOOST_CHECK(b(j, i) == iter);
      iter++;
    }
  }
}

BOOST_AUTO_TEST_CASE(cutt_um_tensor_nonsym) {
  TiledArray::Range range{B, A};

  using Tile = TiledArray::btasUMTensorVarray<int, TiledArray::Range>;

  auto a = Tile(range);

  std::size_t iter = 0;

  for (std::size_t i = 0; i < B; i++) {
    for (std::size_t j = 0; j < A; j++) {
      a[iter] = iter;
      iter++;
    }
  }

  std::vector<int> perm({1, 0});

  TiledArray::Permutation permutation(perm);

  auto b = permute(a, permutation);

  cudaDeviceSynchronize();
  iter = 0;
  for (std::size_t i = 0; i < B; i++) {
    for (std::size_t j = 0; j < A; j++) {
      BOOST_CHECK(b(j, i) == iter);
      iter++;
    }
  }
}

BOOST_AUTO_TEST_CASE(cutt_um_tensor_rank_three) {
  TiledArray::Range range{A, B, C};

  using Tile = TiledArray::btasUMTensorVarray<int, TiledArray::Range>;

  auto a = Tile(range);

  std::size_t iter = 0;

  for (std::size_t i = 0; i < A; i++) {
    for (std::size_t j = 0; j < B; j++) {
      for (std::size_t k = 0; k < C; k++) {
        a[iter] = iter;
        iter++;
      }
    }
  }

  // b(k,i,j) = a(i,j,k)
  {
    TiledArray::Permutation permutation({1, 2, 0});

    auto b = permute(a, permutation);

    cudaDeviceSynchronize();
    iter = 0;
    for (std::size_t i = 0; i < A; i++) {
      for (std::size_t j = 0; j < B; j++) {
        for (std::size_t k = 0; k < C; k++) {
          BOOST_CHECK(b(k, i, j) == iter);
          iter++;
        }
      }
    }
  }

  // b(j,i,k) = a(i,j,k)
  {
    TiledArray::Permutation permutation({1, 0, 2});

    auto b = permute(a, permutation);

    cudaDeviceSynchronize();
    iter = 0;
    for (std::size_t i = 0; i < A; i++) {
      for (std::size_t j = 0; j < B; j++) {
        for (std::size_t k = 0; k < C; k++) {
          BOOST_CHECK(b(j, i, k) == iter);
          iter++;
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(cutt_um_tensor_rank_four) {
  std::size_t a = 2;
  std::size_t b = 3;
  std::size_t c = 6;
  std::size_t d = 4;

  TiledArray::Range range(std::vector<std::size_t>({a, b, c, d}));

  using Tile = TiledArray::btasUMTensorVarray<int, TiledArray::Range>;

  auto tile_a = Tile(range);

  std::size_t iter = 0;

  // initialize tensor
  for (std::size_t i = 0; i < a; i++) {
    for (std::size_t j = 0; j < b; j++) {
      for (std::size_t k = 0; k < c; k++) {
        for (std::size_t l = 0; l < d; l++) {
          tile_a[iter] = iter;
          iter++;
        }
      }
    }
  }

  // b(i,l,k,j) = a(i,j,k,l)
  {
    TiledArray::Permutation permutation({0, 3, 2, 1});

    auto tile_b = permute(tile_a, permutation);

    cudaDeviceSynchronize();
    // validate
    iter = 0;
    for (std::size_t i = 0; i < a; i++) {
      for (std::size_t j = 0; j < b; j++) {
        for (std::size_t k = 0; k < c; k++) {
          for (std::size_t l = 0; l < d; l++) {
            BOOST_CHECK_EQUAL(tile_b(i, l, k, j), iter);
            iter++;
          }
        }
      }
    }
  }

  // b(j, i, l, k) = a(i, j, k, l)
  {
    TiledArray::Permutation permutation({1, 0, 3, 2});

    auto tile_b = permute(tile_a, permutation);

    cudaDeviceSynchronize();
    // validate
    iter = 0;
    for (std::size_t i = 0; i < a; i++) {
      for (std::size_t j = 0; j < b; j++) {
        for (std::size_t k = 0; k < c; k++) {
          for (std::size_t l = 0; l < d; l++) {
            BOOST_CHECK_EQUAL(tile_b(j, i, l, k), iter);
            iter++;
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(cutt_um_tensor_rank_six) {
  std::size_t a = 2;
  std::size_t b = 3;
  std::size_t c = 6;
  std::size_t d = 4;
  std::size_t e = 5;
  std::size_t f = 7;

  TiledArray::Range range(std::vector<std::size_t>({a, b, c, d, e, f}));

  using Tile = TiledArray::btasUMTensorVarray<int, TiledArray::Range>;

  auto tile_a = Tile(range);

  std::size_t iter = 0;

  // initialize tensor
  for (std::size_t i = 0; i < a; i++) {
    for (std::size_t j = 0; j < b; j++) {
      for (std::size_t k = 0; k < c; k++) {
        for (std::size_t l = 0; l < d; l++) {
          for (std::size_t m = 0; m < e; m++) {
            for (std::size_t n = 0; n < f; n++) {
              tile_a[iter] = iter;
              iter++;
            }
          }
        }
      }
    }
  }

  // b(i,j,k,l,m,n) = a(i,l,k,j,n,m)
  {
    TiledArray::Permutation permutation({0, 3, 2, 1, 5, 4});

    auto tile_b = permute(tile_a, permutation);

    cudaDeviceSynchronize();
    // validate
    iter = 0;
    for (std::size_t i = 0; i < a; i++) {
      for (std::size_t j = 0; j < b; j++) {
        for (std::size_t k = 0; k < c; k++) {
          for (std::size_t l = 0; l < d; l++) {
            for (std::size_t m = 0; m < e; m++) {
              for (std::size_t n = 0; n < f; n++) {
                BOOST_CHECK_EQUAL(tile_b(i, l, k, j, n, m), iter);
                iter++;
              }
            }
          }
        }
      }
    }
  }

  // b(j,i,m,l,k,n) = a(i,j,k,l,m,n)
  {
    TiledArray::Permutation permutation({1, 0, 4, 3, 2, 5});

    auto tile_b = permute(tile_a, permutation);

    cudaDeviceSynchronize();
    // validate
    iter = 0;
    for (std::size_t i = 0; i < a; i++) {
      for (std::size_t j = 0; j < b; j++) {
        for (std::size_t k = 0; k < c; k++) {
          for (std::size_t l = 0; l < d; l++) {
            for (std::size_t m = 0; m < e; m++) {
              for (std::size_t n = 0; n < f; n++) {
                BOOST_CHECK_EQUAL(tile_b(j, i, m, l, k, n), iter);
                iter++;
              }
            }
          }
        }
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
#endif
