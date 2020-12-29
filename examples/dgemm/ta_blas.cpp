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

#include <tiledarray.h>
#include <iostream>

int main(int argc, char** argv) {
  // Get command line arguments
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " matrix_size [repetitions]\n";
    return 0;
  }
  const long matrix_size = atol(argv[1]);
  if (matrix_size <= 0) {
    std::cerr << "Error: matrix size must be greater than zero.\n";
    return 1;
  }
  const long repeat = (argc >= 3 ? atol(argv[2]) : 5);
  if (repeat <= 0) {
    std::cerr << "Error: number of repetitions must be greater than zero.\n";
    return 1;
  }

  std::cout << "\nMatrix size       = " << matrix_size << "x" << matrix_size
            << "\nMemory per matrix = "
            << double(matrix_size * matrix_size * sizeof(double)) / 1.0e9
            << " GB\n";

  // Construct matrices
  double* a = NULL;
  if (posix_memalign(reinterpret_cast<void**>(&a), 128,
                     sizeof(double) * matrix_size * matrix_size) != 0)
    return 1;
  double* b = NULL;
  if (posix_memalign(reinterpret_cast<void**>(&b), 128,
                     sizeof(double) * matrix_size * matrix_size) != 0)
    return 1;
  double* c = NULL;
  if (posix_memalign(reinterpret_cast<void**>(&c), 128,
                     sizeof(double) * matrix_size * matrix_size) != 0)
    return 1;
  std::fill_n(a, matrix_size * matrix_size, 1.0);
  std::fill_n(b, matrix_size * matrix_size, 1.0);
  std::fill_n(c, matrix_size * matrix_size, 0.0);

  // BLAS dgemm arguments
  using namespace TiledArray::math::blas;
  auto opa = Op::NoTrans, opb = Op::NoTrans;
  const double alpha = 1l, beta = 0l;
  const integer m = matrix_size, n = matrix_size, k = matrix_size;
  const integer lda = matrix_size, ldb = matrix_size, ldc = matrix_size;

  // Start clock
  const double wall_time_start = madness::wall_time();

  // Do matrix multiplcation
  // Note: If TiledArray has not been configured with blas, this will be an
  // eigen call.
  for (int i = 0; i < repeat; ++i) {
    gemm(opa, opb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  // Stop clock
  const double wall_time_stop = madness::wall_time();

  // Cleanup memory
  free(a);
  free(b);
  free(c);

  std::cout << "Average wall time = "
            << (wall_time_stop - wall_time_start) / double(repeat)
            << "\nAverage GFLOPS = "
            << double(repeat) * 2.0 *
                   double(matrix_size * matrix_size * matrix_size) /
                   (wall_time_stop - wall_time_start) / 1.0e9
            << "\n";

  return 0;
}
