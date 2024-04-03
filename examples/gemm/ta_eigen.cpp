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

#include <TiledArray/util/time.h>
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
  Eigen::MatrixXd a(matrix_size, matrix_size);
  Eigen::MatrixXd b(matrix_size, matrix_size);
  Eigen::MatrixXd c(matrix_size, matrix_size);
  a.fill(1.0);
  b.fill(1.0);
  c.fill(0.0);

  // Do matrix multiplication
  for (int i = 0; i < repeat; ++i) {
    TA_RECORD_DURATION(c.noalias() = 1.0 * a * b + 0.0 * c);
  }

  auto durations = TiledArray::duration_statistics();
  std::cout << "Average wall time = " << durations.mean << "\nAverage GFLOPS = "
            << (2.0 * double(matrix_size * matrix_size * matrix_size) / 1.0e9) *
                   durations.mean_reciprocal
            << std::endl;

  return 0;
}
