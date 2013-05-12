/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  vecor.cpp
 *  May 11, 2013
 *
 */

#define TA_NO_ERROR

#include <iostream>
#include <algorithm>
#include <world/worldtime.h>
#include "TiledArray/math/math.h"
#include "TiledArray/error.h"

#define EIGEN_NO_MALLOC

int main(int argc, char** argv) {
  const std::size_t repeat = 1000;
  // Allocate some memory for tests
  const std::size_t n = 1000000;
  double* a = new double[n];
  double* b = new double[n];
  double* c = new double[n];

  // Init memory
  std::fill_n(a, n, 2.0);
  std::fill_n(b, n, 3.0);
  std::fill_n(c, n, 0.0);

  std::cout << "Sum:\n";
  double start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r)
    for(std::size_t i = 0ul; i < n; ++i)
      c[i] = a[i] + b[i];
  double stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for(; i < n4; i += 8) {
      c[i]   = a[i]   + b[i];
      c[i+1] = a[i+1] + b[i+1];
      c[i+2] = a[i+2] + b[i+2];
      c[i+3] = a[i+3] + b[i+3];
      c[i+4] = a[i+4] + b[i+4];
      c[i+5] = a[i+5] + b[i+5];
      c[i+6] = a[i+6] + b[i+6];
      c[i+7] = a[i+7] + b[i+7];
    }
    for(; i < n; ++i)
      c[i] = a[i] + b[i];
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(n, a, b, c, TiledArray::detail::Plus<double, double, double>());
  }
  stop = madness::wall_time();

  std::cout << "vector: " << stop - start << " s\n";

  std::cout << "\nScale Sum:\n";
  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r)
    for(std::size_t i = 0ul; i < n; ++i)
      c[i] = (a[i] + b[i]) * 3.0;
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for(; i < n4; i += 8) {
      c[i]   = (a[i]   + b[i]) * 3.0;
      c[i+1] = (a[i+1] + b[i+1]) * 3.0;
      c[i+2] = (a[i+2] + b[i+2]) * 3.0;
      c[i+3] = (a[i+3] + b[i+3]) * 3.0;
      c[i+4] = (a[i+4] + b[i+4]) * 3.0;
      c[i+5] = (a[i+5] + b[i+5]) * 3.0;
      c[i+6] = (a[i+6] + b[i+6]) * 3.0;
      c[i+7] = (a[i+7] + b[i+7]) * 3.0;
    }
    for(; i < n; ++i)
      c[i] = (a[i] + b[i]) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(n, a, b, c, TiledArray::detail::ScalPlus<double, double, double>(3.0));
  }
  stop = madness::wall_time();

  std::cout << "vector: " << stop - start << " s\n";

  ////
  std::cout << "\nMultiply:\n";
  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r)
    for(std::size_t i = 0ul; i < n; ++i)
      c[i] = a[i] * b[i];
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for(; i < n4; i += 8) {
      c[i]   = a[i]   * b[i];
      c[i+1] = a[i+1] * b[i+1];
      c[i+2] = a[i+2] * b[i+2];
      c[i+3] = a[i+3] * b[i+3];
      c[i+4] = a[i+4] * b[i+4];
      c[i+5] = a[i+5] * b[i+5];
      c[i+6] = a[i+6] * b[i+6];
      c[i+7] = a[i+7] * b[i+7];
    }
    for(; i < n; ++i)
      c[i] = a[i] + b[i];
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(n, a, b, c, TiledArray::detail::Multiplies<double, double, double>());
  }
  stop = madness::wall_time();

  std::cout << "vector: " << stop - start << " s\n";

  ////
  std::cout << "\nScale Multiply:\n";
  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r)
    for(std::size_t i = 0ul; i < n; ++i)
      c[i] = (a[i] * b[i]) * 3.0;
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for(; i < n4; i += 8) {
      c[i]   = (a[i]   * b[i]) * 3.0;
      c[i+1] = (a[i+1] * b[i+1]) * 3.0;
      c[i+2] = (a[i+2] * b[i+2]) * 3.0;
      c[i+3] = (a[i+3] * b[i+3]) * 3.0;
      c[i+4] = (a[i+4] * b[i+4]) * 3.0;
      c[i+5] = (a[i+5] * b[i+5]) * 3.0;
      c[i+6] = (a[i+6] * b[i+6]) * 3.0;
      c[i+7] = (a[i+7] * b[i+7]) * 3.0;
    }
    for(; i < n; ++i)
      c[i] = (a[i] * b[i]) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(n, a, b, c, TiledArray::detail::ScalMultiplies<double, double, double>(3.0));
  }
  stop = madness::wall_time();

  std::cout << "vector: " << stop - start << " s\n";

  ////
  std::cout << "\nMaxabs:\n";
  double temp = 0.0;
  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r)
    for(std::size_t i = 0ul; i < n; ++i)
      temp = std::max(temp, std::abs(a[i]));
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for(std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for(; i < n4; i += 8) {
      temp = std::max(temp, std::abs(a[i]));
      temp = std::max(temp, std::abs(a[i+1]));
      temp = std::max(temp, std::abs(a[i+2]));
      temp = std::max(temp, std::abs(a[i+3]));
      temp = std::max(temp, std::abs(a[i+4]));
      temp = std::max(temp, std::abs(a[i+5]));
      temp = std::max(temp, std::abs(a[i+6]));
      temp = std::max(temp, std::abs(a[i+7]));
    }
    for(; i < n; ++i)
      temp = std::max(temp, std::abs(a[i]));
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  double x = 0.0;
  for(std::size_t r = 0ul; r < repeat; ++r) {
    x = TiledArray::math::maxabs(n, a);
  }
  stop = madness::wall_time();

  std::cout << "vector: " << stop - start << " s\n";

  // Deallocate memory
  delete [] a;
  delete [] b;
  delete [] c;

  return 0;
}
