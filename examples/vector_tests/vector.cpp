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

#include <madness/world/timers.h>
#include <algorithm>
#include <iostream>
#include "TiledArray/error.h"
#include "TiledArray/initialize.h"
#include "TiledArray/math/blas.h"
#include "TiledArray/math/vector_op.h"

#define EIGEN_NO_MALLOC

int main(int argc, char** argv) {
  auto& world = TiledArray::initialize(argc, argv);

  const std::size_t repeat = 100;
  // Allocate some memory for tests
  const std::size_t n = 10000000;
  double* a = NULL;
  double* b = NULL;
  double* c = NULL;
  if (posix_memalign(reinterpret_cast<void**>(&a), 128, sizeof(double) * n) !=
      0)
    return 1;
  if (posix_memalign(reinterpret_cast<void**>(&b), 128, sizeof(double) * n) !=
      0)
    return 1;
  if (posix_memalign(reinterpret_cast<void**>(&c), 128, sizeof(double) * n) !=
      0)
    return 1;

  // Init memory
  std::fill_n(a, n, 2.0);
  std::fill_n(b, n, 3.0);
  std::fill_n(c, n, 0.0);

  ////==========================================================================
  std::cout << "Sum:\n";
  double start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i) c[i] = a[i] + b[i];
  double stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      c[i] = a[i] + b[i];
      c[i + 1] = a[i + 1] + b[i + 1];
      c[i + 2] = a[i + 2] + b[i + 2];
      c[i + 3] = a[i + 3] + b[i + 3];
      c[i + 4] = a[i + 4] + b[i + 4];
      c[i + 5] = a[i + 5] + b[i + 5];
      c[i + 6] = a[i + 6] + b[i + 6];
      c[i + 7] = a[i + 7] + b[i + 7];
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> A(a, n);
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> B(b, n);
    Eigen::Map<Eigen::ArrayXd, Eigen::AutoAlign> C(c, n);
    C = A + B;
  }
  stop = madness::wall_time();

  std::cout << "Eigen:  " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op_serial(
        [](const double x, const double y) { return x + y; }, n, c, a, b);
  }
  stop = madness::wall_time();

  std::cout << "vector serial: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(
        [](const double x, const double y) { return x + y; }, n, c, a, b);
  }
  stop = madness::wall_time();

  std::cout << "vector parallel: " << stop - start << " s\n";

  ////==========================================================================
  std::cout << "\nScale Sum:\n";
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i) c[i] = (a[i] + b[i]) * 3.0;
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      c[i] = (a[i] + b[i]) * 3.0;
      c[i + 1] = (a[i + 1] + b[i + 1]) * 3.0;
      c[i + 2] = (a[i + 2] + b[i + 2]) * 3.0;
      c[i + 3] = (a[i + 3] + b[i + 3]) * 3.0;
      c[i + 4] = (a[i + 4] + b[i + 4]) * 3.0;
      c[i + 5] = (a[i + 5] + b[i + 5]) * 3.0;
      c[i + 6] = (a[i + 6] + b[i + 6]) * 3.0;
      c[i + 7] = (a[i + 7] + b[i + 7]) * 3.0;
    }
    for (; i < n; ++i) c[i] = (a[i] + b[i]) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> A(a, n);
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> B(b, n);
    Eigen::Map<Eigen::ArrayXd, Eigen::AutoAlign> C(c, n);
    C = (A + B) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "Eigen:  " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op_serial(
        [](const double x, const double y) { return (x + y) * 3.0; }, n, c, a,
        b);
  }
  stop = madness::wall_time();

  std::cout << "vector serial: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(
        [](const double x, const double y) { return (x + y) * 3.0; }, n, c, a,
        b);
  }
  stop = madness::wall_time();

  std::cout << "vector parallel: " << stop - start << " s\n";

  ////==========================================================================
  std::cout << "\nPower Sum:\n";
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i)
      c[i] = (std::pow(a[i], 3) + std::pow(b[i], 3)) * 3.0;
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      c[i] = (std::pow(a[i], 3) + std::pow(b[i], 3)) * 3.0;
      c[i + 1] = (std::pow(a[i + 1], 3) + std::pow(b[i + 1], 3)) * 3.0;
      c[i + 2] = (std::pow(a[i + 2], 3) + std::pow(b[i + 2], 3)) * 3.0;
      c[i + 3] = (std::pow(a[i + 3], 3) + std::pow(b[i + 3], 3)) * 3.0;
      c[i + 4] = (std::pow(a[i + 4], 3) + std::pow(b[i + 4], 3)) * 3.0;
      c[i + 5] = (std::pow(a[i + 5], 3) + std::pow(b[i + 5], 3)) * 3.0;
      c[i + 6] = (std::pow(a[i + 6], 3) + std::pow(b[i + 6], 3)) * 3.0;
      c[i + 7] = (std::pow(a[i + 7], 3) + std::pow(b[i + 7], 3)) * 3.0;
    }
    for (; i < n; ++i) c[i] = (std::pow(a[i], 3) + std::pow(b[i], 3)) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op_serial(
        [](const double x, const double y) {
          return (std::pow(x, 3) + std::pow(y, 3)) * 3.0;
        },
        n, c, a, b);
  }
  stop = madness::wall_time();

  std::cout << "vector serial: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(
        [](const double x, const double y) {
          return (std::pow(x, 3) + std::pow(y, 3)) * 3.0;
        },
        n, c, a, b);
  }
  stop = madness::wall_time();

  std::cout << "vector parallel: " << stop - start << " s\n";

  ////==========================================================================
  std::cout << "\nMultiply:\n";
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i) c[i] = a[i] * b[i];
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      c[i] = a[i] * b[i];
      c[i + 1] = a[i + 1] * b[i + 1];
      c[i + 2] = a[i + 2] * b[i + 2];
      c[i + 3] = a[i + 3] * b[i + 3];
      c[i + 4] = a[i + 4] * b[i + 4];
      c[i + 5] = a[i + 5] * b[i + 5];
      c[i + 6] = a[i + 6] * b[i + 6];
      c[i + 7] = a[i + 7] * b[i + 7];
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> A(a, n);
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> B(b, n);
    Eigen::Map<Eigen::ArrayXd, Eigen::AutoAlign> C(c, n);
    C = A * B;
  }
  stop = madness::wall_time();

  std::cout << "Eigen:  " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op_serial(
        [](const double x, const double y) { return x * y; }, n, c, a, b);
  }
  stop = madness::wall_time();

  std::cout << "vector serial: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(
        [](const double x, const double y) { return x * y; }, n, c, a, b);
  }
  stop = madness::wall_time();

  std::cout << "vector parallel: " << stop - start << " s\n";

  ////==========================================================================
  std::cout << "\nScale Multiply:\n";
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i) c[i] = (a[i] * b[i]) * 3.0;
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      c[i] = (a[i] * b[i]) * 3.0;
      c[i + 1] = (a[i + 1] * b[i + 1]) * 3.0;
      c[i + 2] = (a[i + 2] * b[i + 2]) * 3.0;
      c[i + 3] = (a[i + 3] * b[i + 3]) * 3.0;
      c[i + 4] = (a[i + 4] * b[i + 4]) * 3.0;
      c[i + 5] = (a[i + 5] * b[i + 5]) * 3.0;
      c[i + 6] = (a[i + 6] * b[i + 6]) * 3.0;
      c[i + 7] = (a[i + 7] * b[i + 7]) * 3.0;
    }
    for (; i < n; ++i) c[i] = (a[i] * b[i]) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> A(a, n);
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> B(b, n);
    Eigen::Map<Eigen::ArrayXd, Eigen::AutoAlign> C(c, n);
    C = (A * B) * 3.0;
  }
  stop = madness::wall_time();

  std::cout << "Eigen:  " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op_serial(
        [](const double x, const double y) { return (x * y) * 3.0; }, n, c, a,
        b);
  }
  stop = madness::wall_time();

  std::cout << "vector serial: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::vector_op(
        [](const double x, const double y) { return (x * y) * 3.0; }, n, c, a,
        b);
  }
  stop = madness::wall_time();

  std::cout << "vector parallel: " << stop - start << " s\n";

  ////==========================================================================
  std::cout << "\nScale:\n";
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i) c[i] *= 3.0;
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      c[i] = 3.0;
      c[i + 1] = 3.0;
      c[i + 2] = 3.0;
      c[i + 3] = 3.0;
      c[i + 4] = 3.0;
      c[i + 5] = 3.0;
      c[i + 6] = 3.0;
      c[i + 7] = 3.0;
    }
    for (; i < n; ++i) c[i] = 3.0;
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> A(a, n);
    Eigen::Map<const Eigen::ArrayXd, Eigen::AutoAlign> B(b, n);
    Eigen::Map<Eigen::ArrayXd, Eigen::AutoAlign> C(c, n);
    C *= 3.0;
  }
  stop = madness::wall_time();

  std::cout << "Eigen:  " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::blas::scale(n, 3.0, c);
  }
  stop = madness::wall_time();

  std::cout << "blas:   " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::inplace_vector_op_serial([](double& x) { x *= 3.0; }, n,
                                               c);
  }
  stop = madness::wall_time();

  std::cout << "vector serial: " << stop - start << " s\n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::inplace_vector_op([](double& x) { x *= 3.0; }, n, c);
  }
  stop = madness::wall_time();

  std::cout << "vector parallel: " << stop - start << " s\n";

  ////==========================================================================
  std::cout << "\nMaxabs:\n";
  double temp = 0.0;
  a[10] = 100.5;
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r)
    for (std::size_t i = 0ul; i < n; ++i) temp = std::max(temp, std::abs(a[i]));
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s " << temp << " \n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      temp = std::max(temp, std::abs(a[i]));
      temp = std::max(temp, std::abs(a[i + 1]));
      temp = std::max(temp, std::abs(a[i + 2]));
      temp = std::max(temp, std::abs(a[i + 3]));
      temp = std::max(temp, std::abs(a[i + 4]));
      temp = std::max(temp, std::abs(a[i + 5]));
      temp = std::max(temp, std::abs(a[i + 6]));
      temp = std::max(temp, std::abs(a[i + 7]));
    }
    for (; i < n; ++i) temp = std::max(temp, std::abs(a[i]));
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s " << temp << " \n";

  start = madness::wall_time();
  double x = 0.0;
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::reduce_op_serial(
        [](double& x, const double a) { x = std::max(x, std::abs(a)); }, n, x,
        a);
  }
  stop = madness::wall_time();
  std::cout << "vector serial: " << stop - start << " s " << x << " \n";

  start = madness::wall_time();
  x = 0.0;
  for (std::size_t r = 0ul; r < repeat; ++r) {
    TiledArray::math::reduce_op(
        [](double& x, const double a) { x = std::max(x, std::abs(a)); },
        [](double& x, const double a) { x = std::max(x, a); },
        std::numeric_limits<double>::min(), n, x, a);
  }
  stop = madness::wall_time();
  std::cout << "vector parallel: " << stop - start << " s " << x << " \n";

  ////==========================================================================
  std::cout << "\nReduce Sum:\n";
  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    temp = 0.0;
    for (std::size_t i = 0ul; i < n; ++i) temp += b[i];
  }
  stop = madness::wall_time();

  std::cout << "base:   " << stop - start << " s " << temp << " \n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    temp = 0.0;
    const std::size_t n4 = n - (n % 8);
    std::size_t i = 0ul;
    for (; i < n4; i += 8) {
      temp += b[i];
      temp += b[i + 1];
      temp += b[i + 2];
      temp += b[i + 3];
      temp += b[i + 4];
      temp += b[i + 5];
      temp += b[i + 6];
      temp += b[i + 7];
    }
    for (; i < n; ++i) temp += b[i];
  }
  stop = madness::wall_time();

  std::cout << "unwind: " << stop - start << " s " << temp << " \n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    x = 0.0;
    TiledArray::math::reduce_op_serial(
        [](double& x, const double a) { x += a; }, n, x, b);
  }
  stop = madness::wall_time();
  std::cout << "vector serial: " << stop - start << " s " << x << " \n";

  start = madness::wall_time();
  for (std::size_t r = 0ul; r < repeat; ++r) {
    x = 0.0;
    TiledArray::math::reduce_op([](double& x, const double a) { x += a; },
                                [](double& x, const double a) { x += a; }, 0.0,
                                n, x, b);
  }
  stop = madness::wall_time();
  std::cout << "vector parallel: " << stop - start << " s " << x << " \n";

  // Deallocate memory
  free(a);
  free(b);
  free(c);

  TiledArray::finalize();

  return 0;
}
