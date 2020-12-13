/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  math_blas.cpp
 *  Apr 30, 2014
 *
 */

#include "TiledArray/math/blas.h"
#include "tiledarray.h"
#include "unit_test_config.h"

struct BlasFixture {
  BlasFixture() : m(30), n(50), k(70) {}

  ~BlasFixture() {}

  template <typename T>
  static void rand_fill(T *first, const std::size_t n, const int seed = 23,
                        const T max = 101) {
    GlobalFixture::world->srand(seed);
    for (std::size_t i = 0ul; i < n; ++i)
      first[i] = GlobalFixture::world->rand() % int(max);
  }

  integer m, n, k;
  static const double tol;

};  // BlasFixture

const double BlasFixture::tol = 0.001;

BOOST_FIXTURE_TEST_SUITE(blas_suite, BlasFixture, TA_UT_SKIP_IF_DISTRIBUTED)

typedef boost::mpl::list<int, long, unsigned int, unsigned long> int_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(integral_gemm, T, int_types) {
  // Allocate and initialize test input
  T *a = NULL, *b = NULL, *c = NULL;

  try {
    // Allocate and fill matrices
    a = new T[m * k];
    b = new T[k * n];
    c = new T[m * n];

    rand_fill(a, m * k, 29);
    rand_fill(b, k * n, 47);
    rand_fill(c, m * n, 99);

    const integer lda = k, ldb = n, ldc = n;

    // Test the gemm operation
    BOOST_REQUIRE_NO_THROW(
        TiledArray::math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
                               m, n, k, 3, a, lda, b, ldb, 0, c, ldc));

    for (integer i = 0; i < m; ++i) {
      for (integer j = 0; j < n; ++j) {
        // Compute the expected value
        T expected = 0;
        for (integer x = 0; x < k; ++x)
          expected += a[i * lda + x] * b[x * ldb + j];
        expected *= 3;

        // Check the result against the expected value
        BOOST_CHECK_EQUAL(c[i * ldc + j], expected);
      }
    }

  } catch (...) {
    delete[] a;
    delete[] b;
    delete[] c;

    throw;
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(integral_gemm_ld, T, int_types) {
  // Allocate and initialize test input
  T *a = NULL, *b = NULL, *c = NULL;

  try {
    // Allocate and fill matrices
    a = new T[m * k];
    b = new T[k * n];
    c = new T[m * n];

    rand_fill(a, m * k, 29);
    rand_fill(b, k * n, 47);
    rand_fill(c, m * n, 99);

    integer lda = k, ldb = n, ldc = n;
    m /= 2;
    n /= 2;
    k /= 2;

    // Test the gemm operation
    BOOST_REQUIRE_NO_THROW(
        TiledArray::math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
                               m, n, k, 3, a, lda, b, ldb, 0, c, ldc));

    for (integer i = 0; i < m; ++i) {
      for (integer j = 0; j < n; ++j) {
        // Compute the expected value
        T expected = 0;
        for (integer x = 0; x < k; ++x)
          expected += a[i * lda + x] * b[x * ldb + j];
        expected *= 3;

        // Check the result against the expected value
        BOOST_CHECK_EQUAL(c[i * ldc + j], expected);
      }
    }

  } catch (...) {
    delete[] a;
    delete[] b;
    delete[] c;

    throw;
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

typedef boost::mpl::list<float, double> floating_point_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(floating_point_gemm, T, floating_point_types) {
  // Allocate and initialize test input
  T *a = NULL, *b = NULL, *c = NULL;

  try {
    // Allocate and fill matrices
    a = new T[m * k];
    b = new T[k * n];
    c = new T[m * n];

    rand_fill(a, m * k, 29);
    rand_fill(b, k * n, 47);
    rand_fill(c, m * n, 99);

    const integer lda = k, ldb = n, ldc = n;

    // Test the gemm operation
    BOOST_REQUIRE_NO_THROW(
        TiledArray::math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
                               m, n, k, 3, a, lda, b, ldb, 0, c, ldc));
    for (integer i = 0; i < m; ++i) {
      for (integer j = 0; j < n; ++j) {
        // Compute the expected value
        T expected = 0.0;
        for (integer x = 0; x < k; ++x)
          expected += a[i * lda + x] * b[x * ldb + j];
        expected *= 3.0;

        BOOST_CHECK_CLOSE(c[i * ldc + j], expected, tol);
      }
    }

  } catch (...) {
    delete[] a;
    delete[] b;
    delete[] c;

    throw;
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(floating_point_gemm_ld, T, floating_point_types) {
  // Allocate and initialize test input
  T *a = NULL, *b = NULL, *c = NULL;

  try {
    // Allocate and fill matrices
    a = new T[m * k];
    b = new T[k * n];
    c = new T[m * n];

    rand_fill(a, m * k, 29);
    rand_fill(b, k * n, 47);
    rand_fill(c, m * n, 99);

    integer lda = k, ldb = n, ldc = n;
    m /= 2;
    n /= 2;
    k /= 2;

    // Test the gemm operation
    BOOST_REQUIRE_NO_THROW(
        TiledArray::math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
                               m, n, k, 3, a, lda, b, ldb, 0, c, ldc));

    for (integer i = 0; i < m; ++i) {
      for (integer j = 0; j < n; ++j) {
        // Compute the expected value
        T expected = 0.0;
        for (integer x = 0; x < k; ++x)
          expected += a[i * lda + x] * b[x * ldb + j];
        expected *= 3.0;

        // Check the result against the expected value
        BOOST_CHECK_CLOSE(c[i * ldc + j], expected, tol);
      }
    }

  } catch (...) {
    delete[] a;
    delete[] b;
    delete[] c;

    throw;
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(complex_gemm, T, floating_point_types) {
  // Allocate and initialize test input
  std::complex<T> *a = NULL, *b = NULL, *c = NULL;

  try {
    // Allocate and fill matrices
    a = new std::complex<T>[m * k];
    b = new std::complex<T>[k * n];
    c = new std::complex<T>[m * n];

    rand_fill(reinterpret_cast<T *>(a), 2 * m * k, 29);
    rand_fill(reinterpret_cast<T *>(b), 2 * k * n, 47);
    rand_fill(reinterpret_cast<T *>(c), 2 * m * n, 99);

    const integer lda = k, ldb = n, ldc = n;

    // Test the gemm operation
    BOOST_REQUIRE_NO_THROW(
        TiledArray::math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
                               m, n, k, 3, a, lda, b, ldb, 0, c, ldc));

    for (integer i = 0; i < m; ++i) {
      for (integer j = 0; j < n; ++j) {
        // Compute the expected value
        std::complex<T> expected(0.0, 0.0);
        for (integer x = 0; x < k; ++x) {
          expected += a[i * lda + x] * b[x * ldb + j];
        }
        expected *= 3.0;

        // Check the result against the expected value
        BOOST_CHECK_CLOSE(c[i * ldc + j].real(), expected.real(), tol);
        BOOST_CHECK_CLOSE(c[i * ldc + j].imag(), expected.imag(), tol);
      }
    }

  } catch (...) {
    delete[] a;
    delete[] b;
    delete[] c;

    throw;
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(complex_gemm_ld, T, floating_point_types) {
  // Allocate and initialize test input
  std::complex<T> *a = NULL, *b = NULL, *c = NULL;

  try {
    // Allocate and fill matrices
    a = new std::complex<T>[m * k];
    b = new std::complex<T>[k * n];
    c = new std::complex<T>[m * n];

    rand_fill(reinterpret_cast<T *>(a), 2 * m * k, 29);
    rand_fill(reinterpret_cast<T *>(b), 2 * k * n, 47);
    rand_fill(reinterpret_cast<T *>(c), 2 * m * n, 99);

    const integer lda = k, ldb = n, ldc = n;
    m /= 2;
    n /= 2;
    k /= 2;

    // Test the gemm operation
    BOOST_REQUIRE_NO_THROW(
        TiledArray::math::gemm(madness::cblas::NoTrans, madness::cblas::NoTrans,
                               m, n, k, 3, a, lda, b, ldb, 0, c, ldc));

    for (integer i = 0; i < m; ++i) {
      for (integer j = 0; j < n; ++j) {
        // Compute the expected value
        std::complex<T> expected(0.0, 0.0);
        for (integer x = 0; x < k; ++x) {
          expected += a[i * lda + x] * b[x * ldb + j];
        }
        expected *= 3.0;

        // Check the result against the expected value
        BOOST_CHECK_CLOSE(c[i * ldc + j].real(), expected.real(), tol);
        BOOST_CHECK_CLOSE(c[i * ldc + j].imag(), expected.imag(), tol);
      }
    }

  } catch (...) {
    delete[] a;
    delete[] b;
    delete[] c;

    throw;
  }

  delete[] a;
  delete[] b;
  delete[] c;
}

BOOST_AUTO_TEST_SUITE_END()
