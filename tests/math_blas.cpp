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

  using integer = TiledArray::math::blas::integer;
  integer m, n, k;
  static const double tol;

};  // BlasFixture

const double BlasFixture::tol = 0.001;

BOOST_FIXTURE_TEST_SUITE(blas_suite, BlasFixture, TA_UT_LABEL_SERIAL)

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
        TiledArray::math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                                     TiledArray::math::blas::Op::NoTrans, m, n,
                                     k, 3, a, lda, b, ldb, 0, c, ldc));

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
        TiledArray::math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                                     TiledArray::math::blas::Op::NoTrans, m, n,
                                     k, 3, a, lda, b, ldb, 0, c, ldc));

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
        TiledArray::math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                                     TiledArray::math::blas::Op::NoTrans, m, n,
                                     k, 3, a, lda, b, ldb, 0, c, ldc));
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
        TiledArray::math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                                     TiledArray::math::blas::Op::NoTrans, m, n,
                                     k, 3, a, lda, b, ldb, 0, c, ldc));

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
        TiledArray::math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                                     TiledArray::math::blas::Op::NoTrans, m, n,
                                     k, 3, a, lda, b, ldb, 0, c, ldc));

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
        TiledArray::math::blas::gemm(TiledArray::math::blas::Op::NoTrans,
                                     TiledArray::math::blas::Op::NoTrans, m, n,
                                     k, 3, a, lda, b, ldb, 0, c, ldc));

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

BOOST_AUTO_TEST_CASE(measured_ld_fits) {
  // ld_fits() guards the leading dimensions the arena strided kernels
  // (arena_einsum.h) and Tensor::gemm's arena scale paths MEASURE as a
  // distance between two inner-cell addresses: BLAS++ narrows every dimension
  // to blas_int (max_ld()), and a GEMM of nslab rows at leading dimension ld
  // reaches element (nslab - 1) * ld + extent - 1, which must be
  // representable too.
  namespace blas = TiledArray::math::blas;
  const integer cap = blas::max_ld();
  BOOST_CHECK_EQUAL(cap,
                    static_cast<integer>(std::numeric_limits<blas_int>::max()));

  // the ordinary contiguous slab always fits
  BOOST_CHECK(blas::ld_fits(64, 1, 64));
  BOOST_CHECK(blas::ld_fits(64, 1000, 64));
  // degenerate arguments never do
  BOOST_CHECK(!blas::ld_fits(-1, 1, 1));
  BOOST_CHECK(!blas::ld_fits(64, 1, 0));
  // one slab: only the step itself must be representable
  BOOST_CHECK(blas::ld_fits(cap, 1, 1));
  BOOST_CHECK(blas::ld_fits(cap, 1, cap));
  // more slabs: the last addressed element must be representable
  BOOST_CHECK(blas::ld_fits(cap, 2, 1));       // reaches cap
  BOOST_CHECK(!blas::ld_fits(cap, 2, 2));      // reaches cap + 1
  BOOST_CHECK(blas::ld_fits(cap - 1, 2, 2));   // reaches cap
  BOOST_CHECK(!blas::ld_fits(cap - 1, 2, 3));  // reaches cap + 1
  const integer nslab = 1000, extent = 7;
  const integer ld_max = (cap - (extent - 1)) / (nslab - 1);
  BOOST_CHECK(blas::ld_fits(ld_max, nslab, extent));
  BOOST_CHECK(!blas::ld_fits(ld_max + 1, nslab, extent));
  // past the cap outright: only expressible when blas_int is narrower than
  // integer (LP64); under ILP64 the cap is integer's own maximum
  if (cap < std::numeric_limits<integer>::max()) {
    BOOST_CHECK(!blas::ld_fits(cap + 1, 1, 1));
    // the motivating case: a 2-cell run whose measured stride is the
    // distance between two individually allocated inner tensors
    BOOST_CHECK(!blas::ld_fits(cap + 1, 2, 64));
    BOOST_CHECK(!blas::ld_fits(1, 1, cap + 2));
  }
}

BOOST_AUTO_TEST_SUITE_END()
