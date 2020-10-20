#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
#include "range_fixture.h"
#include "unit_test_config.h"

#include "TiledArray/algebra/lapack/chol.h"
#include "TiledArray/algebra/lapack/heig.h"

using namespace TiledArray::lapack;

struct LAPACKFixture {
  int64_t N;
  std::vector<double> htoeplitz_vector;
  std::vector<double> exact_evals;

  inline double matrix_element_generator(int64_t i, int64_t j) {
    // Generates a Circulant matrix: good condition number
    return htoeplitz_vector[std::abs(i - j)];
  }

  inline double make_ta_reference(TA::Tensor<double>& t,
                                  TA::Range const& range) {
    t = TA::Tensor<double>(range, 0.0);
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m) {
      for (auto n = lo[1]; n < up[1]; ++n) {
        t(m, n) = matrix_element_generator(m, n);
      }
    }

    return t.norm();
  };

  LAPACKFixture(int64_t N) : N(N), htoeplitz_vector(N), exact_evals(N) {
    // Generate an hermitian Circulant vector
    std::fill(htoeplitz_vector.begin(), htoeplitz_vector.begin(), 0);
    htoeplitz_vector[0] = 100;
    std::default_random_engine gen(0);
    std::uniform_real_distribution<> dist(0., 1.);
    for (int64_t i = 1; i <= (N / 2); ++i) {
      double val = dist(gen);
      htoeplitz_vector[i] = val;
      htoeplitz_vector[N - i] = val;
    }

    // Compute exact eigenvalues
    const double ff = 2. * M_PI / N;
    for (int64_t j = 0; j < N; ++j) {
      double val = htoeplitz_vector[0];
      for (int64_t k = 1; k < N; ++k)
        val += htoeplitz_vector[N - k] * std::cos(ff * j * k);
      exact_evals[j] = val;
    }

    std::sort(exact_evals.begin(), exact_evals.end());
  }

  LAPACKFixture() : LAPACKFixture(1000) {}
};

BOOST_FIXTURE_TEST_SUITE(lapack_suite, LAPACKFixture)

BOOST_AUTO_TEST_CASE(chol) {
  auto range = TA::Range{N, N};

  TA::Tensor<double> A;
  this->make_ta_reference(A, range);

  auto L = cholesky(A);

  decltype(A) A_minus_LLt;
  A_minus_LLt = A.clone();
  A_minus_LLt.gemm(L, L, -1,
                   math::GemmHelper{madness::cblas::NoTrans,
                                    madness::cblas::ConjTrans, 2, 2, 2});

  BOOST_CHECK_SMALL(A_minus_LLt.norm(),
                    N * N * std::numeric_limits<double>::epsilon());
}

BOOST_AUTO_TEST_SUITE_END()
