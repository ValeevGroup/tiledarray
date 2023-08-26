#pragma once
#include <tiledarray.h>

template <typename Derived = void>
struct ReferenceFixture {
  size_t N;
  std::vector<double> htoeplitz_vector;
  std::vector<double> exact_evals;

  inline double matrix_element_generator(int64_t i, int64_t j) {
#if 0
    // Generates a Hankel matrix: absurd condition number
    return i+j;
#else
    // Generates a Circulant matrix: good condition number
    return htoeplitz_vector[std::abs(i - j)];
#endif
  }

  template <typename Tile>
  inline auto make_ta_reference(Tile& t, TA::Range const& range) {
    t = Tile(range, 0.0);
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m) {
      for (auto n = lo[1]; n < up[1]; ++n) {
        t(m, n) = matrix_element_generator(m, n);
      }
    }

    return norm(t);
  };

  template <typename Array>
  inline auto generate_ta_reference(TA::World& world, TA::TiledRange trange) {
    return TA::make_array<Array>(world, trange,
      [this](auto& t, TA::Range const& range) -> auto { 
        return this->make_ta_reference(t,range);
      });
  }

  template <typename Tile>
  inline double make_ta_identity(Tile& t, TA::Range const& range) {
    t = Tile(range, 0.0);
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) t(m, n) = 1.;

    return t.norm();
  }

  template <typename Array>
  inline auto generate_ta_identity(TA::World& world, TA::TiledRange trange) {
    return TA::make_array<Array>(world, trange,
      [this](auto& t, TA::Range const& range) -> auto { 
        return this->make_ta_identity(t,range);
      });
  }

  ReferenceFixture(int64_t N = 1000)
      : N(N), htoeplitz_vector(N), exact_evals(N) {
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
      ;
      for (int64_t k = 1; k < N; ++k)
        val += htoeplitz_vector[N - k] * std::cos(ff * j * k);
      exact_evals[j] = val;
    }

    std::sort(exact_evals.begin(), exact_evals.end());
  }


  void heig_same_tiling_test(TA::World& world); 
  void heig_diff_tiling_test(TA::World& world); 
  void heig_generalized_test(TA::World& world); 

  void cholesky_test(TA::World& world); 
  void cholesky_linv_test(TA::World& world); 
  void cholesky_linv_retl_test(TA::World& world); 
  void cholesky_solve_test(TA::World& world); 
  void cholesky_lsolve_test(TA::World& world); 

  void lu_solve_test(TA::World& world); 
  void lu_inv_test(TA::World& world); 

  void svd_values_only_test(TA::World& world);
  void svd_leftvectors_test(TA::World& world);
  void svd_rightvectors_test(TA::World& world);
  void svd_allvectors_test(TA::World& world);

  void householder_qr_q_only_test(TA::World& world);
  void householder_qr_test(TA::World& world);
};

// Macro to generate tests
#define LINALG_TEST_IMPL_MPI_SAFE(NAME, SERIAL_ONLY)    \
BOOST_AUTO_TEST_CASE(NAME) {                            \
  const auto world_size = GlobalFixture::world->size(); \
  if(SERIAL_ONLY and  world_size > 1) return;           \
  NAME##_##test(*GlobalFixture::world);                 \
}

#define LINALG_TEST_IMPL(NAME) LINALG_TEST_IMPL_MPI_SAFE(NAME,false)

