#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
#include "range_fixture.h"
#include "unit_test_config.h"

#include "TiledArray/algebra/lapack/chol.h"
#include "TiledArray/algebra/lapack/heig.h"
#include "TiledArray/algebra/scalapack/all.h"

using namespace TiledArray::scalapack;

struct ScaLAPACKFixture {
  blacspp::Grid grid;
  BlockCyclicMatrix<double> ref_matrix;  // XXX: Just double is fine?

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

  inline void construct_scalapack(BlockCyclicMatrix<double>& A) {
    auto [M, N] = A.dims();
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        if (A.dist().i_own(i, j)) {
          auto [i_local, j_local] = A.dist().local_indx(i, j);
          A.local_mat()(i_local, j_local) = matrix_element_generator(i, j);
        }
  }

  ScaLAPACKFixture(int64_t N, int64_t NB)
      : grid(blacspp::Grid::square_grid(MPI_COMM_WORLD)),  // XXX: Is this safe?
        ref_matrix(*GlobalFixture::world, grid, N, N, NB, NB),
        htoeplitz_vector(N),
        exact_evals(N) {
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

    // Fill reference matrix
    construct_scalapack(ref_matrix);
  }

  ScaLAPACKFixture() : ScaLAPACKFixture(1000, 128) {}
};

TA::TiledRange gen_trange(size_t N, const std::vector<size_t>& TA_NBs) {
  assert(TA_NBs.size() > 0);

  std::default_random_engine gen(0);
  std::uniform_int_distribution<> dist(0, TA_NBs.size() - 1);
  auto rand_indx = [&]() { return dist(gen); };
  auto rand_nb = [&]() { return TA_NBs[rand_indx()]; };

  std::vector<size_t> t_boundaries = {0};
  auto TA_NB = rand_nb();
  while (t_boundaries.back() + TA_NB < N) {
    t_boundaries.emplace_back(t_boundaries.back() + TA_NB);
    TA_NB = rand_nb();
  }
  t_boundaries.emplace_back(N);

  std::vector<TA::TiledRange1> ranges(
      2, TA::TiledRange1(t_boundaries.begin(), t_boundaries.end()));

  return TA::TiledRange(ranges.begin(), ranges.end());
};

BOOST_FIXTURE_TEST_SUITE(scalapack_suite, ScaLAPACKFixture)

BOOST_AUTO_TEST_CASE(bc_to_uniform_dense_tiled_array_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_ta = block_cyclic_to_array<TA::TArray<double>>(ref_matrix, trange);
  GlobalFixture::world->gop.fence();

  auto norm_diff =
      (ref_ta("i,j") - test_ta("i,j")).norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(bc_to_uniform_dense_tiled_array_all_small_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB / 2)});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_ta = block_cyclic_to_array<TA::TArray<double>>(ref_matrix, trange);
  GlobalFixture::world->gop.fence();

  auto norm_diff =
      (ref_ta("i,j") - test_ta("i,j")).norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(uniform_dense_tiled_array_to_bc_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_matrix = array_to_block_cyclic(ref_ta, grid, NB, NB);
  GlobalFixture::world->gop.fence();

  double local_norm_diff =
      (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
  local_norm_diff *= local_norm_diff;

  double norm_diff;
  MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  norm_diff = std::sqrt(norm_diff);

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(bc_to_random_dense_tiled_array_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  [[maybe_unused]] auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_ta = block_cyclic_to_array<TA::TArray<double>>(ref_matrix, trange);
  GlobalFixture::world->gop.fence();

  auto norm_diff =
      (ref_ta("i,j") - test_ta("i,j")).norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(random_dense_tiled_array_to_bc_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {107ul, 113ul, 211ul, 151ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_matrix = array_to_block_cyclic(ref_ta, grid, NB, NB);
  GlobalFixture::world->gop.fence();

  double local_norm_diff =
      (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
  local_norm_diff *= local_norm_diff;

  double norm_diff;
  MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  norm_diff = std::sqrt(norm_diff);

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(bc_to_sparse_tiled_array_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});

  auto ref_ta = TA::make_array<TA::TSpArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_ta =
      block_cyclic_to_array<TA::TSpArray<double>>(ref_matrix, trange);
  GlobalFixture::world->gop.fence();

  auto norm_diff =
      (ref_ta("i,j") - test_ta("i,j")).norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(sparse_tiled_array_to_bc_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});

  auto ref_ta = TA::make_array<TA::TSpArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_matrix = array_to_block_cyclic(ref_ta, grid, NB, NB);
  GlobalFixture::world->gop.fence();

  double local_norm_diff =
      (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
  local_norm_diff *= local_norm_diff;

  double norm_diff;
  MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  norm_diff = std::sqrt(norm_diff);

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(const_tiled_array_to_bc_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});

  const TA::TArray<double> ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto test_matrix = array_to_block_cyclic(ref_ta, grid, NB, NB);
  GlobalFixture::world->gop.fence();

  double local_norm_diff =
      (test_matrix.local_mat() - ref_matrix.local_mat()).norm();
  local_norm_diff *= local_norm_diff;

  double norm_diff;
  MPI_Allreduce(&local_norm_diff, &norm_diff, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  norm_diff = std::sqrt(norm_diff);

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(sca_heig_same_tiling) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  const auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [evals, evecs] = heig(ref_ta);
  auto [evals_lapack, evecs_lapack] = lapack::heig(ref_ta);
  // auto evals = heig( ref_ta );

  BOOST_CHECK(evecs.trange() == ref_ta.trange());

  // check eigenvectors against lapack only, for now ...
  decltype(evecs) evecs_error;
  evecs_error("i,j") = evecs_lapack("i,j") - evecs("i,j");
  // TODO need to fix phases of the eigenvectors to be able to compare ...
  // BOOST_CHECK_SMALL(evecs_error("i,j").norm().get(),
  //                  N * N * std::numeric_limits<double>::epsilon());

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i) {
    BOOST_CHECK_SMALL(std::abs(evals[i] - exact_evals[i]), tol);
    BOOST_CHECK_SMALL(std::abs(evals_lapack[i] - exact_evals[i]), tol);
  }

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_heig_diff_tiling) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);
  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto new_trange = gen_trange(N, {64ul});
  auto [evals, evecs] = heig(ref_ta, new_trange, 128);
  auto [evals_lapack, evecs_lapack] = lapack::heig(ref_ta, new_trange);

  BOOST_CHECK(evecs.trange() == new_trange);

  // check eigenvectors against lapack only, for now ...
  decltype(evecs) evecs_error;
  evecs_error("i,j") = evecs_lapack("i,j") - evecs("i,j");
  // TODO need to fix phases of the eigenvectors to be able to compare ...
  // BOOST_CHECK_SMALL(evecs_error("i,j").norm().get(),
  //                  N * N * std::numeric_limits<double>::epsilon());

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i) {
    BOOST_CHECK_SMALL(std::abs(evals[i] - exact_evals[i]), tol);
    BOOST_CHECK_SMALL(std::abs(evals_lapack[i] - exact_evals[i]), tol);
  }

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_heig_generalized) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TSpArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto dense_iden = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [](TA::Tensor<double>& t, TA::Range const& range) -> double {
        t = TA::Tensor<double>(range, 0.0);
        auto lo = range.lobound_data();
        auto up = range.upbound_data();
        for (auto m = lo[0]; m < up[0]; ++m)
          for (auto n = lo[1]; n < up[1]; ++n)
            if (m == n) t(m, n) = 1.;

        return t.norm();
      });

  GlobalFixture::world->gop.fence();
  auto [evals, evecs] = heig(ref_ta, dense_iden);
  // auto evals = heig( ref_ta );

  BOOST_CHECK(evecs.trange() == ref_ta.trange());

  // TODO: Check validity of eigenvectors, not crucial for the time being

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(evals[i] - exact_evals[i]), tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_chol) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto L = cholesky(A);

  BOOST_CHECK(L.trange() == A.trange());

  decltype(A) A_minus_LLt;
  A_minus_LLt("i,j") = A("i,j") - L("i,k") * L("j,k").conj();

  BOOST_CHECK_SMALL(A_minus_LLt("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  // check against LAPACK also
  auto L_ref = TiledArray::lapack::cholesky(A);
  decltype(L) L_diff;
  L_diff("i,j") = L("i,j") - L_ref("i,j");

  BOOST_CHECK_SMALL(L_diff("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_chol_linv) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto Linv = cholesky_linv(A);
  auto Linv_lapack = lapack::cholesky_linv(A);

  BOOST_CHECK(Linv.trange() == A.trange());

  TA::TArray<double> tmp(*GlobalFixture::world, trange);
  tmp("i,j") = Linv("i,k") * A("k,j");
  A("i,j") = tmp("i,k") * Linv("j,k");

  TA::foreach_inplace(A, [](TA::Tensor<double>& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) {
          tile(m, n) -= 1.;
        }
  });

  double norm = A("i,j").norm().get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  // test against LAPACK
  decltype(Linv) Linv_error;
  Linv_error("i,j") = Linv("i,j") - Linv_lapack("i,j");
  BOOST_CHECK_SMALL(Linv_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_chol_linv_retl) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [L, Linv] = cholesky_linv<decltype(A), true>(A);
  auto [L_lapack, Linv_lapack] = lapack::cholesky_linv<decltype(A), true>(A);

  BOOST_CHECK(Linv.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  TA::TArray<double> tmp(*GlobalFixture::world, trange);
  tmp("i,j") = Linv("i,k") * L("k,j");

  TA::foreach_inplace(tmp, [](TA::Tensor<double>& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) {
          tile(m, n) -= 1.;
        }
  });

  double norm = tmp("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  // test against LAPACK
  decltype(L) L_error;
  L_error("i,j") = L("i,j") - L_lapack("i,j");
  BOOST_CHECK_SMALL(L_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());
  decltype(Linv) Linv_error;
  Linv_error("i,j") = Linv("i,j") - Linv_lapack("i,j");
  BOOST_CHECK_SMALL(Linv_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_chol_solve) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto iden = cholesky_solve(A, A);
  BOOST_CHECK(iden.trange() == A.trange());

  auto iden_lapack = lapack::cholesky_solve(A, A);
  decltype(iden) iden_error;
  iden_error("i,j") = iden("i,j") - iden_lapack("i,j");
  BOOST_CHECK_SMALL(iden_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  TA::foreach_inplace(iden, [](TA::Tensor<double>& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) {
          tile(m, n) -= 1.;
        }
  });

  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_chol_lsolve) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  // Should produce X = L**H
  auto [L, X] = cholesky_lsolve(TransposeFlag::NoTranspose, A, A);
  BOOST_CHECK(X.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  // first, test against LAPACK
  auto [L_lapack, X_lapack] =
      lapack::cholesky_lsolve(TransposeFlag::NoTranspose, A, A);
  decltype(L) L_error;
  L_error("i,j") = L("i,j") - L_lapack("i,j");
  BOOST_CHECK_SMALL(L_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());
  decltype(X) X_error;
  X_error("i,j") = X("i,j") - X_lapack("i,j");
  BOOST_CHECK_SMALL(X_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  X("i,j") -= L("j,i");

  double norm = X("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_lu_solve) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto iden = lu_solve(ref_ta, ref_ta);

  BOOST_CHECK(iden.trange() == ref_ta.trange());

  TA::foreach_inplace(iden, [](TA::Tensor<double>& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) {
          tile(m, n) -= 1.;
        }
  });

  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_lu_inv) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  TA::TArray<double> iden(*GlobalFixture::world, trange);

  auto Ainv = lu_inv(ref_ta);
  iden("i,j") = Ainv("i,k") * ref_ta("k,j");

  BOOST_CHECK(iden.trange() == ref_ta.trange());

  TA::foreach_inplace(iden, [](TA::Tensor<double>& tile) {
    auto range = tile.range();
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m)
      for (auto n = lo[1]; n < up[1]; ++n)
        if (m == n) {
          tile(m, n) -= 1.;
        }
  });

  double norm = iden("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

#if 1
BOOST_AUTO_TEST_CASE(sca_svd_values_only) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto S = svd<SVDValuesOnly>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_svd_leftvectors) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S, U] = svd<SVDLeftVectors>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_svd_rightvectors) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S, VT] = svd<SVDRightVectors>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(sca_svd_allvectors) {
  GlobalFixture::world->gop.fence();
  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S, U, VT] = svd<SVDAllVectors>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}
#endif

BOOST_AUTO_TEST_SUITE_END()
