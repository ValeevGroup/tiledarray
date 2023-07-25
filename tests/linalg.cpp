#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
//#include "range_fixture.h"
#include "unit_test_config.h"

#include "TiledArray/math/linalg/non-distributed/cholesky.h"
#include "TiledArray/math/linalg/non-distributed/heig.h"
#include "TiledArray/math/linalg/non-distributed/lu.h"
#include "TiledArray/math/linalg/non-distributed/svd.h"

#include "TiledArray/math/linalg/cholesky.h"
#include "TiledArray/math/linalg/heig.h"
#include "TiledArray/math/linalg/lu.h"
#include "TiledArray/math/linalg/svd.h"

namespace TA = TiledArray;
namespace non_dist = TA::math::linalg::non_distributed;

#if TILEDARRAY_HAS_SCALAPACK
namespace scalapack = TA::math::linalg::scalapack;
#include "TiledArray/math/linalg/scalapack/all.h"
#define TILEDARRAY_SCALAPACK_TEST(F, E)                           \
  GlobalFixture::world->gop.fence();                              \
  compare("TiledArray::scalapack", non_dist::F, scalapack::F, E); \
  GlobalFixture::world->gop.fence();                              \
  compare("TiledArray", non_dist::F, TiledArray::F, E);
#else
#define TILEDARRAY_SCALAPACK_TEST(...)
#endif

#if TILEDARRAY_HAS_SLATE
#include <TiledArray/conversions/slate.h>
#include <TiledArray/math/linalg/slate/cholesky.h>
#include <TiledArray/math/linalg/slate/lu.h>
namespace slate_la = TA::math::linalg::slate;
#define TILEDARRAY_SLATE_TEST(F, E)                           \
  GlobalFixture::world->gop.fence();                              \
  compare("TiledArray::slate", non_dist::F, slate_la::F, E); \
  GlobalFixture::world->gop.fence();                              \
  compare("TiledArray", non_dist::F, TiledArray::F, E);
#else
#define TILEDARRAY_SLATE_TEST(...)
#endif

#if TILEDARRAY_HAS_TTG
#include "TiledArray/math/linalg/ttg/cholesky.h"
#define TILEDARRAY_TTG_TEST(F, E)    \
  GlobalFixture::world->gop.fence(); \
  compare("TiledArray::ttg", non_dist::F, TiledArray::math::linalg::ttg::F, E);
#else
#define TILEDARRAY_TTG_TEST(...)
#endif

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
  inline double make_ta_reference(Tile& t, TA::Range const& range) {
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
};

struct LinearAlgebraFixture : ReferenceFixture {
#if TILEDARRAY_HAS_SCALAPACK

  blacspp::Grid grid;
  scalapack::BlockCyclicMatrix<double> ref_matrix;  // XXX: Just double is fine?

  LinearAlgebraFixture(int64_t N = 1000, int64_t NB = 128)
      : ReferenceFixture(N),
        grid(blacspp::Grid::square_grid(MPI_COMM_WORLD)),  // XXX: Is this safe?
        ref_matrix(*GlobalFixture::world, grid, N, N, NB, NB) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        if (ref_matrix.dist().i_own(i, j)) {
          auto [i_local, j_local] = ref_matrix.dist().local_indx(i, j);
          ref_matrix.local_mat()(i_local, j_local) =
              matrix_element_generator(i, j);
        }
      }
    }
  }
#endif

#if TILEDARRAY_HAS_SLATE

  LinearAlgebraFixture(int64_t N = 1000) : ReferenceFixture(N) {}

  slate::Matrix<double> make_ref_slate(int64_t N, TA::SlateFunctors& slate_functors,
    MPI_Comm comm) {

    slate::Matrix<double> A(N, N, slate_functors.tileMb(), slate_functors.tileNb(), 
      slate_functors.tileRank(), slate_functors.tileDevice(), comm);
    
    A.insertLocalTiles();
    int64_t j_off = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {

      int64_t i_off = 0;
      for (int64_t i = 0; i < A.mt(); ++i) {

        if(A.tileIsLocal(i,j)) {
          auto T = A(i,j);
          for(auto jj = 0; jj < T.nb(); ++jj)
          for(auto ii = 0; ii < T.mb(); ++ii) {
            T.at(ii,jj) = matrix_element_generator(i_off+ii,j_off+jj);
          }
        }

        i_off += A.tileMbFunc()(i);
      }

      j_off += A.tileNbFunc()(j);
    }

    return A;
  }
#endif


  template <class A>
  static void compare(const char* context, const A& non_dist, const A& result,
                      double e) {
    // clang-format off
    BOOST_TEST_CONTEXT(context)
    ;
    // clang-format on
    auto diff_with_non_dist = (non_dist("i,j") - result("i,j")).norm().get();
    BOOST_CHECK_SMALL(diff_with_non_dist, e);
  }

  template <typename T, typename F, int... Is>
  static void for_each_pair_of_tuples_impl(T&& t1, T&& t2, F f,
                                           std::integer_sequence<int, Is...>) {
    auto l = {(f(std::get<Is>(t1), std::get<Is>(t2)), 0)...};
  }

  template <typename... Ts, typename F>
  static void for_each_pair_of_tuples(std::tuple<Ts...> const& t1,
                                      std::tuple<Ts...> const& t2, F f) {
    for_each_pair_of_tuples_impl(
        t1, t2, f, std::make_integer_sequence<int, sizeof...(Ts)>());
  }

  template <class... As>
  static void compare(const char* context, const std::tuple<As...>& non_dist,
                      const std::tuple<As...>& result, double e) {
    for_each_pair_of_tuples(non_dist, result, [&](auto& arg1, auto& arg2) {
      compare(context, arg1, arg2, e);
    });
  }
};

TA::TiledRange gen_trange(size_t N, const std::vector<size_t>& TA_NBs) {
  TA_ASSERT(TA_NBs.size() > 0);

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

BOOST_FIXTURE_TEST_SUITE(linear_algebra_suite, LinearAlgebraFixture)

#if TILEDARRAY_HAS_SCALAPACK

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
  auto test_ta =
      scalapack::block_cyclic_to_array<TA::TArray<double>>(ref_matrix, trange);
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
  auto test_ta =
      scalapack::block_cyclic_to_array<TA::TArray<double>>(ref_matrix, trange);
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
  auto test_matrix = scalapack::array_to_block_cyclic(ref_ta, grid, NB, NB);
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
  auto test_ta =
      scalapack::block_cyclic_to_array<TA::TArray<double>>(ref_matrix, trange);
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
  auto test_matrix = scalapack::array_to_block_cyclic(ref_ta, grid, NB, NB);
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

  // test with TA and btas tile
  using typelist_t =
      std::tuple<TA::Tensor<double>, btas::Tensor<double, TA::Range>>;
  typelist_t typevals;

  auto test = [&](const auto& typeval_ref) {
    using Tile = std::decay_t<decltype(typeval_ref)>;
    using Array = TA::DistArray<Tile, TA::SparsePolicy>;

    auto ref_ta = TA::make_array<Array>(
        *GlobalFixture::world, trange,
        [this](Tile& t, TA::Range const& range) -> double {
          return this->make_ta_reference(t, range);
        });

    GlobalFixture::world->gop.fence();
    auto test_ta = scalapack::block_cyclic_to_array<Array>(ref_matrix, trange);
    GlobalFixture::world->gop.fence();

    auto norm_diff =
        (ref_ta("i,j") - test_ta("i,j")).norm(*GlobalFixture::world).get();

    BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

    GlobalFixture::world->gop.fence();
  };

  test(std::get<0>(typevals));
  test(std::get<1>(typevals));
};

BOOST_AUTO_TEST_CASE(sparse_tiled_array_to_bc_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});

  // test with TA and btas tile
  using typelist_t =
      std::tuple<TA::Tensor<double>, btas::Tensor<double, TA::Range>>;
  typelist_t typevals;

  auto test = [&](const auto& typeval_ref) {
    using Tile = std::decay_t<decltype(typeval_ref)>;
    using Array = TA::DistArray<Tile, TA::SparsePolicy>;

    auto ref_ta = TA::make_array<Array>(
        *GlobalFixture::world, trange,
        [this](Tile& t, TA::Range const& range) -> double {
          return this->make_ta_reference(t, range);
        });

    GlobalFixture::world->gop.fence();
    auto test_matrix = scalapack::array_to_block_cyclic(ref_ta, grid, NB, NB);
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

  test(std::get<0>(typevals));
  test(std::get<1>(typevals));
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
  auto test_matrix = scalapack::array_to_block_cyclic(ref_ta, grid, NB, NB);
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

#endif  // TILEDARRAY_HAS_SCALAPACK

#if TILEDARRAY_HAS_SLATE

BOOST_AUTO_TEST_CASE(dense_tiled_array_to_slate_matrix_test) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {static_cast<size_t>(128)});
  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  auto slate_matrix = TA::array_to_slate(ref_ta);
  GlobalFixture::world->gop.fence();

  TA::SlateFunctors slate_functors( trange, ref_ta.pmap() );
  auto ref_slate = this->make_ref_slate(N, slate_functors, MPI_COMM_WORLD);

  slate::add( 1.0, ref_slate, -1.0, slate_matrix );
  auto norm_diff = slate::norm(slate::Norm::Fro, slate_matrix);
  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(slate_matrix_to_dense_tiled_array_test) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {static_cast<size_t>(128)});
  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  TA::SlateFunctors slate_functors( trange, ref_ta.pmap() );
  auto ref_slate = this->make_ref_slate(N, slate_functors, MPI_COMM_WORLD);

  
  GlobalFixture::world->gop.fence();
  auto test_ta = TA::slate_to_array<TA::TArray<double>>(ref_slate, *GlobalFixture::world);
  GlobalFixture::world->gop.fence();

  auto norm_diff =
      (ref_ta("i,j") - test_ta("i,j")).norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm_diff, std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}
#endif // TILEDARRAY_HAS_SLATE

BOOST_AUTO_TEST_CASE(heig_same_tiling) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  const auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [evals, evecs] = non_dist::heig(ref_ta);
  auto [evals_non_dist, evecs_non_dist] = non_dist::heig(ref_ta);
  // auto evals = heig( ref_ta );

  BOOST_CHECK(evecs.trange() == ref_ta.trange());

  // check eigenvectors against non_dist only, for now ...
  decltype(evecs) evecs_error;
  evecs_error("i,j") = evecs_non_dist("i,j") - evecs("i,j");
  // TODO need to fix phases of the eigenvectors to be able to compare ...
  // BOOST_CHECK_SMALL(evecs_error("i,j").norm().get(),
  //                  N * N * std::numeric_limits<double>::epsilon());

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i) {
    BOOST_CHECK_SMALL(std::abs(evals[i] - exact_evals[i]), tol);
    BOOST_CHECK_SMALL(std::abs(evals_non_dist[i] - exact_evals[i]), tol);
  }

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(heig_diff_tiling) {
  GlobalFixture::world->gop.fence();
  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto new_trange = gen_trange(N, {64ul});
  auto [evals, evecs] = non_dist::heig(ref_ta, new_trange);
  auto [evals_non_dist, evecs_non_dist] = non_dist::heig(ref_ta, new_trange);

  BOOST_CHECK(evecs.trange() == new_trange);

  // check eigenvectors against non_dist only, for now ...
  decltype(evecs) evecs_error;
  evecs_error("i,j") = evecs_non_dist("i,j") - evecs("i,j");
  // TODO need to fix phases of the eigenvectors to be able to compare ...
  // BOOST_CHECK_SMALL(evecs_error("i,j").norm().get(),
  //                  N * N * std::numeric_limits<double>::epsilon());

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i) {
    BOOST_CHECK_SMALL(std::abs(evals[i] - exact_evals[i]), tol);
    BOOST_CHECK_SMALL(std::abs(evals_non_dist[i] - exact_evals[i]), tol);
  }

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(heig_generalized) {
  GlobalFixture::world->gop.fence();

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
  auto [evals, evecs] = non_dist::heig(ref_ta, dense_iden);
  // auto evals = heig( ref_ta );

  BOOST_CHECK(evecs.trange() == ref_ta.trange());

  // TODO: Check validity of eigenvectors, not crucial for the time being

  // Check eigenvalue correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(evals[i] - exact_evals[i]), tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(cholesky) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto L = non_dist::cholesky(A);

  BOOST_CHECK(L.trange() == A.trange());

  decltype(A) A_minus_LLt;
  A_minus_LLt("i,j") = A("i,j") - L("i,k") * L("j,k").conj();

  const double epsilon = N * N * std::numeric_limits<double>::epsilon();

  BOOST_CHECK_SMALL(A_minus_LLt("i,j").norm().get(), epsilon);

  // check against NON_DIST also
  auto L_ref = non_dist::cholesky(A);
  decltype(L) L_diff;
  L_diff("i,j") = L("i,j") - L_ref("i,j");

  BOOST_CHECK_SMALL(L_diff("i,j").norm().get(), epsilon);

  TILEDARRAY_SCALAPACK_TEST(cholesky(A), epsilon);
  TILEDARRAY_SLATE_TEST(cholesky(A), epsilon);

  TILEDARRAY_TTG_TEST(cholesky(A), epsilon);
}

BOOST_AUTO_TEST_CASE(cholesky_linv) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });
  decltype(A) Acopy = A.clone();

  auto Linv = TA::cholesky_linv(A);

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

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = A("i,j").norm().get();

  BOOST_CHECK_SMALL(norm, epsilon);

  TILEDARRAY_SCALAPACK_TEST(cholesky_linv<false>(Acopy), epsilon);
  TILEDARRAY_SLATE_TEST(cholesky_linv<false>(Acopy), epsilon);

  TILEDARRAY_TTG_TEST(cholesky_linv<false>(Acopy), epsilon);
}

BOOST_AUTO_TEST_CASE(cholesky_linv_retl) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [L, Linv] = TA::cholesky_linv<true>(A);

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

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = tmp("i,j").norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm, epsilon);

  TILEDARRAY_SCALAPACK_TEST(cholesky_linv<true>(A), epsilon);
  TILEDARRAY_SLATE_TEST(cholesky_linv<true>(A), epsilon);

  TILEDARRAY_TTG_TEST(cholesky_linv<true>(A), epsilon);
}

BOOST_AUTO_TEST_CASE(cholesky_solve) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto iden = non_dist::cholesky_solve(A, A);
  BOOST_CHECK(iden.trange() == A.trange());

  auto iden_non_dist = non_dist::cholesky_solve(A, A);
  decltype(iden) iden_error;
  iden_error("i,j") = iden("i,j") - iden_non_dist("i,j");
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

BOOST_AUTO_TEST_CASE(cholesky_lsolve) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto A = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  // Should produce X = L**H
  auto [L, X] = non_dist::cholesky_lsolve(TA::NoTranspose, A, A);
  BOOST_CHECK(X.trange() == A.trange());
  BOOST_CHECK(L.trange() == A.trange());

  // first, test against NON_DIST
  auto [L_non_dist, X_non_dist] =
      non_dist::cholesky_lsolve(TA::NoTranspose, A, A);
  decltype(L) L_error;
  L_error("i,j") = L("i,j") - L_non_dist("i,j");
  BOOST_CHECK_SMALL(L_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());
  decltype(X) X_error;
  X_error("i,j") = X("i,j") - X_non_dist("i,j");
  BOOST_CHECK_SMALL(X_error("i,j").norm().get(),
                    N * N * std::numeric_limits<double>::epsilon());

  X("i,j") -= L("j,i");

  double norm = X("i,j").norm(*GlobalFixture::world).get();
  BOOST_CHECK_SMALL(norm, N * N * std::numeric_limits<double>::epsilon());

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(lu_solve) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto iden = non_dist::lu_solve(ref_ta, ref_ta);

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

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm, epsilon);
  TILEDARRAY_SCALAPACK_TEST(lu_solve(ref_ta, ref_ta), epsilon);
  TILEDARRAY_SLATE_TEST(lu_solve(ref_ta, ref_ta), epsilon);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(lu_inv) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  TA::TArray<double> iden(*GlobalFixture::world, trange);

  auto Ainv = non_dist::lu_inv(ref_ta);
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

  double epsilon = N * N * std::numeric_limits<double>::epsilon();
  double norm = iden("i,j").norm(*GlobalFixture::world).get();

  BOOST_CHECK_SMALL(norm, epsilon);
  TILEDARRAY_SCALAPACK_TEST(lu_inv(ref_ta), epsilon);

  GlobalFixture::world->gop.fence();
}

#if 1
BOOST_AUTO_TEST_CASE(svd_values_only) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto S = non_dist::svd<TA::SVD::ValuesOnly>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_leftvectors) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S, U] = non_dist::svd<TA::SVD::LeftVectors>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_rightvectors) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S, VT] = non_dist::svd<TA::SVD::RightVectors>(ref_ta, trange, trange);

  std::vector exact_singular_values = exact_evals;
  std::sort(exact_singular_values.begin(), exact_singular_values.end(),
            std::greater<double>());

  // Check singular value correctness
  double tol = N * N * std::numeric_limits<double>::epsilon();
  for (int64_t i = 0; i < N; ++i)
    BOOST_CHECK_SMALL(std::abs(S[i] - exact_singular_values[i]), tol);
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(svd_allvectors) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  auto [S, U, VT] = non_dist::svd<TA::SVD::AllVectors>(ref_ta, trange, trange);

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

template <bool use_scalapack, typename ArrayT>
void householder_qr_q_only_test(const ArrayT& A, double tol) {
  using value_type = typename ArrayT::element_type;

#if TILEDARRAY_HAS_SCALAPACK
  auto Q = use_scalapack ? scalapack::householder_qr<true>(A)
                         : non_dist::householder_qr<true>(A);
#else
  static_assert(not use_scalapack);
  auto Q = non_dist::householder_qr<true>(A);
#endif

  // Make sure the Q is orthogonal at least
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  Iden.make_replicated();
  auto I_eig = TA::array_to_eigen(Iden);
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK_SMALL((I_eig - decltype(I_eig)::Identity(N, N)).norm(), tol);
}

template <bool use_scalapack, typename ArrayT>
void householder_qr_test(const ArrayT& A, double tol) {
#if TILEDARRAY_HAS_SCALAPACK
  auto [Q, R] = use_scalapack ? scalapack::householder_qr<false>(A)
                              : non_dist::householder_qr<false>(A);
#else
  static_assert(not use_scalapack);
  auto [Q, R] = non_dist::householder_qr<false>(A);
#endif

  // Check reconstruction error
  TA::TArray<double> QR_ERROR;
  QR_ERROR("i,j") = A("i,j") - Q("i,k") * R("k,j");
  BOOST_CHECK_SMALL(QR_ERROR("i,j").norm().get(), tol);

  // Check orthonormality of Q
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  Iden.make_replicated();
  auto I_eig = TA::array_to_eigen(Iden);
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK_SMALL((I_eig - decltype(I_eig)::Identity(N, N)).norm(), tol);
}

BOOST_AUTO_TEST_CASE(householder_qr_q_only) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  double tol = N * N * std::numeric_limits<double>::epsilon();
  householder_qr_q_only_test<false>(ref_ta, tol);
#if TILEDARRAY_HAS_SCALAPACK
  householder_qr_q_only_test<true>(ref_ta, tol);
#endif

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(householder_qr) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  double tol = N * N * std::numeric_limits<double>::epsilon();
  householder_qr_test<false>(ref_ta, tol);
#if TILEDARRAY_HAS_SCALAPACK
  householder_qr_test<true>(ref_ta, tol);
#endif

  GlobalFixture::world->gop.fence();
}

template <typename ArrayT>
void cholesky_qr_q_only_test(const ArrayT& A, double tol) {
  using value_type = typename ArrayT::element_type;

  auto Q = TiledArray::math::linalg::cholesky_qr<true>(A);

  // Make sure the Q is orthogonal at least
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  Iden.make_replicated();
  auto I_eig = TA::array_to_eigen(Iden);
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK_SMALL((I_eig - decltype(I_eig)::Identity(N, N)).norm(), tol);
}

template <typename ArrayT>
void cholesky_qr_test(const ArrayT& A, double tol) {
  auto [Q, R] = TiledArray::math::linalg::cholesky_qr<false>(A);

  // Check reconstruction error
  TA::TArray<double> QR_ERROR;
  QR_ERROR("i,j") = A("i,j") - Q("i,k") * R("k,j");
  BOOST_CHECK_SMALL(QR_ERROR("i,j").norm().get(), tol);

  // Check orthonormality of Q
  TA::TArray<double> Iden;
  Iden("i,j") = Q("k,i") * Q("k,j");
  Iden.make_replicated();
  auto I_eig = TA::array_to_eigen(Iden);
  const auto N = A.trange().dim(1).extent();
  BOOST_CHECK_SMALL((I_eig - decltype(I_eig)::Identity(N, N)).norm(), tol);
}

BOOST_AUTO_TEST_CASE(cholesky_qr_q_only) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  double tol = N * N * std::numeric_limits<double>::epsilon();
  cholesky_qr_q_only_test(ref_ta, tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(cholesky_qr) {
  GlobalFixture::world->gop.fence();

  auto trange = gen_trange(N, {128ul});

  auto ref_ta = TA::make_array<TA::TArray<double>>(
      *GlobalFixture::world, trange,
      [this](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return this->make_ta_reference(t, range);
      });

  double tol = N * N * std::numeric_limits<double>::epsilon();
  cholesky_qr_test(ref_ta, tol);

  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_SUITE_END()
