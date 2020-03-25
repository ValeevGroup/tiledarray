#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
#include "range_fixture.h"
#include "unit_test_config.h"

struct ScaLAPACKFixture {
  blacspp::Grid grid;
  ScaLAPACKMatrix<double> ref_matrix;  // XXX: Just double is fine?

  static double make_ta_reference(TA::Tensor<double>& t,
                                  TA::Range const& range) {
    t = TA::Tensor<double>(range, 0.0);
    auto lo = range.lobound_data();
    auto up = range.upbound_data();
    for (auto m = lo[0]; m < up[0]; ++m) {
      for (auto n = lo[1]; n < up[1]; ++n) {
        t(m, n) = m + n;
      }
    }

    return t.norm();
  };

  ScaLAPACKFixture(int64_t N, int64_t NB)
      : grid(blacspp::Grid::square_grid(MPI_COMM_WORLD)),  // XXX: Is this safe?
        ref_matrix(*GlobalFixture::world, grid, N, N, NB, NB) {
    // Fill reference matrix (Needs to be deterministic for later checks)
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < N; ++j)
        if (ref_matrix.dist().i_own(i, j)) {
          auto [i_local, j_local] = ref_matrix.dist().local_indx(i, j);
          ref_matrix.local_mat()(i_local, j_local) = i + j;
        }
    std::cout << "HERE" << std::endl;
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

BOOST_AUTO_TEST_CASE(sca_to_uniform_tiled_array_test) {
  GlobalFixture::world->gop.fence();

  auto [M, N] = ref_matrix.dims();
  BOOST_REQUIRE_EQUAL(M, N);

  auto NB = ref_matrix.dist().nb();

  auto trange = gen_trange(N, {static_cast<size_t>(NB)});
  std::cout << trange << std::endl;

  auto ref_ta = TA::make_array<TA::TArray<double> >(
      *GlobalFixture::world, trange, 
      [](TA::Tensor<double>& t, TA::Range const& range) -> double {
        return ScaLAPACKFixture::make_ta_reference(t, range);
      });

  GlobalFixture::world->gop.fence();
  // auto test_ta = ref_matrix.tensor_from_matrix( trange );
  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_SUITE_END()
