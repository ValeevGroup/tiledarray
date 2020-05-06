#include <tiledarray.h>
#include <random>
#include "TiledArray/config.h"
#include "range_fixture.h"
#include "unit_test_config.h"

auto gen_trange1(size_t N, const std::vector<size_t>& TA_NBs) {
  assert(TA_NBs.size() > 0);
  static int seed = 0;

  std::default_random_engine gen(seed++);
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

  return TA::TiledRange1(t_boundaries.begin(), t_boundaries.end());
};

BOOST_AUTO_TEST_SUITE(diagonal_array_suite)

BOOST_AUTO_TEST_CASE(make_constant_diagonal_array) {
  GlobalFixture::world->gop.fence();

  const auto M = 48, N = 32;

  auto trange0 = gen_trange1(M, {7ul, 13ul, 3ul, 11ul});
  auto trange1 = gen_trange1(N, {3ul, 11ul, 7ul, 13ul});
  auto trange = TA::TiledRange({trange0, trange1});

  auto const_ta_array = TA::diagonal_array<TA::TSpArray<double> >(
      *GlobalFixture::world, trange, 2.0);
  // std::cout << const_ta_array << std::endl;

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_CASE(make_diagonal_array) {
  const auto M = 48, N = 32;

  auto trange0 = gen_trange1(M, {7ul, 13ul, 3ul, 11ul});
  auto trange1 = gen_trange1(N, {3ul, 11ul, 7ul, 13ul});
  auto trange = TA::TiledRange({trange0, trange1});

  std::vector<double> v(32, 1.5);
  v[0] = 1.2;
  v[1] = 1.3;
  v[2] = 1.4;
  auto ta_array = TA::diagonal_array<TA::TSpArray<double> >(
      *GlobalFixture::world, trange, v.begin(), v.end());
  // std::cout << ta_array << std::endl;

  GlobalFixture::world->gop.fence();
};

BOOST_AUTO_TEST_SUITE_END()
