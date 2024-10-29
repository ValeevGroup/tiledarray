#include <tiledarray.h>
#include "TiledArray/config.h"
#include "range_fixture.h"
#include "unit_test_config.h"

BOOST_AUTO_TEST_SUITE(retile_suite)

BOOST_AUTO_TEST_CASE(retile_tensor) {
    TA::detail::matrix_il<double> some_values = {
      {0.1, 0.2, 0.3, 0.4, 0.5},
      {0.6, 0.7, 0.8, 0.9, 1.0},
      {1.1, 1.2, 1.3, 1.4, 1.5},
      {1.6, 1.7, 1.8, 1.9, 2.0},
      {2.1, 2.2, 2.3, 2.4, 2.5}
    };

    auto range0 = TA::TiledRange1(0, 3, 5);
    auto range1 = TA::TiledRange1(0, 4, 5);
    auto trange = TA::TiledRange({range0, range1});

    TA::TArrayD default_dense(*GlobalFixture::world, some_values);
    TA::TSpArrayD default_sparse(*GlobalFixture::world, some_values);

    auto result_dense = retile(default_dense, trange);
    auto result_sparse = retile(default_sparse, trange);

    BOOST_CHECK_EQUAL(result_dense.trange(), trange);
    BOOST_CHECK_EQUAL(result_sparse.trange(), trange);
}

BOOST_AUTO_TEST_CASE(retile_more) {
  using Numeric = int;
  using T = TA::Tensor<Numeric>;
  using ToT = TA::Tensor<T>;
  using ArrayT = TA::DistArray<T, TA::SparsePolicy>;
  using ArrayToT = TA::DistArray<ToT, TA::SparsePolicy>;

  auto& world = TA::get_default_world();

  auto const tr_source = TA::TiledRange({{0, 2, 4, 8}, {0, 3, 5}});
  auto const tr_target = TA::TiledRange({{0, 4, 6, 8}, {0, 2, 4, 5}});
  auto const& elem_rng = tr_source.elements_range();

  BOOST_REQUIRE(elem_rng.volume() == tr_target.elements_range().volume());

  auto const inner_rng = TA::Range({3, 3});

  auto rand_tensor = [](auto const& rng) -> T {
    return T(rng, [](auto&&) {
      return TA::detail::MakeRandom<Numeric>::generate_value();
    });
  };

  auto set_random_tensor_tile = [rand_tensor](auto& tile, auto const& rng) {
    tile = rand_tensor(rng);
    return tile.norm();
  };

  auto rand_tensor_of_tensor = [rand_tensor,
                                inner_rng](auto const& rng) -> ToT {
    return ToT(rng, [rand_tensor, inner_rng](auto&&) {
      return rand_tensor(inner_rng);
    });
  };

  auto set_random_tensor_of_tensor_tile = [rand_tensor_of_tensor](
                                              auto& tile, auto const& rng) {
    tile = rand_tensor_of_tensor(rng);
    return tile.norm();
  };

  auto get_elem = [](auto const& arr, auto const& eix) {
    auto tix = arr.trange().element_to_tile(eix);
    auto&& tile = arr.find(tix).get(false);
    return tile(eix);
  };

  auto arr_source0 =
      TA::make_array<ArrayT>(world, tr_source, set_random_tensor_tile);
  auto arr_target0 = TA::retile(arr_source0, tr_target);

  for (auto&& eix : elem_rng) {
    BOOST_REQUIRE(get_elem(arr_source0, eix) == get_elem(arr_target0, eix));
  }

  auto arr_source = TA::make_array<ArrayToT>(world, tr_source,
                                             set_random_tensor_of_tensor_tile);
  auto arr_target = TA::retile(arr_source, tr_target);

  arr_source.make_replicated();
  arr_target.make_replicated();
  arr_source.truncate();
  arr_target.truncate();
  world.gop.fence();

  for (auto&& eix : elem_rng) {
    BOOST_REQUIRE(get_elem(arr_source, eix) == get_elem(arr_target, eix));
  }
}

BOOST_AUTO_TEST_SUITE_END()