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

BOOST_AUTO_TEST_SUITE_END()