#include <TiledArray/util/allocator.h>
#include "unit_test_config.h"

/*
 */
BOOST_AUTO_TEST_SUITE(allocator_suite)

BOOST_AUTO_TEST_CASE(core) {
  static_assert(
      sizeof(TiledArray::batch_allocator<double, std::allocator<double>, 0>) ==
      sizeof(std::size_t));  // std::allocator is stateless
  static_assert(
      std::is_standard_layout_v<
          TiledArray::batch_allocator<double, std::allocator<double>, 0>>);
  static_assert(
      sizeof(TiledArray::batch_allocator<double, std::allocator<double>, 5>) ==
      sizeof(std::allocator<double>));
  static_assert(
      std::is_standard_layout_v<
          TiledArray::batch_allocator<double, std::allocator<double>, 5>>);

  static_assert(
      !TiledArray::detail::is_batch_allocator_v<std::allocator<double>>);
  static_assert(
      TiledArray::detail::is_batch_allocator_v<
          TiledArray::batch_allocator<double, std::allocator<double>, 0>>);
  static_assert(
      TiledArray::detail::is_batch_allocator_v<
          TiledArray::batch_allocator<double, std::allocator<double>, 5>>);

  static_assert(
      !TiledArray::detail::has_batch_allocator_v<std::vector<double>>);
  static_assert(TiledArray::detail::has_batch_allocator_v<
                std::vector<double, TiledArray::batch_allocator<double>>>);

  BOOST_CHECK_NO_THROW(
      (TiledArray::batch_allocator<double, std::allocator<double>, 0>{}));
  BOOST_CHECK((TiledArray::batch_allocator<double, std::allocator<double>, 0>{}
                   .batch_size() == 1));
  BOOST_CHECK_THROW(
      (TiledArray::batch_allocator<double, std::allocator<double>, 0>{0}),
      TiledArray::Exception);
  BOOST_CHECK_NO_THROW(
      (TiledArray::batch_allocator<double, std::allocator<double>, 0>{5}));
  BOOST_CHECK((TiledArray::batch_allocator<double, std::allocator<double>, 0>{5}
                   .batch_size() == 5));
  BOOST_CHECK_NO_THROW(
      (TiledArray::batch_allocator<double, std::allocator<double>, 5>{}));
  BOOST_CHECK((TiledArray::batch_allocator<double, std::allocator<double>, 5>{}
                   .batch_size() == 5));
}

BOOST_AUTO_TEST_SUITE_END()  // allocator_suite
