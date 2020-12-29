#include <TiledArray/util/random.h>
#include "unit_test_config.h"

using namespace TiledArray::detail;

namespace {

using false_types = boost::mpl::list<int*, char, int&>;

using true_types = boost::mpl::list<int, float, double, std::complex<float>,
                                    std::complex<double>>;
}  // namespace

BOOST_AUTO_TEST_SUITE(can_make_random, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE_TEMPLATE(can_make_random_false, ValueType, false_types) {
  using can_make_random_t = CanMakeRandom<ValueType>;
  BOOST_CHECK(can_make_random_t::value == false);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(can_make_random_true, ValueType, true_types) {
  using can_make_random_t = CanMakeRandom<ValueType>;
  BOOST_CHECK(can_make_random_t::value);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(can_make_random_v_false, ValueType, false_types) {
  BOOST_CHECK(can_make_random_v<ValueType> == false);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(can_make_random_v_true, ValueType, true_types) {
  BOOST_CHECK(can_make_random_v<ValueType> == true);
}

// This should fail to compile
/*BOOST_AUTO_TEST_CASE_TEMPLATE(enable_if_false, ValueType, false_types) {
  using enable_if_t = enable_if_can_make_random_t<ValueType>;
  auto result = std::is_same_v<enable_if_t, void>;
  BOOST_CHECK(result);
}*/

BOOST_AUTO_TEST_CASE_TEMPLATE(enable_if_true, ValueType, true_types) {
  using enable_if_t = enable_if_can_make_random_t<ValueType>;
  auto result = std::is_same_v<enable_if_t, void>;
  BOOST_CHECK(result);
}

BOOST_AUTO_TEST_SUITE_END()
