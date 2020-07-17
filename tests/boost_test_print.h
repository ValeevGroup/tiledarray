//
// Created by Eduard Valeyev on 7/14/20.
//

#ifndef TILEDARRAY_TESTS_BOOST_TEST_PRINT_H_
#define TILEDARRAY_TESTS_BOOST_TEST_PRINT_H_

#include <iosfwd>
#include <vector>

// teach Boost.Test how to print std::vector
// boost printing method
namespace boost {
namespace test_tools {
namespace tt_detail {
template <typename T>
struct print_log_value<std::vector<T> > {
  void operator()(std::ostream& s, const std::vector<T>& v) {
    const auto sz = v.size();

    if (sz == 0) {
      s << "[]";
    } else {
      s << "[ ";
      if (sz > 1)
        for (std::size_t i = 0; i != sz - 2; ++i) {
          s << v[i] << ", ";
        }
    }
    s << v.back() << " ]";
  }
};
}  // namespace tt_detail
}  // namespace test_tools
}  // namespace boost

#endif  // TILEDARRAY_TESTS_BOOST_TEST_PRINT_H_
