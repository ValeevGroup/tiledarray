/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  tensor_view.cpp
 *  May 29, 2015
 *
 */

#include <chrono>
#include "TiledArray/tensor/tensor_interface.h"
#include "TiledArray/util/random.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct TensorViewFixture {
  TensorViewFixture() {}

  ~TensorViewFixture() {}

  static Tensor<int> random_tensor(const Range& range) {
    Tensor<int> result(range);

    for (auto& value : result) value = TiledArray::rand() % 101;

    return result;
  }

  static const std::array<int, 3> lower_bound;
  static const std::array<int, 3> upper_bound;

  Tensor<int> t{random_tensor(Range(lower_bound, upper_bound))};

  constexpr static std::size_t ntiles_per_test = 10;

};  // TensorViewFixture

const std::array<int, 3> TensorViewFixture::lower_bound{{0, 1, 2}};
const std::array<int, 3> TensorViewFixture::upper_bound{{5, 7, 11}};

BOOST_FIXTURE_TEST_SUITE(tensor_view_suite, TensorViewFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(non_const_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(t.block(lower, upper));
        TensorView<int> view = t.block(lower, upper);

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for (unsigned int i = 0u; i < t.range().rank(); ++i) {
          BOOST_CHECK_EQUAL(view.range().lobound(i), lower[i]);
          BOOST_CHECK_EQUAL(view.range().upbound(i), upper[i]);
          BOOST_CHECK_EQUAL(view.range().extent(i), upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(view.range().stride(i), t.range().stride(i));
          volume *= upper[i] - lower[i];
        }
        BOOST_CHECK_EQUAL(view.size(), volume);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        Range::size_type i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(view(i), view(*it));
          BOOST_CHECK_EQUAL(view(*it), t(*it));
          BOOST_CHECK_EQUAL(view(i), t(*it));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(const_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(t.block(lower, upper));
        TensorConstView<int> view = t.block(lower, upper);

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for (unsigned int i = 0u; i < t.range().rank(); ++i) {
          BOOST_CHECK_EQUAL(view.range().lobound(i), lower[i]);
          BOOST_CHECK_EQUAL(view.range().upbound(i), upper[i]);
          BOOST_CHECK_EQUAL(view.range().extent(i), upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(view.range().stride(i), t.range().stride(i));
          volume *= upper[i] - lower[i];
        }
        BOOST_CHECK_EQUAL(view.size(), volume);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        Range::size_type i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(view(i), view(*it));
          BOOST_CHECK_EQUAL(view(*it), t(*it));
          BOOST_CHECK_EQUAL(view(i), t(*it));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(transitive_read_write) {
  std::default_random_engine generator(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> distribution(0, 100);

  TensorView<int> view = t.block({2, 1, 5}, {4, 6, 8});

  std::size_t i = 0ul;
  for (auto it = view.range().begin(); it != view.range().end(); ++it, ++i) {
    // Test that the view and tensor have the same value
    BOOST_CHECK_EQUAL(view(*it), t(*it));
    BOOST_CHECK_EQUAL(view(i), t(*it));

    // Test that changes to t are visible in the view
    t(*it) = distribution(generator) + 100;
    BOOST_CHECK_EQUAL(view(*it), t(*it));
    BOOST_CHECK_EQUAL(view(i), t(*it));

    // Test that changes to view are visible in t
    view(*it) = distribution(generator) + 200;
    BOOST_CHECK_EQUAL(t(*it), view(*it));
    BOOST_CHECK_EQUAL(t(*it), view(i));
  }
}

BOOST_AUTO_TEST_CASE(assign_tensor_to_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        TensorView<int> view = t.block(lower, upper);
        Tensor<int> tensor = random_tensor(Range(lower, upper));

        BOOST_CHECK_NO_THROW(view = tensor);

        // Check that the view values match that of tensor and that the values
        // are also set in t
        std::size_t i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(view(*it), tensor(*it));
          BOOST_CHECK_EQUAL(view(*it), tensor(i));
          BOOST_CHECK_EQUAL(view(i), tensor(*it));
          BOOST_CHECK_EQUAL(view(i), tensor(i));

          BOOST_CHECK_EQUAL(t(*it), tensor(*it));
          BOOST_CHECK_EQUAL(t(*it), tensor(i));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(copy_view_to_tensor) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        TensorView<int> view = t.block(lower, upper);
        BOOST_CHECK_NO_THROW(Tensor<int> tensor(view););
        Tensor<int> tensor(view);

        // Check that the values of the tensor are equal to that of the view
        std::size_t i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(tensor(*it), view(*it));
          BOOST_CHECK_EQUAL(tensor(*it), view(i));
          BOOST_CHECK_EQUAL(tensor(i), view(*it));
          BOOST_CHECK_EQUAL(tensor(i), view(i));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(assign_view_to_tensor) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        TensorView<int> view = t.block(lower, upper);
        Tensor<int> tensor(Range(lower, upper), 0);

        BOOST_CHECK_NO_THROW(tensor = view);

        // Check that the values of the tensor are equal to that of the view
        std::size_t i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(tensor(*it), view(*it));
          BOOST_CHECK_EQUAL(tensor(*it), view(i));
          BOOST_CHECK_EQUAL(tensor(i), view(*it));
          BOOST_CHECK_EQUAL(tensor(i), view(i));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(add_tensor_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        TensorView<int> view = t.block(lower, upper);
        Tensor<int> right = random_tensor(detail::clone_range(view));

        Tensor<int> result;
        BOOST_CHECK_NO_THROW(result = view.add(right));

        BOOST_CHECK_EQUAL(result.range(), view.range());
        BOOST_CHECK_EQUAL(result.range(), right.range());

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        for (auto it = result.range().begin(); it != result.range().end();
             ++it) {
          BOOST_CHECK_EQUAL(result(*it), view(*it) + right(*it));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(add_tensor_to_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(t.block(lower, upper));
        TensorView<int> view = t.block(lower, upper);
        Tensor<int> tensor = random_tensor(Range(lower, upper));

        Tensor<int> temp(view);
        BOOST_CHECK_NO_THROW(view.add_to(tensor));

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        std::size_t i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(view(*it), temp(*it) + tensor(*it));
          BOOST_CHECK_EQUAL(view(i), temp(*it) + tensor(*it));
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        TensorView<int> view = t.block(lower, upper);

        Tensor<int> result;
        BOOST_CHECK_NO_THROW(result = view.scale(3));

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        std::size_t i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(result(*it), view(*it) * 3);
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_to_view) {
  std::size_t tile_count = 0;
  for (auto lower_it = t.range().begin(); lower_it != t.range().end();
       ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = t.range().begin(); upper_it != t.range().end();
         ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        TensorView<int> view = t.block(lower, upper);
        Tensor<int> tensor = random_tensor(Range(lower, upper));

        Tensor<int> temp(view);
        BOOST_CHECK_NO_THROW(view.scale_to(3));

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        std::size_t i = 0ul;
        for (auto it = view.range().begin(); it != view.range().end();
             ++it, ++i) {
          BOOST_CHECK_EQUAL(view(*it), temp(*it) * 3);
          BOOST_CHECK_EQUAL(view(i), temp(*it) * 3);
        }

        ++tile_count;
        if (tile_count == ntiles_per_test) return;
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
