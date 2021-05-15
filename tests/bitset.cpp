/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/bitset.h"
#include <algorithm>
#include "tiledarray.h"
#include "unit_test_config.h"

#include <climits>

using namespace TiledArray;

struct BitsetFixture {
  typedef TiledArray::detail::Bitset<> Bitset;

  BitsetFixture() : set(size) {}

  static const std::size_t size;
  static const std::size_t blocks;
  Bitset set;
};

const std::size_t BitsetFixture::size =
    sizeof(BitsetFixture::Bitset::block_type) * CHAR_BIT * 10.5;
const std::size_t BitsetFixture::blocks = 11;

// =============================================================================
// Bitset Test Suite

BOOST_FIXTURE_TEST_SUITE(bitset_suite, BitsetFixture, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(size_constructor) {
  // Check for error free construction
  BOOST_REQUIRE_NO_THROW(Bitset b(64));

  Bitset b(64);
  // Check that the size of the bitset is correct
  BOOST_CHECK_EQUAL(b.size(), 64ul);
  BOOST_CHECK_EQUAL(b.num_blocks(), 1ul);

  // check that all bits are correctly initialized to false.
  for (std::size_t i = 0; i < b.size(); ++i) BOOST_CHECK(!b[i]);
}

BOOST_AUTO_TEST_CASE(array_constructor) {
  std::array<int, 125> a;
  std::fill(a.begin(), a.end(), 1);
  // Check for error free copy construction
  BOOST_REQUIRE_NO_THROW(Bitset b(a.begin(), a.end()));

  Bitset b(a.begin(), a.end());
  // Check that the size of the bitset is correct
  BOOST_CHECK_EQUAL(b.size(), 125ul);
  BOOST_CHECK_EQUAL(b.num_blocks(), 2ul);

  // check that all bits are correctly initialized to false.
  for (std::size_t i = 0; i < b.size(); ++i) BOOST_CHECK(b[i]);
}

BOOST_AUTO_TEST_CASE(copy_constructor) {
  // Check for error free copy construction
  BOOST_REQUIRE_NO_THROW(Bitset b(set));

  Bitset b(set);
  // Check that the size of the bitset is correct
  BOOST_CHECK_EQUAL(b.size(), size);
  BOOST_CHECK_EQUAL(b.size(), set.size());
  BOOST_CHECK_EQUAL(b.num_blocks(), blocks);
  BOOST_CHECK_EQUAL(b.num_blocks(), set.num_blocks());

  // check that all bits are correctly initialized to false.
  for (std::size_t i = 0; i < b.size(); ++i) BOOST_CHECK(!b[i]);
}

BOOST_AUTO_TEST_CASE(get) { BOOST_CHECK(set.get() != NULL); }

BOOST_AUTO_TEST_CASE(set_size) { BOOST_CHECK_EQUAL(set.size(), size); }

BOOST_AUTO_TEST_CASE(num_blocks) {
  BOOST_CHECK_EQUAL(set.num_blocks(), blocks);
}

BOOST_AUTO_TEST_CASE(accessor) {
  // check that all bits are correctly initialized to false.
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  // Check that exceptions are thrown when accessing an element that is out of
  // range.
  BOOST_CHECK_THROW(set[set.size()], Exception);
  BOOST_CHECK_THROW(set[set.size() + 1], Exception);
}

BOOST_AUTO_TEST_CASE(set_bit) {
  // Check that the bits are not set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  // Check that we can set all bits
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.set(i);

    for (std::size_t ii = 0; ii < set.size(); ++ii) {
      if (ii <= i)
        BOOST_CHECK(set[ii]);  // Check bits that are set
      else
        BOOST_CHECK(!set[ii]);  // Check bits that are not set
    }

    // Check setting a bit that is already set.
    set.set(i);
    BOOST_CHECK(set[i]);
  }

  // Check that we can unset all bits
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.set(i, false);

    for (std::size_t ii = 0; ii < set.size(); ++ii) {
      if (ii <= i)
        BOOST_CHECK(!set[ii]);  // Check bits that are not set
      else
        BOOST_CHECK(set[ii]);  // Check bits that are set
    }

    // Check setting a bit that is already set.
    set.set(i, false);
    BOOST_CHECK(!set[i]);
  }

  // Check that exceptions are thrown when accessing an element that is out of
  // range.
  BOOST_CHECK_THROW(set.set(set.size()), Exception);
  BOOST_CHECK_THROW(set.set(set.size() + 1), Exception);
}

BOOST_AUTO_TEST_CASE(set_all) {
  // Check that the bits are not set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  set.set();

  // Check that the bits are set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(set[i]);

  set.set();

  // Check that the bits are still set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(set[i]);
}

BOOST_AUTO_TEST_CASE(set_range) {
  // Check that the bits are not set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  set.set_range(10, 20);

  // Check that the bits are set
  std::size_t i = 0ul;
  for (; i < 10ul; ++i) {
    BOOST_CHECK(!set[i]);
    if (set[i]) std::cout << "i = " << i << "\n";
  }
  for (; i <= 20; ++i) {
    BOOST_CHECK(set[i]);
    if (!set[i]) std::cout << "i = " << i << "\n";
  }
  for (; i < size; ++i) {
    BOOST_CHECK(!set[i]);
    if (set[i]) std::cout << "i = " << i << "\n";
  }

  set.set_range(30, 225);

  // Check that the bits are setset
  for (i = 0ul; i < 10ul; ++i) {
    BOOST_CHECK(!set[i]);
    if (set[i]) std::cout << "i = " << i << "\n";
  }
  for (; i <= 20; ++i) {
    BOOST_CHECK(set[i]);
    if (!set[i]) std::cout << "i = " << i << "\n";
  }
  for (; i < 30ul; ++i) {
    BOOST_CHECK(!set[i]);
    if (set[i]) std::cout << "i = " << i << "\n";
  }
  for (; i <= 225; ++i) {
    BOOST_CHECK(set[i]);
    if (!set[i]) std::cout << "i = " << i << "\n";
  }
  for (; i < size; ++i) {
    BOOST_CHECK(!set[i]);
    if (set[i]) std::cout << "i = " << i << "\n";
  }
}

BOOST_AUTO_TEST_CASE(reset_bit) {
  set.set();

  // Check resetting a bit
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.reset(i);

    for (std::size_t ii = 0; ii < set.size(); ++ii) {
      if (ii <= i)
        BOOST_CHECK(!set[ii]);  // Check bits that are not set
      else
        BOOST_CHECK(set[ii]);  // Check bits that are set
    }
  }

  // Check resetting a bit that is not set
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.reset(i);

    for (std::size_t ii = 0; ii < set.size(); ++ii)
      BOOST_CHECK(!set[ii]);  // Check bits that are not set
  }

  // Check that exceptions are thrown when accessing an element that is out of
  // range.
  BOOST_CHECK_THROW(set.reset(set.size()), Exception);
  BOOST_CHECK_THROW(set.reset(set.size() + 1), Exception);
}

BOOST_AUTO_TEST_CASE(reset_all) {
  set.reset();

  // Check that the bits are not set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  set.set();
  set.reset();

  // Check that the bits are set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);
}

BOOST_AUTO_TEST_CASE(bit_flip) {
  // Check that the bits are not set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  // Check that we can set all bits
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.flip(i);

    for (std::size_t ii = 0; ii < set.size(); ++ii) {
      if (ii <= i)
        BOOST_CHECK(set[ii]);  // Check bits that are set
      else
        BOOST_CHECK(!set[ii]);  // Check bits that are not set
    }
  }

  // Check that we can unset all bits
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.flip(i);

    for (std::size_t ii = 0; ii < set.size(); ++ii) {
      if (ii <= i)
        BOOST_CHECK(!set[ii]);  // Check bits that are not set
      else
        BOOST_CHECK(set[ii]);  // Check bits that are set
    }
  }

  // Check that exceptions are thrown when accessing an element that is out of
  // range.
  BOOST_CHECK_THROW(set.flip(set.size()), Exception);
  BOOST_CHECK_THROW(set.flip(set.size() + 1), Exception);
}

BOOST_AUTO_TEST_CASE(flip_all) {
  set.flip();

  // Check that the bits are set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(set[i]);

  set.flip();

  // Check that the bits are not set
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(!set[i]);

  // Check that we can flip alternating bits
  for (std::size_t i = 0; i < set.size(); ++i)
    if (i % 2) set.flip(i);

  set.flip();

  for (std::size_t i = 0; i < set.size(); ++i) {
    if (i % 2)
      BOOST_CHECK(!set[i]);
    else
      BOOST_CHECK(set[i]);
  }
}

BOOST_AUTO_TEST_CASE(assignment) {
  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  Bitset b(size);

  // Check that assignment does not throw.
  BOOST_REQUIRE_NO_THROW(b = set);

  // Check that all bits were copied from set.
  BOOST_CHECK_EQUAL(b.size(), set.size());
  for (std::size_t i = 0ul; i < set.size(); ++i) {
    BOOST_REQUIRE_NO_THROW(b[i]);
    BOOST_CHECK(((b[i] != 0ul) && (set[i] != 0ul)) ||
                ((b[i] == 0ul) && (set[i] == 0ul)));
  }

  // Check that assignment of bitsets with different size is done correctly.
  Bitset small(size / 2);
  small.set();
  small = set;

  BOOST_CHECK_EQUAL(small.size(), set.size());
  for (std::size_t i = 0ul; i < set.size(); ++i) {
    BOOST_REQUIRE_NO_THROW(small[i]);
    BOOST_CHECK(((small[i] != 0ul) && (set[i] != 0ul)) ||
                ((small[i] == 0ul) && (set[i] == 0ul)));
  }

  // Check that assignment of bitsets with different size is done correctly.
  Bitset big(size * 2);
  big.set();
  big = set;

  BOOST_CHECK_EQUAL(big.size(), set.size());
  BOOST_CHECK_EQUAL(big.num_blocks(), set.num_blocks());
  for (std::size_t i = 0ul; i < set.size(); ++i) {
    BOOST_REQUIRE_NO_THROW(big[i]);
    BOOST_CHECK(((big[i] != 0ul) && (set[i] != 0ul)) ||
                ((big[i] == 0ul) && (set[i] == 0ul)));
  }
}

BOOST_AUTO_TEST_CASE(bit_assignment) {
  Bitset even(size);
  Bitset odd(size);
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.set(i);
    if (i % 2)
      even.set(i);
    else
      odd.set(i);
  }

  // Check that and-assignment does not throw.
  BOOST_REQUIRE_NO_THROW(set &= even);

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    if (i % 2)
      BOOST_CHECK(set[i]);
    else
      BOOST_CHECK(!set[i]);
  }

  // Check that or-assignment does not throw.
  BOOST_REQUIRE_NO_THROW(set |= odd);

  // Check for correct or-assignement (all are set)
  for (std::size_t i = 0; i < set.size(); ++i) BOOST_CHECK(set[i]);

  BOOST_REQUIRE_NO_THROW(set ^= odd);
  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    if (i % 2)
      BOOST_CHECK(set[i]);
    else
      BOOST_CHECK(!set[i]);
  }

  // Check that assignment of bitsets with different size throws.
  Bitset bad(size / 2);
  BOOST_CHECK_THROW(bad &= set, Exception);
  BOOST_CHECK_THROW(bad |= set, Exception);
  BOOST_CHECK_THROW(bad ^= set, Exception);
}

BOOST_AUTO_TEST_CASE(bit_operators) {
  Bitset even(size);
  Bitset odd(size);
  for (std::size_t i = 0; i < set.size(); ++i) {
    set.set(i);
    if (i % 2)
      even.set(i);
    else
      odd.set(i);
  }

  // Check and-operator
  set = even & even;

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    if (i % 2)
      BOOST_CHECK(set[i]);
    else
      BOOST_CHECK(!set[i]);
  }

  set = even & odd;

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    BOOST_CHECK(!set[i]);
  }

  // Check or-operator
  set = even | even;

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    if (i % 2)
      BOOST_CHECK(set[i]);
    else
      BOOST_CHECK(!set[i]);
  }

  set = even | odd;

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    BOOST_CHECK(set[i]);
  }

  // Check xor-operator
  set = even ^ even;

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    BOOST_CHECK(!set[i]);
  }

  set = even ^ odd;

  // Check for correct and-assignement (evens are set)
  for (std::size_t i = 0; i < set.size(); ++i) {
    BOOST_CHECK(set[i]);
  }

  // Check that assignment of bitsets with different size throws.
  Bitset bad(size / 2);
  BOOST_CHECK_THROW(bad & set, Exception);
  BOOST_CHECK_THROW(bad | set, Exception);
  BOOST_CHECK_THROW(bad ^ set, Exception);
}

BOOST_AUTO_TEST_CASE(left_shift_assign) {
  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Check each bit shift from 0 to size
  for (std::size_t shift = 0; shift <= size; ++shift) {
    // Shift bitset data
    Bitset temp = set;
    temp <<= shift;

    // Check that the head is filled with zeros
    std::size_t i = 0;
    std::size_t j = 0;
    while ((i < shift) && (i < size)) {
      BOOST_CHECK(temp[i] == 0ul);
      if (temp[i]) std::cout << "i = " << i << "\n";
      ++i;
    }
    // Check that the data has been shifted correctly
    while (i < size) {
      BOOST_CHECK(((temp[i] != 0ul) && (set[j] != 0ul)) ||
                  ((temp[i] == 0ul) && (set[j] == 0ul)));
      if (!(((temp[i] != 0ul) && (set[j] != 0ul)) ||
            ((temp[i] == 0ul) && (set[j] == 0ul))))
        std::cout << "i = " << i << ", j = " << j << "\n";
      ++i;
      ++j;
    }
  }
}

BOOST_AUTO_TEST_CASE(left_shift) {
  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Check each bit shift from 0 to size
  for (std::size_t shift = 0; shift <= size; ++shift) {
    // Store shifted copy of bitset
    Bitset temp = set << shift;

    // Check that the head is filled with zeros
    std::size_t i = 0;
    std::size_t j = 0;
    while ((i < shift) && (i < size)) {
      BOOST_CHECK(temp[i] == 0ul);
      if (temp[i]) std::cout << "i = " << i << "\n";
      ++i;
    }
    // Check that the data has been shifted correctly
    while (i < size) {
      BOOST_CHECK(((temp[i] != 0ul) && (set[j] != 0ul)) ||
                  ((temp[i] == 0ul) && (set[j] == 0ul)));
      if (!(((temp[i] != 0ul) && (set[j] != 0ul)) ||
            ((temp[i] == 0ul) && (set[j] == 0ul))))
        std::cout << "i = " << i << ", j = " << j << "\n";
      ++i;
      ++j;
    }
  }
}

BOOST_AUTO_TEST_CASE(right_shift_assign) {
  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Check each bit shift from 0 to size
  for (std::size_t shift = 0; shift <= size; ++shift) {
    // Shift bitset data
    Bitset temp = set;
    temp >>= shift;

    // Check that the data has been shifted correctly
    std::size_t i = 0;
    std::size_t j = shift;
    while ((i < (size - shift)) && (j < size)) {
      BOOST_CHECK(((temp[i] != 0ul) && (set[j] != 0ul)) ||
                  ((temp[i] == 0ul) && (set[j] == 0ul)));
      if (!(((temp[i] != 0ul) && (set[j] != 0ul)) ||
            ((temp[i] == 0ul) && (set[j] == 0ul))))
        std::cout << "i = " << i << ", j = " << j << "\n";
      ++i;
      ++j;
    }
    // Check that the tail is filled with zeros
    while (i < size) {
      BOOST_CHECK(temp[i] == 0ul);
      if (temp[i]) std::cout << "i = " << i << "\n";
      ++i;
    }
  }
}

BOOST_AUTO_TEST_CASE(right_shift) {
  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Check each bit shift from 0 to size
  for (std::size_t shift = 0ul; shift <= size; ++shift) {
    // Store shifted copy of bitset
    Bitset temp = set >> shift;

    // Check that the data has been shifted correctly
    std::size_t i = 0;
    std::size_t j = shift;
    while ((i < (size - shift)) && (j < size)) {
      BOOST_CHECK(((temp[i] != 0ul) && (set[j] != 0ul)) ||
                  ((temp[i] == 0ul) && (set[j] == 0ul)));
      if (!(((temp[i] != 0ul) && (set[j] != 0ul)) ||
            ((temp[i] == 0ul) && (set[j] == 0ul))))
        std::cout << "i = " << i << ", j = " << j << "\n";
      ++i;
      ++j;
    }
    // Check that the tail is filled with zeros
    while (i < size) {
      BOOST_CHECK(temp[i] == 0ul);
      if (temp[i]) std::cout << "i = " << i << "\n";
      ++i;
    }
  }
}

BOOST_AUTO_TEST_CASE(bit_count) {
  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Count bits
  std::size_t count = 0ul;
  for (std::size_t i = 0ul; i < size; ++i)
    if (set[i]) ++count;

  BOOST_CHECK_EQUAL(set.count(), count);
}

BOOST_AUTO_TEST_CASE(operator_bool) {
  // Check that a bitset full of zeros returns false
  BOOST_CHECK_EQUAL(static_cast<bool>(set), false);

  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Check that a bitset with non-zero data returns true
  BOOST_CHECK_EQUAL(static_cast<bool>(set), true);
}

BOOST_AUTO_TEST_CASE(operator_not) {
  // Check that a bitset full of zeros returns true
  BOOST_CHECK_EQUAL(!set, true);

  // Fill bitset with random data
  std::size_t n = size * 0.25;
  GlobalFixture::world->srand(27);
  for (std::size_t i = 0; i < n; ++i)
    set.set(std::size_t(GlobalFixture::world->rand()) % size);

  // Check that a bitset with non-zero data returns false
  BOOST_CHECK_EQUAL(!set, false);
}

BOOST_AUTO_TEST_SUITE_END()
