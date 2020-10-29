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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  symm_irrep.cpp
 *  May 19, 2015
 *
 */

#include "TiledArray/symm/irrep.h"
#include "unit_test_config.h"

struct IrrepFixture {
  IrrepFixture() {}

  ~IrrepFixture() {}

};  // IrrepFixture

using TiledArray::Irrep;

BOOST_FIXTURE_TEST_SUITE(symm_irrep_suite, IrrepFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(constructor) {
  {
    BOOST_CHECK_NO_THROW(Irrep e_111_123({1, 1, 1}, {1, 2, 3}));
    Irrep e_111_123({1, 1, 1}, {1, 2, 3});

    BOOST_CHECK_EQUAL(e_111_123.degree(), 3u);
    BOOST_CHECK_EQUAL(e_111_123.data()[0], 1u);
    BOOST_CHECK_EQUAL(e_111_123.data()[1], 1u);
    BOOST_CHECK_EQUAL(e_111_123.data()[2], 1u);
    BOOST_CHECK_EQUAL(e_111_123.data()[3], 1u);
    BOOST_CHECK_EQUAL(e_111_123.data()[4], 2u);
    BOOST_CHECK_EQUAL(e_111_123.data()[5], 3u);
  }

  {
    BOOST_CHECK_NO_THROW(Irrep e_3_111({3}, {1, 1, 1}));
    Irrep e_3_111({3}, {1, 1, 1});

    BOOST_CHECK_EQUAL(e_3_111.degree(), 3u);
    BOOST_CHECK_EQUAL(e_3_111.data()[0], 3u);
    BOOST_CHECK_EQUAL(e_3_111.data()[1], 0u);
    BOOST_CHECK_EQUAL(e_3_111.data()[2], 0u);
    BOOST_CHECK_EQUAL(e_3_111.data()[3], 1u);
    BOOST_CHECK_EQUAL(e_3_111.data()[4], 1u);
    BOOST_CHECK_EQUAL(e_3_111.data()[5], 1u);
  }

  {
    BOOST_CHECK_NO_THROW(Irrep e_21_112({2, 1}, {1, 1, 2}));
    Irrep e_21_112({2, 1}, {1, 1, 2});

    BOOST_CHECK_EQUAL(e_21_112.degree(), 3u);
    BOOST_CHECK_EQUAL(e_21_112.data()[0], 2u);
    BOOST_CHECK_EQUAL(e_21_112.data()[1], 1u);
    BOOST_CHECK_EQUAL(e_21_112.data()[2], 0u);
    BOOST_CHECK_EQUAL(e_21_112.data()[3], 1u);
    BOOST_CHECK_EQUAL(e_21_112.data()[4], 1u);
    BOOST_CHECK_EQUAL(e_21_112.data()[5], 2u);
  }

  {
    BOOST_CHECK_NO_THROW(Irrep e_21_121({2, 1}, {1, 2, 1}));
    Irrep e_21_121({2, 1}, {1, 2, 1});

    BOOST_CHECK_EQUAL(e_21_121.degree(), 3u);
    BOOST_CHECK_EQUAL(e_21_121.data()[0], 2u);
    BOOST_CHECK_EQUAL(e_21_121.data()[1], 1u);
    BOOST_CHECK_EQUAL(e_21_121.data()[2], 0u);
    BOOST_CHECK_EQUAL(e_21_121.data()[3], 1u);
    BOOST_CHECK_EQUAL(e_21_121.data()[4], 2u);
    BOOST_CHECK_EQUAL(e_21_121.data()[5], 1u);
  }
}

BOOST_AUTO_TEST_SUITE_END()
