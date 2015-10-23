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
 *  Ed Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  dense_shape.cpp
 *  Oct 21, 2015
 *
 */

#include "TiledArray/symm/permutation_group.h"
#include "TiledArray/symm/representation.h"
#include "TiledArray/symm/oper.h"
#include "TiledArray/symm/dense_shape.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::symmetry::Representation;
using TiledArray::symmetry::PermutationGroup;
using TiledArray::symmetry::SymmetricGroup;
using TiledArray::symmetry::Permutation;
using IKOper = TiledArray::symmetry::IKGroupOperator;

using Rep = Representation<PermutationGroup,IKOper>;

struct SymmetricDenseShapeFixture {

  SymmetricDenseShapeFixture() {
    using TiledArray::symmetry::Permutation;
    triv = std::make_shared<Rep>();
    {
      std::map<Permutation, IKOper> genops;
      genops[Permutation{1,0,2,3}] = IKOper::E();
      s01 = std::make_shared<Rep>(genops);
    }
    {
      std::map<Permutation, IKOper> genops;
      genops[Permutation{1,0,2,3}] = IKOper::I();
      genops[Permutation{0,1,3,2}] = IKOper::I();
      genops[Permutation{2,3,0,1}] = IKOper::K();
      can2b = std::make_shared<Rep>(genops);
    }
  }

  ~SymmetricDenseShapeFixture() { }

  std::shared_ptr<Rep> triv; // trivial representation
  std::shared_ptr<Rep> s01;  // identity under permutation 0<->1
  std::shared_ptr<Rep> can2b;  // canonical symmetry of 2-body fermionic operator, negate under 0<->1, 2<->3, complex conjugate under {0,2}<->{1,3}

}; // SymmetricDenseShapeFixture

BOOST_FIXTURE_TEST_SUITE( symmetric_dense_shape_suite, SymmetricDenseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_CHECK_NO_THROW(TiledArray::symmetry::SymmetricDenseShape<Rep>{});
  BOOST_CHECK_NO_THROW(TiledArray::symmetry::SymmetricDenseShape<Rep>{triv});
  BOOST_CHECK_NO_THROW(TiledArray::symmetry::SymmetricDenseShape<Rep>{s01});
  BOOST_CHECK_NO_THROW(TiledArray::symmetry::SymmetricDenseShape<Rep>{can2b});
}

BOOST_AUTO_TEST_CASE( is_unique )
{
  {
    auto shp = TiledArray::symmetry::SymmetricDenseShape<Rep>{};
    BOOST_CHECK(shp.is_unique(std::vector<int>{}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,1,2}) == true);
    BOOST_CHECK(shp.is_unique({0,1,2}) == true); // check initializer_list index
    BOOST_CHECK(shp.is_unique({'a','b','c'}) == true); // check initializer_list<char>
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,0,2,3}) == true);
    BOOST_CHECK(shp.is_unique({1,0,2,3}) == true); // check initializer_list index
  }
  {
    auto shp = TiledArray::symmetry::SymmetricDenseShape<Rep>{s01};
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,1}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,0}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,0}) == false);
  }
  {
    auto shp = TiledArray::symmetry::SymmetricDenseShape<Rep>{can2b};
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,0,0,0}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,0,0,0}) == false);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,1,0,0}) == false);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,0,1,0}) == false);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,0,0,1}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,1,0,1}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,0,1,0}) == false);
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,1,0,0}) == false);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,0,1,1}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,1,0,2}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,2,0,0}) == false);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,0,1,2}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{0,1,2,3}) == true);
    BOOST_CHECK(shp.is_unique(std::vector<int>{1,0,3,2}) == false);
  }
}

BOOST_AUTO_TEST_CASE( find_unique )
{
  {
    auto shp = TiledArray::symmetry::SymmetricDenseShape<Rep>{};
//    BOOST_CHECK(shp.find_unique(std::vector<int>{}) == std::make_tuple(Permutation{}, identity<IKOper>()));
  }
}

BOOST_AUTO_TEST_SUITE_END()
