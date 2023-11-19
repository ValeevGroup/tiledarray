/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  dist_op_group.cpp
 *  Oct 21, 2013
 *
 */

#include "TiledArray/external/madness.h"
#include "unit_test_config.h"

using madness::DistributedID;
using madness::Group;

struct GroupFixture {
  GroupFixture() : did(madness::uniqueidT(), 0ul), group_list() {
    for (ProcessID p = GlobalFixture::world->rank() % 2;
         p < GlobalFixture::world->size(); p += 2)
      group_list.push_back(p);
  }

  ~GroupFixture() {}

  DistributedID did;
  std::vector<ProcessID> group_list;

};  // GroupFixture

BOOST_FIXTURE_TEST_SUITE(dist_op_group_suite, GroupFixture)

BOOST_AUTO_TEST_CASE(constructor_empty) {
  // Check default constructor
  BOOST_CHECK_NO_THROW(Group());
  Group empty_group;

  // Check that the group is empty.
  BOOST_CHECK(empty_group.empty());
  BOOST_CHECK_EQUAL(empty_group.size(), 0);

#if defined(MADNESS_ASSERTIONS_THROW)
  // Check that accessing group data throws exceptions for an empty group.
  BOOST_CHECK_TA_ASSERT(empty_group.id(), madness::MadnessException);
  BOOST_CHECK_TA_ASSERT(empty_group.get_world(), madness::MadnessException);
  BOOST_CHECK_TA_ASSERT(empty_group.rank(), madness::MadnessException);
  BOOST_CHECK_TA_ASSERT(empty_group.rank(0), madness::MadnessException);
  BOOST_CHECK_TA_ASSERT(empty_group.world_rank(0), madness::MadnessException);
  ProcessID parent, child1, child2;
  BOOST_CHECK_TA_ASSERT(empty_group.make_tree(0, parent, child1, child2),
                        madness::MadnessException);
#endif  // MADNESS_ASSERTIONS_THROW
}

BOOST_AUTO_TEST_CASE(constructor_new_group) {
  // Check new group constructor
  BOOST_CHECK_NO_THROW(Group(*GlobalFixture::world, group_list, did));
  Group new_group(*GlobalFixture::world, group_list, did);

  // Check that the group is not empty
  BOOST_CHECK(!new_group.empty());

  // Check that the rank and size of the group are correct
  BOOST_CHECK_EQUAL(new_group.rank(), GlobalFixture::world->rank() / 2);
  BOOST_CHECK_EQUAL(new_group.size(), group_list.size());

  // Check that the group id and world are correct
  BOOST_CHECK_EQUAL(new_group.id(), did);
  BOOST_CHECK_EQUAL(&new_group.get_world(), GlobalFixture::world);

  // Check that the group correctly maps parent world ranks to/from group ranks
  for (std::size_t i = 0ul; i < group_list.size(); ++i) {
    BOOST_CHECK_EQUAL(new_group.rank(group_list[i]), i);
    BOOST_CHECK_EQUAL(new_group.world_rank(i), group_list[i]);
  }

  // Check that binary tree returns processes in the group list.
  ProcessID parent, child1, child2;
  BOOST_CHECK_NO_THROW(new_group.make_tree(0, parent, child1, child2));
  BOOST_CHECK((parent == -1) || (std::find(group_list.begin(), group_list.end(),
                                           parent) != group_list.end()));
  BOOST_CHECK((child1 == -1) || (std::find(group_list.begin(), group_list.end(),
                                           child1) != group_list.end()));
  BOOST_CHECK((child2 == -1) || (std::find(group_list.begin(), group_list.end(),
                                           child2) != group_list.end()));
}

BOOST_AUTO_TEST_CASE(copy_group) {
  Group group(*GlobalFixture::world, group_list, did);
  BOOST_CHECK_NO_THROW(Group copy_group(group));

  // Check copy constructor
  Group copy_group(group);

  // Check that the group is not empty
  BOOST_CHECK_EQUAL(copy_group.empty(), group.empty());

  // Check that the rank and size of the group are correct
  BOOST_CHECK_EQUAL(copy_group.rank(), group.rank());
  BOOST_CHECK_EQUAL(copy_group.size(), group.size());

  // Check that the group id and world are correct
  BOOST_CHECK_EQUAL(copy_group.id(), group.id());
  BOOST_CHECK_EQUAL(&group.get_world(), &group.get_world());

  // Check that the group correctly maps parent world ranks to/from group ranks
  for (std::size_t i = 0ul; i < group_list.size(); ++i) {
    BOOST_CHECK_EQUAL(copy_group.rank(group_list[i]),
                      group.rank(group_list[i]));
    BOOST_CHECK_EQUAL(copy_group.world_rank(i), group.world_rank(i));
  }

  // Check that binary tree returns processes in the group list.
  ProcessID parent, child1, child2;
  BOOST_CHECK_NO_THROW(copy_group.make_tree(0, parent, child1, child2));
  BOOST_CHECK((parent == -1) || (std::find(group_list.begin(), group_list.end(),
                                           parent) != group_list.end()));
  BOOST_CHECK((child1 == -1) || (std::find(group_list.begin(), group_list.end(),
                                           child1) != group_list.end()));
  BOOST_CHECK((child2 == -1) || (std::find(group_list.begin(), group_list.end(),
                                           child2) != group_list.end()));
}

BOOST_AUTO_TEST_SUITE_END()
