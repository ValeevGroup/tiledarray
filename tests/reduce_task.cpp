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

#include "TiledArray/reduce_task.h"
#include "config.h"
#include <functional>

using namespace TiledArray;
using namespace TiledArray::detail;

template <typename T>
struct plus {
  typedef T result_type;
  typedef T argument_type;

  result_type operator()() const { return result_type(); }

  void operator()(result_type& result, const argument_type& arg) const {
    result += arg;
  }

  void operator()(result_type& result, const argument_type& arg1, const argument_type& arg2) const {
    result += arg1 + arg2;
  }
};


struct ReduceTaskFixture {

  ReduceTaskFixture() : world(*GlobalFixture::world), rt(world, plus<int>()) {

  }

  madness::World& world;
  ReduceTask<plus<int> > rt;

}; // struct ReduceTaskFixture

struct ReduceOp {
  typedef int result_type;
  typedef int first_argument_type;
  typedef int second_argument_type;

  result_type operator()() const { return result_type(); }

  void operator()(result_type& result, const result_type& arg) const {
    result += arg;
  }

  void operator()(result_type& result, const first_argument_type& first, const second_argument_type& second) const {
    result += first * second;
  }

  result_type operator()(const first_argument_type& first1, const second_argument_type& second1,
      const first_argument_type& first2, const second_argument_type& second2) const {
    return first1 * second1 + first2 * second2;
  }
}; // struct ReduceOp

struct ReducePairTaskFixture {

  ReducePairTaskFixture() : world(*GlobalFixture::world), rt(world, ReduceOp()) {

  }

  madness::World& world;
  ReducePairTask<ReduceOp> rt;

}; // struct ReducePairTaskFixture

BOOST_FIXTURE_TEST_SUITE( reduce_task_suite, ReduceTaskFixture )

BOOST_AUTO_TEST_CASE( reduce_value )
{
  int sum = 0;
  for(int i = 0; i < 100; ++i) {
    sum += i;
    rt.add(i);
  }

  madness::Future<int> result = rt.submit();

  BOOST_CHECK_EQUAL(result.get(), sum);
}

BOOST_AUTO_TEST_CASE( reduce_future )
{
  std::vector<madness::Future<int> > fut_vec;

  for(int i = 0; i < 100; ++i) {
    madness::Future<int> f;
    fut_vec.push_back(f);
    rt.add(f);
  }

  madness::Future<int> result = rt.submit();

  BOOST_CHECK(!(result.probe()));

  int sum = 0;
  for(int i = 0; i < 99; ++i) {
    sum += i;
    fut_vec[i].set(i);
    BOOST_CHECK(!(result.probe()));
  }

  sum += 99;
  fut_vec[99].set(99);

  world.gop.fence();

  BOOST_CHECK(result.probe());

  BOOST_CHECK_EQUAL(result.get(), sum);

}

BOOST_AUTO_TEST_SUITE_END()



BOOST_FIXTURE_TEST_SUITE( reduce_pair_task_suite, ReducePairTaskFixture )

BOOST_AUTO_TEST_CASE( reduce_value )
{
  int sum = 0;
  for(int i = 0; i < 100; ++i) {
    sum += i * i;
    BOOST_CHECK_EQUAL(rt.count(), i);
    rt.add(i, i);
    BOOST_CHECK_EQUAL(rt.count(), i + 1);
  }

  madness::Future<int> result = rt.submit();

  BOOST_CHECK_EQUAL(result.get(), sum);
}

BOOST_AUTO_TEST_CASE( reduce_one_value )
{

  BOOST_CHECK_EQUAL(rt.count(), 0);
  rt.add(2, 3);
  BOOST_CHECK_EQUAL(rt.count(), 1);

  madness::Future<int> result = rt.submit();

  BOOST_CHECK_EQUAL(result.get(), 6);
}

BOOST_AUTO_TEST_CASE( reduce_future )
{
  std::vector<madness::Future<int> > fut1_vec;
  std::vector<madness::Future<int> > fut2_vec;

  for(int i = 0; i < 100; ++i) {
    madness::Future<int> f1;
    madness::Future<int> f2;
    fut1_vec.push_back(f1);
    fut2_vec.push_back(f2);
    BOOST_CHECK_EQUAL(rt.count(), i);
    rt.add(f1, f2);
    BOOST_CHECK_EQUAL(rt.count(), i + 1);
  }

  madness::Future<int> result = rt.submit();

  BOOST_CHECK(!(result.probe()));

  int sum = 0;
  for(int i = 0; i < 100; ++i) {
    BOOST_CHECK(!(result.probe()));
    sum += i * i;
    fut1_vec[i].set(i);
    fut2_vec[i].set(i);
  }

  BOOST_CHECK_EQUAL(result.get(), sum);

}

BOOST_AUTO_TEST_CASE( reduce_one_future )
{

  madness::Future<int> f1;
  madness::Future<int> f2;
  rt.add(f1, f2);

  madness::Future<int> result = rt.submit();

  BOOST_CHECK(!(result.probe()));

  BOOST_CHECK(!(result.probe()));
  f1.set(2);
  f2.set(3);

  BOOST_CHECK_EQUAL(result.get(), 6);

}

BOOST_AUTO_TEST_CASE( reduce_zero )
{
  BOOST_CHECK_EQUAL(rt.count(), 0);
  madness::Future<int> result = rt.submit();

  BOOST_CHECK_EQUAL(result.get(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
