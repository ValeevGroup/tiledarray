/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  tensor_fixture.h
 *  May 8, 2014
 *
 */

#include "TiledArray/tensor.h"
#include "global_fixture.h"

#ifndef TILEDARRAY_TEST_TENSOR_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_TENSOR_FIXTURE_H__INCLUDED

using namespace TiledArray;

struct TensorFixture {
  typedef Tensor<int> TensorN;
  typedef Tensor<double> TensorD;
  typedef Tensor<std::complex<double>> TensorZ;
  typedef TensorN::value_type value_type;
  typedef TensorN::range_type::index index;
  typedef TensorN::size_type size_type;
  typedef TensorN::range_type::index_view_type index_view_type;
  typedef TensorN::range_type range_type;

  static const range_type r;

  TensorFixture() : t(r) { rand_fill(18, t.size(), t.data()); }

  ~TensorFixture() {}

  static range_type make_range(const int seed) {
    GlobalFixture::world->srand(seed);
    std::array<std::size_t, GlobalFixture::dim> start, finish;

    for (unsigned int i = 0ul; i < GlobalFixture::dim; ++i) {
      start[i] = GlobalFixture::world->rand() % 10;
      finish[i] = GlobalFixture::world->rand() % 8 + start[i] + 2;
    }

    return range_type(start, finish);
  }

  template <typename T, typename = std::enable_if_t<std::is_fundamental_v<T>>>
  static void rand_fill(const int seed, const size_type n, T* const data) {
    GlobalFixture::world->srand(seed);
    for (size_type i = 0ul; i < n; ++i)
      data[i] = static_cast<T>(GlobalFixture::world->rand() % 42);
  }

  template <typename T>
  static void rand_fill(const int seed, const size_type n,
                        std::complex<T>* const data) {
    GlobalFixture::world->srand(seed);
    for (size_type i = 0ul; i < n; ++i)
      data[i] = std::complex<T>(GlobalFixture::world->rand() % 42,
                                GlobalFixture::world->rand() % 42);
  }

  static TensorN make_tensor(const int range_seed, const int data_seed) {
    TensorN tensor(make_range(range_seed));
    rand_fill(data_seed, tensor.size(), tensor.data());
    return tensor;
  }

  // make permutation definition object
  static Permutation make_perm() {
    std::array<unsigned int, GlobalFixture::dim> temp;
    for (std::size_t i = 0; i < temp.size(); ++i) temp[i] = i + 1;

    temp.back() = 0;

    return Permutation(temp.begin(), temp.end());
  }

  TensorN t;
};

#endif  // TILEDARRAY_TEST_TENSOR_FIXTURE_H__INCLUDED
