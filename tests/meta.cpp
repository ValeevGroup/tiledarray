/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2017  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  type_traits.cpp
 *  Apr 7, 2017
 *
 */

#include "unit_test_config.h"

#include "TiledArray/meta.h"

#include <cmath>

struct MetaFixture {};  // MetaFixture

BOOST_FIXTURE_TEST_SUITE(meta_suite, MetaFixture, TA_UT_LABEL_SERIAL)

double sin(double x) { return std::sin(x); }
double cos(double x) { return std::cos(x); }
madness::Future<double> async_cos(double x) {
  return TiledArray::get_default_world().taskq.add(cos, x);
}

BOOST_AUTO_TEST_CASE(sanity) {
  using namespace TiledArray;
  meta::invoke(sin, meta::invoke(cos, 2.0));
  meta::invoke(sin, meta::invoke(async_cos, 2.0));
}

template <typename T>
class vec {
 public:
  vec() = default;
  vec(const vec& other)
      : size_(other.size_), data_(std::make_unique<T[]>(size_)) {
    std::copy(other.begin(), other.end(), this->begin());
  }
  vec(vec& other) : size_(other.size_), data_(std::make_unique<T[]>(size_)) {
    std::copy(other.begin(), other.end(), this->begin());
  }
  vec(vec&& other) {
    std::cout << "moving vec\n";
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
  }
  explicit vec(size_t size) : size_(size), data_(std::make_unique<T[]>(size)) {}
  vec(size_t size, T value) : size_(size), data_(std::make_unique<T[]>(size)) {
    std::fill(begin(), end(), value);
  }
  vec& operator=(const vec& other) {
    size_ = other.size_;
    data_ = std::make_unique<T[]>(size_);
    std::copy(other.begin(), other.end(), this->begin());
    return *this;
  }
  vec& operator=(vec&& other) = default;
  T* begin() { return data_.get(); }
  T* end() { return begin() + size_; }
  const T* begin() const { return data_.get(); }
  const T* end() const { return begin() + size_; }
  size_t size() const { return size_; }
  template <typename Archive>
  void serialize(Archive& ar) {
    abort();
  }
  T& operator[](size_t idx) { return *(data_ + idx); };

 private:
  size_t size_;
  std::unique_ptr<T[]> data_;
};

vec<double> vsin(const vec<double>& x) {
  vec<double> result(x.size());
  std::transform(x.begin(), x.end(), result.begin(), sin);
  return result;
}
vec<double> vcos(const vec<double>& x) {
  vec<double> result(x.size());
  std::transform(x.begin(), x.end(), result.begin(), cos);
  return result;
}

// these must be disabled since call to taskq.add() cannot deduce rvalue refs as
// args
// vec<double> vsin(vec<double>&& x) {
//  for(auto& v : x) {
//    v = sin(v);
//  }
//  return std::move(x);
//}
// vec<double> vcos(vec<double>&& x) {
//  for(auto& v : x) {
//    v = cos(v);
//  }
//  return std::move(x);
//}

template <typename T>
struct type_printer;
madness::Future<vec<double>> async_vcos(const vec<double>& x) {
  auto exec = [](const vec<double>& x) { return vcos(x); };
  return TiledArray::get_default_world().taskq.add(exec, x);
}

BOOST_AUTO_TEST_CASE(movability) {
  using namespace TiledArray;
  meta::invoke(vsin, meta::invoke(vcos, vec<double>(13, 2.0)));
  meta::invoke(vsin, meta::invoke(async_vcos, vec<double>(13, 2.0)));
}

BOOST_AUTO_TEST_SUITE_END()
