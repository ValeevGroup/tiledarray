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
 *  tensor_of_tensor.cpp
 *  Jun 12, 2015
 *
 */

#include "TiledArray/tensor.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <boost/mpl/list.hpp>

#ifdef TILEDARRAY_HAS_BTAS
#include <btas/tensor.h>
#endif

using namespace TiledArray;

#ifdef TILEDARRAY_HAS_BTAS
namespace TiledArray {
namespace detail {

template <typename T, typename ... Args>
struct is_tensor_helper<btas::Tensor<T, Args...> > : public std::true_type { };

template <typename T, typename ... Args>
struct is_contiguous_tensor_helper<btas::Tensor<T, Args...> > : public std::true_type { };

}
}
#endif

struct TensorOfTensorFixture {

  TensorOfTensorFixture() :
    a(make_rand_tensor_of_tensor(Range(size))),
    b(make_rand_tensor_of_tensor(Range(size))),
    c(a - b)
#ifdef TILEDARRAY_HAS_BTAS
    ,
    d(make_rand_TobT(Range(size))),
    e(make_rand_TobT(Range(size))),
    f(d - e)
#endif
  { }

  ~TensorOfTensorFixture() { }


  // Fill a tensor with random data
  static Tensor<int> make_rand_tensor(const Range& r) {
    Tensor<int> tensor(r);
    for(std::size_t i = 0ul; i < tensor.size(); ++i)
      tensor[i] = GlobalFixture::world->rand() % 42 + 1;
    return tensor;
  }

  // Fill a tensor with random data
  static Tensor<Tensor<int> > make_rand_tensor_of_tensor(const Range& r) {
    Tensor<Tensor<int> > tensor(r);
    for(std::size_t i = 0ul; i < r.extent(0); ++i) {
      for(std::size_t j = 0ul; j < r.extent(1); ++j) {
        const std::array<std::size_t, 2> lower_bound = {{ i * 10, j * 10 }};
        const std::array<std::size_t, 2> upper_bound = {{ (i + 1) * 10, (j + 1) * 10 }};
        tensor(i,j) = make_rand_tensor(Range(lower_bound, upper_bound));
      }
    }
    return tensor;
  }

  // Fill a tensor with random data
  static Tensor<btas::Tensor<int> > make_rand_TobT(const Range& r) {
    Tensor<btas::Tensor<int> > tensor(r);
    for(std::size_t i = 0ul; i < r.extent(0); ++i) {
      for(std::size_t j = 0ul; j < r.extent(1); ++j) {

        auto make_rand_tensor = [](size_t dim0, size_t dim1) -> btas::Tensor<int> {
          btas::Tensor<int> tensor(dim0, dim1);
          tensor.generate( []() { return GlobalFixture::world->rand() % 42; } );
          return tensor;
        };

        tensor(i,j) = make_rand_tensor(10+i, 10+j);
      }
    }
    return tensor;
  }

  static const std::array<std::size_t, 2> size;
  static const Permutation perm;

  Tensor<Tensor<int> > a, b, c;
  Tensor<btas::Tensor<int>> d, e, f;

  template <typename T>
  Tensor<T>& ToT(size_t idx);

}; // TensorOfTensorFixture

template<>
Tensor<Tensor<int>>&
TensorOfTensorFixture::ToT<Tensor<int>>(size_t idx) {
  if (idx == 0)
    return a;
  else if (idx == 1)
    return b;
  else if (idx == 2)
    return c;
  else
    throw std::range_error("idx out of range");
}

#ifdef TILEDARRAY_HAS_BTAS
template<>
Tensor<btas::Tensor<int>>&
TensorOfTensorFixture::ToT<btas::Tensor<int>>(size_t idx) {
  if (idx == 0)
    return d;
  else if (idx == 1)
    return e;
  else if (idx == 2)
    return f;
  else
    throw std::range_error("idx out of range");
}
#endif

const std::array<std::size_t, 2> TensorOfTensorFixture::size{{10, 10}};
const Permutation TensorOfTensorFixture::perm{1, 0};

BOOST_FIXTURE_TEST_SUITE( tensor_of_tensor_suite, TensorOfTensorFixture )

#ifdef TILEDARRAY_HAS_BTAS
typedef boost::mpl::list<TiledArray::Tensor<int>, btas::Tensor<int>> itensor_types;
#else
typedef boost::mpl::list<TiledArray::Tensor<int>> itensor_types;
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE( default_constructor, ITensor, itensor_types )
{
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t);
  Tensor<ITensor> t;
  BOOST_CHECK(t.data() == nullptr);
  BOOST_CHECK(t.empty());
  BOOST_CHECK_EQUAL(t.size(), 0ul);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( unary_constructor, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(a, [] (const int l) { return l * 2; }));
  Tensor<ITensor> t(a, [] (const int l) { return l * 2; });

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * 2);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( unary_perm_constructor, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(a, [] (const int l) { return l * 2; }, perm));
  Tensor<ITensor> t(a, [] (const int l) { return l * 2; }, perm);

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] * 2);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( binary_constructor, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(a, b,
      [] (const int l, const int r) { return l + r; }));
  Tensor<ITensor> t(a, b, [] (const int l, const int r) { return l + r; });

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( binary_perm_constructor, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);
 BOOST_CHECK_NO_THROW(Tensor<ITensor> t(a, b,
      [] (const int l, const int r) { return l + r; }, perm));
  Tensor<ITensor> t(a, b,
      [] (const int l, const int r) { return l + r; }, perm);

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] + b(j,i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( clone, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.clone());

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());
  BOOST_CHECK_NE(t.data(), a.data());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      BOOST_CHECK_NE(t(i,j).data(), a(i,j).data());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( permute, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.permute(perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      BOOST_CHECK_NE(t(i,j).data(), a(j,i).data());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scale, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.scale(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scale_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.scale(3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scale_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.scale_to(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( add, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scal_add, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] + b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( add_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] + b(j,i)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( scal_add_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b, 3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(j,i)[index] + b(j,i)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( add_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scal_add_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] + b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( add_const, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( add_to_const, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( subt, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scal_subt, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] - b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( subt_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] - b(j,i)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( scal_subt_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, 3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(j,i)[index] - b(j,i)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( subt_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scal_subt_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] - b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( subt_const, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( subt_to_const, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( mult, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scal_mult, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] * b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( mult_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] * b(j,i)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( scal_mult_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, 3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(j,i)[index] * b(j,i)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( mult_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.mult_to(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scal_mult_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.mult_to(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] * b(i,j)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( neg, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.neg());

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], -a(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( neg_perm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.neg(perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(j,i).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], -a(j,i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( neg_to, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.neg_to());

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().extent(0); ++i) {
    for(std::size_t j = 0ul; j < t.range().extent(1); ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], -a(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( sum, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.sum());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( product, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  int x = 1, expected = 1;

  BOOST_CHECK_NO_THROW(x = a.product());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected *= a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( squared_norm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.squared_norm());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( norm, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.norm());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * a[i][j];

  expected = std::sqrt(expected);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( min, ITensor, itensor_types )
{
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = std::numeric_limits<int>::max();

  BOOST_CHECK_NO_THROW(x = c.min());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::min(expected, c[i][j]);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( max, ITensor, itensor_types )
{
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = std::numeric_limits<int>::min();

  BOOST_CHECK_NO_THROW(x = c.max());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::max(expected, c[i][j]);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( abs_min, ITensor, itensor_types )
{
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = std::numeric_limits<int>::max();

  BOOST_CHECK_NO_THROW(x = c.abs_min());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::min(expected, std::abs(c[i][j]));

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( abs_max, ITensor, itensor_types )
{
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = c.abs_max());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::max(expected, std::abs(c[i][j]));

  BOOST_CHECK_EQUAL(x, expected);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( dot, ITensor, itensor_types )
{
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);
  int x = 1, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.dot(b));

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * b[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_SUITE_END()
