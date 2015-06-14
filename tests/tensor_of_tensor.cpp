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

using namespace TiledArray;

struct TensorOfTensorFixture {

  TensorOfTensorFixture() :
    a(make_rand_tensor_of_tensor(Range(size))),
    b(make_rand_tensor_of_tensor(Range(size))),
    c(a - b)
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
    for(std::size_t i = 0ul; i < r.size()[0]; ++i) {
      for(std::size_t j = 0ul; j < r.size()[1]; ++j) {
        const std::array<std::size_t, 2> lower_bound = {{ i * 10, j * 10 }};
        const std::array<std::size_t, 2> upper_bound = {{ (i + 1) * 10, (j + 1) * 10 }};
        tensor(i,j) = make_rand_tensor(Range(lower_bound, upper_bound));
      }
    }
    return tensor;
  }

  static const std::array<std::size_t, 2> size;
  static const Permutation perm;

  Tensor<Tensor<int> > a, b, c;

}; // TensorOfTensorFixture

const std::array<std::size_t, 2> TensorOfTensorFixture::size{{10, 10}};
const Permutation TensorOfTensorFixture::perm{1, 0};

BOOST_FIXTURE_TEST_SUITE( tensor_of_tensor_suite, TensorOfTensorFixture )

BOOST_AUTO_TEST_CASE( default_constructor )
{
  BOOST_CHECK_NO_THROW(Tensor<Tensor<int> > t);
  Tensor<Tensor<int> > t;
  BOOST_CHECK(t.data() == nullptr);
  BOOST_CHECK(t.empty());
  BOOST_CHECK_EQUAL(t.size(), 0ul);
}

BOOST_AUTO_TEST_CASE( unary_constructor )
{
  BOOST_CHECK_NO_THROW(Tensor<Tensor<int> > t(a, [] (const int l) { return l * 2; }));
  Tensor<Tensor<int> > t(a, [] (const int l) { return l * 2; });

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * 2);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( unary_perm_constructor )
{
  BOOST_CHECK_NO_THROW(Tensor<Tensor<int> > t(a, [] (const int l) { return l * 2; }, perm));
  Tensor<Tensor<int> > t(a, [] (const int l) { return l * 2; }, perm);

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] * 2);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( binary_constructor )
{
  BOOST_CHECK_NO_THROW(Tensor<Tensor<int> > t(a, b,
      [] (const int l, const int r) { return l + r; }));
  Tensor<Tensor<int> > t(a, b, [] (const int l, const int r) { return l + r; });

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( binary_perm_constructor )
{
  BOOST_CHECK_NO_THROW(Tensor<Tensor<int> > t(a, b,
      [] (const int l, const int r) { return l + r; }, perm));
  Tensor<Tensor<int> > t(a, b,
      [] (const int l, const int r) { return l + r; }, perm);

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] + b(j,i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( clone )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.clone());

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());
  BOOST_CHECK_NE(t.data(), a.data());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      BOOST_CHECK_NE(t(i,j).data(), a(i,j).data());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( permute )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.permute(perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      BOOST_CHECK_NE(t(i,j).data(), a(j,i).data());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scale )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.scale(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scale_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.scale(3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scale_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.scale_to(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( add )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.add(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scal_add )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.add(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] + b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( add_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.add(b, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] + b(j,i)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( scal_add_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.add(b, 3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(j,i)[index] + b(j,i)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( add_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scal_add_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] + b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( add_const )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.add(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( add_to_const )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] + 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( subt )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.subt(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scal_subt )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] - b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( subt_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] - b(j,i)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( scal_subt_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, 3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(j,i)[index] - b(j,i)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( subt_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scal_subt_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] - b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( subt_const )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.subt(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( subt_to_const )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] - 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( mult )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.mult(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scal_mult )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] * b(i,j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( mult_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(j,i)[index] * b(j,i)[index]);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( scal_mult_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, 3, perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(j,i)[index] * b(j,i)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( mult_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.mult_to(b));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], a(i,j)[index] * b(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( scal_mult_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.mult_to(b, 3));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], (a(i,j)[index] * b(i,j)[index]) * 3);
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( neg )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.neg());

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], -a(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( neg_perm )
{
  Tensor<Tensor<int> > t;
  BOOST_CHECK_NO_THROW(t = a.neg(perm));

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(j,i).range());
      for(std::size_t index = 0ul; index < t(j,i).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], -a(j,i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( net_to )
{
  Tensor<Tensor<int> > t = a.clone();
  BOOST_CHECK_NO_THROW(t.neg_to());

  BOOST_CHECK(! t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for(std::size_t i = 0ul; i < t.range().size()[0]; ++i) {
    for(std::size_t j = 0ul; j < t.range().size()[1]; ++j) {
      BOOST_CHECK(! t(i,j).empty());
      BOOST_CHECK_EQUAL(t(i,j).range(), a(i,j).range());
      for(std::size_t index = 0ul; index < t(i,j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i,j)[index], -a(i,j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( sum )
{
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.sum());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( product )
{
  int x = 1, expected = 1;

  BOOST_CHECK_NO_THROW(x = a.product());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected *= a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( squared_norm )
{
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.squared_norm());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( norm )
{
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.norm());

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * a[i][j];

  expected = std::sqrt(expected);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( min )
{
  int x = 0, expected = std::numeric_limits<int>::max();

  BOOST_CHECK_NO_THROW(x = c.min());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::min(expected, c[i][j]);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( max )
{
  int x = 0, expected = std::numeric_limits<int>::min();

  BOOST_CHECK_NO_THROW(x = c.max());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::max(expected, c[i][j]);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( abs_min )
{
  int x = 0, expected = std::numeric_limits<int>::max();

  BOOST_CHECK_NO_THROW(x = c.abs_min());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::min(expected, std::abs(c[i][j]));

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE( abs_max )
{
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = c.abs_max());

  for(std::size_t i = 0ul; i < c.size(); ++i)
    for(std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::max(expected, std::abs(c[i][j]));

  BOOST_CHECK_EQUAL(x, expected);
}


BOOST_AUTO_TEST_CASE( dot )
{
  int x = 1, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.dot(b));

  for(std::size_t i = 0ul; i < a.size(); ++i)
    for(std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * b[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_SUITE_END()
