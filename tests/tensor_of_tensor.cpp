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

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_BTAS
#include <TiledArray/external/btas.h>
#include "btas/generic/contract.h"
#endif

#include <TiledArray/tensor.h>
#include <TiledArray/tile_interface/add.h>
#include <TiledArray/tile_interface/scale.h>
#include <TiledArray/tile_op/tile_interface.h>

#include <../tests/unit_test_config.h>

#include <boost/mpl/list.hpp>

using namespace TiledArray;

using bTensorI = btas::Tensor<int, Range>;

struct TensorOfTensorFixture {
  TensorOfTensorFixture()
      : a(make_rand_tensor_of_tensor(Range(size))),
        b(make_rand_tensor_of_tensor(Range(size))),
        c(a - b)
#ifdef TILEDARRAY_HAS_BTAS
        ,
        d(make_rand_TobT(Range(size))),
        e(make_rand_TobT(Range(size))),
        f(d - e),
        g(make_rand_TobT_uniform(Range(size))),
        h(make_rand_TobT_uniform(Range(size)))
#endif
  {
  }

  ~TensorOfTensorFixture() {}

  // Fill a tensor with random data
  static Tensor<int> make_rand_tensor(const Range& r) {
    Tensor<int> tensor(r);
    for (std::size_t i = 0ul; i < tensor.size(); ++i)
      tensor[i] = GlobalFixture::world->rand() % 42 + 1;
    return tensor;
  }

  // Fill a tensor with random data
  static Tensor<Tensor<int>> make_rand_tensor_of_tensor(const Range& r) {
    Tensor<Tensor<int>> tensor(r);
    for (decltype(r.extent(0)) i = 0; i < r.extent(0); ++i) {
      for (decltype(r.extent(1)) j = 0; j < r.extent(1); ++j) {
        const std::array<std::size_t, 2> lower_bound = {{i * 10ul, j * 10ul}};
        const std::array<std::size_t, 2> upper_bound = {
            {(i + 1ul) * 10ul, (j + 1ul) * 10ul}};
        tensor(i, j) = make_rand_tensor(Range(lower_bound, upper_bound));
      }
    }
    return tensor;
  }

#ifdef TILEDARRAY_HAS_BTAS
  // Fill a tensor with random data
  static Tensor<bTensorI> make_rand_TobT(const Range& r) {
    Tensor<bTensorI> tensor(r);
    for (decltype(r.extent(0)) i = 0ul; i < r.extent(0); ++i) {
      for (decltype(r.extent(1)) j = 0ul; j < r.extent(1); ++j) {
        auto make_rand_tensor = [](size_t dim0, size_t dim1) -> bTensorI {
          bTensorI tensor(dim0, dim1);
          tensor.generate([]() { return GlobalFixture::world->rand() % 42; });
          return tensor;
        };

        tensor(i, j) = make_rand_tensor(10 + i, 10 + j);
      }
    }
    return tensor;
  }
  // same as make_rand_TobT but with identically-sized tiles
  static Tensor<bTensorI> make_rand_TobT_uniform(const Range& r) {
    Tensor<bTensorI> tensor(r);
    for (decltype(r.extent(0)) i = 0ul; i < r.extent(0); ++i) {
      for (decltype(r.extent(1)) j = 0ul; j < r.extent(1); ++j) {
        auto make_rand_tensor = [](size_t dim0, size_t dim1) -> bTensorI {
          bTensorI tensor(dim0, dim1);
          tensor.generate([]() { return GlobalFixture::world->rand() % 42; });
          return tensor;
        };

        tensor(i, j) = make_rand_tensor(12, 13);
      }
    }
    return tensor;
  }
#endif  // defined(TILEDARRAY_HAS_BTAS)

  static const std::array<std::size_t, 2> size;
  static const Permutation perm;
  static const BipartitePermutation bperm;

  Tensor<Tensor<int>> a, b, c;
#ifdef TILEDARRAY_HAS_BTAS
  Tensor<bTensorI> d, e, f, g, h;
#endif  // defined(TILEDARRAY_HAS_BTAS)

  template <typename T>
  Tensor<T>& ToT(size_t idx);

};  // TensorOfTensorFixture

template <>
Tensor<Tensor<int>>& TensorOfTensorFixture::ToT<Tensor<int>>(size_t idx) {
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
template <>
Tensor<bTensorI>& TensorOfTensorFixture::ToT<bTensorI>(size_t idx) {
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

const std::array<std::size_t, 2> TensorOfTensorFixture::size{{10, 11}};
const Permutation TensorOfTensorFixture::perm{1, 0};
const BipartitePermutation TensorOfTensorFixture::bperm(Permutation{1, 0, 3, 2},
                                                        2);

BOOST_FIXTURE_TEST_SUITE(tensor_of_tensor_suite, TensorOfTensorFixture,
                         TA_UT_LABEL_SERIAL)

#ifdef TILEDARRAY_HAS_BTAS
typedef boost::mpl::list<TiledArray::Tensor<int>, bTensorI> itensor_types;
#else
typedef boost::mpl::list<TiledArray::Tensor<int>> itensor_types;
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(default_constructor, ITensor, itensor_types) {
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t);
  Tensor<ITensor> t;
  BOOST_CHECK(t.data() == nullptr);
  BOOST_CHECK(t.empty());
  BOOST_CHECK_EQUAL(t.size(), 0ul);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unary_constructor, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  // apply element-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(a, [](const int l) { return l * 2; }));
  Tensor<ITensor> t(a, [](const int l) { return l * 2; });

  // apply tensor-wise op
  BOOST_CHECK_NO_THROW(
      Tensor<ITensor> t2(a, [](const ITensor& v) { return scale(v, 2); }));
  Tensor<ITensor> t2(a, [](const ITensor& v) { return scale(v, 2); });

  for (auto&& tref : {std::cref(t), std::cref(t2)}) {
    auto& t = tref.get();

    BOOST_CHECK(!t.empty());
    BOOST_CHECK_EQUAL(t.range(), a.range());
    for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
      for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
        BOOST_CHECK(!t(i, j).empty());
        BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
        for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
          BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] * 2);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unary_perm_constructor, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);

  // apply element-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(
      a, [](const int l) { return l * 2; }, perm));
  Tensor<ITensor> t(
      a, [](const int l) { return l * 2; }, perm);

  // apply tensor-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t2(
      a, [](const ITensor& v) { return scale(v, 2); }, perm));
  Tensor<ITensor> t2(
      a, [](const ITensor& v) { return scale(v, 2); }, perm);

  for (auto&& tref : {std::cref(t), std::cref(t2)}) {
    auto& t = tref.get();
    BOOST_CHECK(!t.empty());
    BOOST_CHECK_EQUAL(t.range(), perm * a.range());

    for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
      for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
        BOOST_CHECK(!t(i, j).empty());
        BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
        for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
          BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index] * 2);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unary_bperm_constructor, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);

  // apply element-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(
      a, [](const int l) { return l * 2; }, bperm));
  Tensor<ITensor> t(
      a, [](const int l) { return l * 2; }, bperm);

  // apply tensor-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t2(
      a, [](const ITensor& v) { return scale(v, 2); }, bperm));
  Tensor<ITensor> t2(
      a, [](const ITensor& v) { return scale(v, 2); }, bperm);

  // apply tensor-wise op with explicit permutation of inner tiles
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t3(
      a, [p = inner(bperm)](const ITensor& v) { return scale(v, 2, p); },
      outer(bperm)));
  Tensor<ITensor> t3(
      a, [p = inner(bperm)](const ITensor& v) { return scale(v, 2, p); },
      outer(bperm));

  for (auto&& tref : {std::cref(t), std::cref(t2), std::cref(t3)}) {
    auto& t = tref.get();

    BOOST_CHECK(!t.empty());
    BOOST_CHECK_EQUAL(t.range(), outer(bperm) * a.range());

    for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
      for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
        BOOST_CHECK(!t(i, j).empty());
        BOOST_CHECK_EQUAL(t(i, j).range(), permute(a(j, i).range(), {1, 0}));
        for (auto&& idx : t(i, j).range()) {
          BOOST_CHECK_EQUAL(t(i, j)(idx[0], idx[1]),
                            a(j, i)(idx[1], idx[0]) * 2);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(binary_constructor, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);

  // apply element-wise op
  BOOST_CHECK_NO_THROW(
      Tensor<ITensor> t(a, b, [](const int l, const int r) { return l + r; }));
  Tensor<ITensor> t(a, b, [](const int l, const int r) { return l + r; });

  // apply tensor-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t2(
      a, b, [](const ITensor& l, const ITensor& r) { return add(l, r); }));
  Tensor<ITensor> t2(
      a, b, [](const ITensor& l, const ITensor& r) { return add(l, r); });

  for (auto&& tref : {std::cref(t), std::cref(t2)}) {
    auto& t = tref.get();

    BOOST_CHECK(!t.empty());
    BOOST_CHECK_EQUAL(t.range(), a.range());
    for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
      for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
        BOOST_CHECK(!t(i, j).empty());
        BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
        for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
          BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] + b(i, j)[index]);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(binary_perm_constructor, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);

  // apply element-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(
      a, b, [](const int l, const int r) { return l + r; }, perm));
  Tensor<ITensor> t(
      a, b, [](const int l, const int r) { return l + r; }, perm);

  // apply tensor-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t2(
      a, b, [](const ITensor& l, const ITensor& r) { return add(l, r); },
      perm));
  Tensor<ITensor> t2(
      a, b, [](const ITensor& l, const ITensor& r) { return add(l, r); }, perm);

  for (auto&& tref : {std::cref(t), std::cref(t2)}) {
    auto& t = tref.get();

    BOOST_CHECK(!t.empty());
    BOOST_CHECK_EQUAL(t.range(), perm * a.range());

    for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
      for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
        BOOST_CHECK(!t(i, j).empty());
        BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
        for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
          BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index] + b(j, i)[index]);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(binary_bperm_constructor, ITensor,
                              itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);

  // apply element-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t(
      a, b, [](const int l, const int r) { return l + r; }, bperm));
  Tensor<ITensor> t(
      a, b, [](const int l, const int r) { return l + r; }, bperm);

  // apply tensor-wise op
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t2(
      a, b, [](const ITensor& l, const ITensor& r) { return add(l, r); },
      bperm));
  Tensor<ITensor> t2(
      a, b, [](const ITensor& l, const ITensor& r) { return add(l, r); },
      bperm);

  // apply tensor-wise op with explicit permutation of inner tiles
  BOOST_CHECK_NO_THROW(Tensor<ITensor> t3(
      a, b,
      [p = inner(bperm)](const ITensor& l, const ITensor& r) {
        return add(l, r, p);
      },
      outer(bperm)));
  Tensor<ITensor> t3(
      a, b,
      [p = inner(bperm)](const ITensor& l, const ITensor& r) {
        return add(l, r, p);
      },
      outer(bperm));

  for (auto&& tref : {std::cref(t), std::cref(t2), std::cref(t3)}) {
    auto& t = tref.get();

    BOOST_CHECK(!t.empty());
    BOOST_CHECK_EQUAL(t.range(), outer(bperm) * a.range());

    for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
      for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
        BOOST_CHECK(!t(i, j).empty());
        BOOST_CHECK_EQUAL(t(i, j).range(), permute(a(j, i).range(), {1, 0}));
        for (auto&& idx : t(i, j).range()) {
          BOOST_CHECK_EQUAL(t(i, j)(idx[0], idx[1]),
                            a(j, i)(idx[1], idx[0]) + b(j, i)(idx[1], idx[0]));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(clone, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.clone());

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());
  BOOST_CHECK_NE(t.data(), a.data());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      BOOST_CHECK_NE(t(i, j).data(), a(i, j).data());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(permutation, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.permute(perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      BOOST_CHECK_NE(t(i, j).data(), a(j, i).data());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bpermutation, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.permute(bperm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), outer(bperm) * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_NE(t(i, j).data(), a(j, i).data());
      BOOST_CHECK_EQUAL(t(i, j).range(), permute(a(j, i).range(), {1, 0}));
      for (auto&& idx : t(i, j).range()) {
        BOOST_CHECK_EQUAL(t(i, j)(idx[0], idx[1]), a(j, i)(idx[1], idx[0]));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scale, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.scale(3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scale_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.scale(3, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scale_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.scale_to(3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(add, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] + b(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_add, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b, 3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(i, j)[index] + b(i, j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(add_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index] + b(j, i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_add_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(b, 3, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(j, i)[index] + b(j, i)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(add_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(b));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] + b(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_add_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(b, 3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(i, j)[index] + b(i, j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(add_const, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.add(3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] + 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(add_to_const, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.add_to(3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] + 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(subt, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] - b(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_subt, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, 3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(i, j)[index] - b(i, j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(subt_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index] - b(j, i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_subt_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(b, 3, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(j, i)[index] - b(j, i)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(subt_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(b));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] - b(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_subt_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(b, 3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(i, j)[index] - b(i, j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(subt_const, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.subt(3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] - 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(subt_to_const, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.subt_to(3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] - 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mult, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] * b(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_mult, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, 3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(i, j)[index] * b(i, j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mult_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(j, i)[index] * b(j, i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_mult_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.mult(b, 3, perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(j, i)[index] * b(j, i)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mult_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.mult_to(b));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], a(i, j)[index] * b(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scal_mult_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.mult_to(b, 3));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index],
                          (a(i, j)[index] * b(i, j)[index]) * 3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(neg, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.neg());

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], -a(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(neg_perm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t;
  BOOST_CHECK_NO_THROW(t = a.neg(perm));

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), perm * a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(j, i).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], -a(j, i)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(neg_to, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  Tensor<ITensor> t = a.clone();
  BOOST_CHECK_NO_THROW(t.neg_to());

  BOOST_CHECK(!t.empty());
  BOOST_CHECK_EQUAL(t.range(), a.range());

  for (decltype(t.range().extent(0)) i = 0; i < t.range().extent(0); ++i) {
    for (decltype(t.range().extent(1)) j = 0; j < t.range().extent(1); ++j) {
      BOOST_CHECK(!t(i, j).empty());
      BOOST_CHECK_EQUAL(t(i, j).range(), a(i, j).range());
      for (std::size_t index = 0ul; index < t(i, j).size(); ++index) {
        BOOST_CHECK_EQUAL(t(i, j)[index], -a(i, j)[index]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sum, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.sum());

  for (std::size_t i = 0ul; i < a.size(); ++i)
    for (std::size_t j = 0ul; j < a[i].size(); ++j) expected += a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(product, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  int x = 1, expected = 1;

  BOOST_CHECK_NO_THROW(x = a.product());

  for (std::size_t i = 0ul; i < a.size(); ++i)
    for (std::size_t j = 0ul; j < a[i].size(); ++j) expected *= a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(squared_norm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.squared_norm());

  for (std::size_t i = 0ul; i < a.size(); ++i)
    for (std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * a[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(norm, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.norm());

  for (std::size_t i = 0ul; i < a.size(); ++i)
    for (std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * a[i][j];

  expected = std::sqrt(expected);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(min, ITensor, itensor_types) {
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = std::numeric_limits<int>::max();

  BOOST_CHECK_NO_THROW(x = c.min());

  for (std::size_t i = 0ul; i < c.size(); ++i)
    for (std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::min(expected, c[i][j]);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(max, ITensor, itensor_types) {
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = std::numeric_limits<int>::min();

  BOOST_CHECK_NO_THROW(x = c.max());

  for (std::size_t i = 0ul; i < c.size(); ++i)
    for (std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::max(expected, c[i][j]);

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(abs_min, ITensor, itensor_types) {
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = std::numeric_limits<int>::max();

  BOOST_CHECK_NO_THROW(x = c.abs_min());

  for (std::size_t i = 0ul; i < c.size(); ++i)
    for (std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::min(expected, std::abs(c[i][j]));

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(abs_max, ITensor, itensor_types) {
  const auto& c = ToT<ITensor>(2);
  int x = 0, expected = 0;

  BOOST_CHECK_NO_THROW(x = c.abs_max());

  for (std::size_t i = 0ul; i < c.size(); ++i)
    for (std::size_t j = 0ul; j < c[i].size(); ++j)
      expected = std::max(expected, std::abs(c[i][j]));

  BOOST_CHECK_EQUAL(x, expected);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dot, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  const auto& b = ToT<ITensor>(1);
  int x = 1, expected = 0;

  BOOST_CHECK_NO_THROW(x = a.dot(b));

  for (std::size_t i = 0ul; i < a.size(); ++i)
    for (std::size_t j = 0ul; j < a[i].size(); ++j)
      expected += a[i][j] * b[i][j];

  BOOST_CHECK_EQUAL(x, expected);
}

#ifdef TILEDARRAY_HAS_BTAS

/// This reduction operation is used to reduce \c Tensor<bTensor> objects via
/// contraction \sa Tensor::reduce \tparam Tile The tile type; currently
/// supported is btas::Tensor<T>
template <typename Tile>
class Contract {
 public:
  // typedefs
  typedef Tile result_type;
  typedef Tile first_argument_type;
  typedef Tile second_argument_type;

  Contract(double alpha, const btas::DEFAULT::index_type& left_annotation,
           const btas::DEFAULT::index_type& right_annotation,
           const btas::DEFAULT::index_type& result_annotation)
      : alpha_(alpha),
        left_annotation_(left_annotation),
        right_annotation_(right_annotation),
        result_annotation_(result_annotation) {}

  /// Contract \c left and \c right and return the result
  /// \param[in] left The left-hand tile to be contracted
  /// \param[in] right The right-hand tile to be contracted
  void operator()(result_type& result, const first_argument_type* left,
                  const second_argument_type* right) {
    btas::contract(alpha(), *left, left_annotation(), *right,
                   right_annotation(), 1.0, result, result_annotation());
  }

  auto alpha() const { return alpha_; }
  const auto& left_annotation() const { return left_annotation_; }
  const auto& right_annotation() const { return right_annotation_; }
  const auto& result_annotation() const { return result_annotation_; }

 private:
  double alpha_;
  btas::DEFAULT::index_type left_annotation_;
  btas::DEFAULT::index_type right_annotation_;
  btas::DEFAULT::index_type result_annotation_;

};  // class Contract

BOOST_AUTO_TEST_CASE(reduce) {
  using Tile = bTensorI;
  // computes sum_{ij} g("a_ij,b_ij") * h("c_ij,b_ij,")
  auto contract_12_32 = Contract<Tile>{1.0, {1, 2}, {3, 2}, {1, 3}};
  static_assert(detail::is_reduce_op_v<Contract<Tile>, Tile, Tile, Tile>,
                "ouch");
  auto x = g.reduce(
      h, contract_12_32,
      [](auto& result, const auto& arg) {
        if (result.empty())
          result = std::move(arg);
        else {
          result += arg;
        }
      },
      Tile{});

  const auto& range = g.range();
  Tile x_ref;
  for (decltype(range.extent(0)) i = 0ul; i < range.extent(0); ++i) {
    for (decltype(range.extent(1)) j = 0ul; j < range.extent(1); ++j) {
      contract_12_32(x_ref, &g(i, j), &h(i, j));
    }
  }
  BOOST_CHECK_EQUAL(x, x_ref);
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(serialization, ITensor, itensor_types) {
  const auto& a = ToT<ITensor>(0);
  std::size_t buf_size = 10000000;  // enough to store: impossible to compute
                                    // precisely for general ITensor
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  BOOST_REQUIRE_NO_THROW(oar & a);
  std::size_t nbyte = oar.size();
  oar.close();

  typename std::decay<decltype(a)>::type a_roundtrip;
  madness::archive::BufferInputArchive iar(buf, nbyte);
  BOOST_REQUIRE_NO_THROW(iar & a_roundtrip);
  iar.close();

  delete[] buf;

  BOOST_CHECK_EQUAL(a.range(), a_roundtrip.range());
  using std::cbegin;
  using std::cend;
  BOOST_CHECK_EQUAL_COLLECTIONS(cbegin(a), cend(a), cbegin(a_roundtrip),
                                cend(a_roundtrip));
}

BOOST_AUTO_TEST_SUITE_END()
