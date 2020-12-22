/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  btas.cpp
 *  January 18, 2018
 *
 */

#include <TiledArray/config.h>
#include <TiledArray/tensor/type_traits.h>

#ifdef TILEDARRAY_HAS_BTAS

#include <TiledArray/conversions/btas.h>
#include <TiledArray/external/btas.h>
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

static_assert(detail::ordinal_traits<btas::RangeNd<>>::type ==
                  OrdinalType::RowMajor,
              "btas::RangeNd<> is row-major");
static_assert(detail::ordinal_traits<btas::RangeNd<CblasRowMajor>>::type ==
                  OrdinalType::RowMajor,
              "btas::RangeNd<CblasRowMajor> is row-major");
static_assert(detail::ordinal_traits<btas::RangeNd<CblasColMajor>>::type ==
                  OrdinalType::ColMajor,
              "btas::RangeNd<CblasColMajor> is col-major");
static_assert(detail::ordinal_traits<btas::Tensor<double>>::type ==
                  OrdinalType::RowMajor,
              "btas::Tenspr<T> is row-major");
static_assert(
    detail::ordinal_traits<btas::Tensor<double, TiledArray::Range>>::type ==
        OrdinalType::RowMajor,
    "btas::Tenspr<T, TA::Range> is row-major");
static_assert(
    detail::ordinal_traits<
        TiledArray::Tile<btas::Tensor<double, TiledArray::Range>>>::type ==
        OrdinalType::RowMajor,
    "TA::Tile<btas::Tenspr<T, TA::Range>> is row-major");

// test both bare (deep-copy) BTAS tensor as well as its shallow-copy wrap in
// Tile<>, using both btas::RangeNd<> and TiledArray::Range as the range type
typedef boost::mpl::list<
    DistArray<
        Tile<btas::Tensor<double, TiledArray::Range, btas::varray<double>>>,
        DensePolicy>,
    DistArray<btas::Tensor<double, TiledArray::Range, btas::varray<double>>,
              DensePolicy>
    // DistArray<Tile<btas::Tensor<double>> , DensePolicy>, DistArray<
    // btas::Tensor<double>                                          ,
    // DensePolicy>
    >
    array_types;

typedef boost::mpl::list<
    btas::Tensor<double, TiledArray::Range, btas::varray<double>>
    // btas::Tensor<double>
    >
    tensor_types;

struct BTASFixture : public TiledRangeFixture {
  BTASFixture()
      : a0(*GlobalFixture::world, trange2),
        b0(*GlobalFixture::world, trange3),
        c0(*GlobalFixture::world, trange4),
        a1(*GlobalFixture::world, trange2),
        b1(*GlobalFixture::world, trange3),
        c1(*GlobalFixture::world, trange4),
        a2(*GlobalFixture::world, trange2),
        b2(*GlobalFixture::world, trange3),
        c2(*GlobalFixture::world, trange4),
        a3(*GlobalFixture::world, trange2),
        b3(*GlobalFixture::world, trange3),
        c3(*GlobalFixture::world, trange4) {
    random_fill(a0);
    random_fill(b0);
    random_fill(c0);
    random_fill(a1);
    random_fill(b1);
    random_fill(c1);
    random_fill(a2);
    random_fill(b2);
    random_fill(c2);
    random_fill(a3);
    random_fill(b3);
    random_fill(c3);
    GlobalFixture::world->gop.fence();
  }

  template <typename Tile>
  static void random_fill(DistArray<Tile>& array) {
    using Range = typename DistArray<Tile>::range_type;
    typename DistArray<Tile>::pmap_interface::const_iterator it =
        array.pmap()->begin();
    typename DistArray<Tile>::pmap_interface::const_iterator end =
        array.pmap()->end();
    for (; it != end; ++it) {
      array.set(
          *it, array.world().taskq.add(BTASFixture::make_rand_tile<Tile, Range>,
                                       array.trange().make_tile_range(*it)));
    }
  }

  template <typename T>
  static void set_random(T& t) {
    // with 50% generate nonzero integer value in [0,101)
    auto rand_int = GlobalFixture::world->rand();
    t = (rand_int < 0x8fffff) ? rand_int % 101 : 0;
  }

  template <typename T>
  static void set_random(std::complex<T>& t) {
    // with 50% generate nonzero value
    auto rand_int1 = GlobalFixture::world->rand();
    if (rand_int1 < 0x8ffffful) {
      t = std::complex<T>{T(rand_int1 % 101),
                          T(GlobalFixture::world->rand() % 101)};
    } else
      t = std::complex<T>{0, 0};
  }

  // Fill a tile with random data
  template <typename Tile, typename Range>
  static Tile make_rand_tile(const Range& r) {
    Tile tile(r);
    for (auto& v : tile) {
      set_random(v);
    }
    return tile;
  }

  ~BTASFixture() { GlobalFixture::world->gop.fence(); }

  const static TiledRange trange1;
  const static TiledRange trange2;
  const static TiledRange trange3;
  const static TiledRange trange4;

  using TArrayDSB = DistArray<
      Tile<btas::Tensor<double, TiledArray::Range, btas::varray<double>>>,
      DensePolicy>;
  TArrayDSB a0, b0, c0;
  using TArrayDB =
      DistArray<btas::Tensor<double, TiledArray::Range, btas::varray<double>>,
                DensePolicy>;
  TArrayDB a1, b1, c1;
  using TArrayDSB0 = DistArray<Tile<btas::Tensor<double>>, DensePolicy>;
  TArrayDSB0 a2, b2, c2;
  using TArrayDB0 = DistArray<btas::Tensor<double>, DensePolicy>;
  TArrayDB0 a3, b3, c3;

  template <typename Array>
  Array& array(size_t idx);

};  // BTASFixture

// Instantiate static variables for fixture
const TiledRange BTASFixture::trange1{{0, 2, 5, 10, 17, 28, 41}};
const TiledRange BTASFixture::trange2{{0, 2, 5, 10, 17, 28, 41},
                                      {0, 3, 6, 11, 18, 29, 42}};
const TiledRange BTASFixture::trange3{{0, 2, 5, 10, 17, 28, 41},
                                      {0, 3, 6, 11, 18, 29, 42},
                                      {0, 4, 5, 12, 17, 30, 41}};
const TiledRange BTASFixture::trange4{trange2.data()[0], trange2.data()[1],
                                      trange2.data()[0], trange2.data()[1]};

template <>
BTASFixture::TArrayDSB& BTASFixture::array<BTASFixture::TArrayDSB>(size_t idx) {
  if (idx == 0)
    return a0;
  else if (idx == 1)
    return b0;
  else if (idx == 2)
    return c0;
  else
    throw std::range_error("idx out of range");
}

template <>
BTASFixture::TArrayDB& BTASFixture::array<BTASFixture::TArrayDB>(size_t idx) {
  if (idx == 0)
    return a1;
  else if (idx == 1)
    return b1;
  else if (idx == 2)
    return c1;
  else
    throw std::range_error("idx out of range");
}

template <>
BTASFixture::TArrayDSB0& BTASFixture::array<BTASFixture::TArrayDSB0>(
    size_t idx) {
  if (idx == 0)
    return a2;
  else if (idx == 1)
    return b2;
  else if (idx == 2)
    return c2;
  else
    throw std::range_error("idx out of range");
}

template <>
BTASFixture::TArrayDB0& BTASFixture::array<BTASFixture::TArrayDB0>(size_t idx) {
  if (idx == 0)
    return a3;
  else if (idx == 1)
    return b3;
  else if (idx == 2)
    return c3;
  else
    throw std::range_error("idx out of range");
}

BOOST_FIXTURE_TEST_SUITE(btas_suite, BTASFixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(tensor_ctor, Tensor, tensor_types) {
  BOOST_REQUIRE_NO_THROW(Tensor{});
  Tensor t0;
  BOOST_CHECK(t0.empty());

  // copy of empty Tensor should be empty ... this makes sure that Tensor's
  // range treats rank 0 as null state, not as rank-0 state with volume 1
  BOOST_REQUIRE_NO_THROW(Tensor t1 = t0);
  Tensor t1 = t0;
  BOOST_CHECK(t1.empty());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(copy, Array, array_types) {
  const auto& a = array<Array>(0);
  TArrayD b;
  BOOST_REQUIRE_NO_THROW(b("i,j") = a("i,j"));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(contract, Array, array_types) {
  // contract 2 tensors
  const auto& a = array<Array>(0);
  const auto& b = array<Array>(1);
  Array c;
  c("j,k,l") = a("i,j") * b("i,k,l");

  // copy result to standard tensor, to be able to compare with the reference
  TArrayD c_copy(c);

  // compute the reference result using standard tensor
  TArrayD a_copy(a);
  TArrayD b_copy(b);
  TArrayD c_ref;
  c_ref("j,k,l") = a_copy("i,j") * b_copy("i,k,l");

  BOOST_CHECK(
      std::sqrt((c_copy("i,j,k") - c_ref("i,j,k")).squared_norm().get()) <
      1e-10);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(from_btas_subtensor, bTensor, tensor_types) {
  using range_type = typename bTensor::range_type;
  bTensor src = make_rand_tile<bTensor>(range_type({4, 5}));

  Tensor<double> dst(TiledArray::Range({1, 1}, {3, 4}));
  BOOST_REQUIRE_NO_THROW(btas_subtensor_to_tensor(src, dst));

  //  btas_subtensor_to_tensor(src, dst);

  for (const auto& i : dst.range()) {
    BOOST_CHECK_EQUAL(src(i), dst(i));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(to_btas_subtensor, bTensor, tensor_types) {
  Tensor<double> src =
      make_rand_tile<Tensor<double>>(TiledArray::Range({1, 1}, {3, 3}));

  using range_type = typename bTensor::range_type;
  bTensor dst(range_type({4, 5}), 0.0);

  BOOST_REQUIRE_NO_THROW(tensor_to_btas_subtensor(src, dst));

  for (const auto& i : src.range()) {
    BOOST_CHECK_EQUAL(src(i), dst(i));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dense_array_conversion, bTensor, tensor_types) {
  // make random btas::Tensor on World rank 0, and replicate
  const auto root = 0;
  bTensor src;
  if (GlobalFixture::world->rank() == root)
    src = make_rand_tile<bTensor>(typename bTensor::range_type({20, 22, 24}));
  if (GlobalFixture::world->size() != 0)
    GlobalFixture::world->gop.broadcast_serializable(src, root);

  // make tiled range
  using trange1_t = TiledArray::TiledRange1;
  TiledArray::TiledRange trange(
      {trange1_t(0, 10, 20), trange1_t(0, 11, 22), trange1_t(0, 12, 24)});

  // convert to a replicated DistArray
  using T = typename bTensor::value_type;
  using TArray = TiledArray::TArray<T>;
  TArray dst;
  const auto replicated = true;
  if (GlobalFixture::world->size() > 1)
    BOOST_REQUIRE_THROW(dst = btas_tensor_to_array<TArray>(
                            *GlobalFixture::world, trange, src, not replicated),
                        TiledArray::Exception);
  BOOST_REQUIRE_NO_THROW(dst = btas_tensor_to_array<TArray>(
                             *GlobalFixture::world, trange, src, replicated));

  // check the array contents
  for (auto&& t : dst) {
    const auto& tile = t.get();
    const auto& tile_range = tile.range();
    auto src_blk_range = TiledArray::BlockRange(
        trange.elements_range(), tile_range.lobound(), tile_range.upbound());
    using std::data;
    auto src_view = TiledArray::make_const_map(data(src), src_blk_range);

    for (const auto& i : tile_range) {
      BOOST_CHECK_EQUAL(src_view(i), tile(i));
    }
  }

  // convert the replicated DistArray back to a btas::Tensor
  btas::Tensor<T> src_copy;
  BOOST_REQUIRE_NO_THROW(src_copy = array_to_btas_tensor(dst));
  for (const auto& i : src.range()) {
    BOOST_CHECK_EQUAL(src(i), src_copy(i));
  }

  // convert the replicated DistArray to a btas::Tensor on rank 0 only
  {
    btas::Tensor<T> src_copy;
    BOOST_REQUIRE_NO_THROW(src_copy = array_to_btas_tensor(dst, 0));
    if (GlobalFixture::world->rank() == 0) {
      for (const auto& i : src.range()) {
        BOOST_CHECK_EQUAL(src(i), src_copy(i));
      }
    } else {
      BOOST_CHECK(src_copy == btas::Tensor<T>{});
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sparse_array_conversion, bTensor, tensor_types) {
  // make random btas::Tensor on World rank 0, and replicate
  const auto root = 0;
  bTensor src;
  if (GlobalFixture::world->rank() == root)
    src = make_rand_tile<bTensor>(typename bTensor::range_type({20, 22, 24}));
  if (GlobalFixture::world->size() != 0)
    GlobalFixture::world->gop.broadcast_serializable(src, root);

  // make tiled range
  using trange1_t = TiledArray::TiledRange1;
  TiledArray::TiledRange trange(
      {trange1_t(0, 10, 20), trange1_t(0, 11, 22), trange1_t(0, 12, 24)});

  // convert to a replicated sparse policy DistArray
  using T = typename bTensor::value_type;
  using TSpArray = TiledArray::TSpArray<T>;
  TSpArray dst;
  const auto replicated = true;
  if (GlobalFixture::world->size() > 1)
    BOOST_REQUIRE_THROW(dst = btas_tensor_to_array<TSpArray>(
                            *GlobalFixture::world, trange, src, not replicated),
                        TiledArray::Exception);
  BOOST_REQUIRE_NO_THROW(dst = btas_tensor_to_array<TSpArray>(
                             *GlobalFixture::world, trange, src, replicated));

  // check the array contents
  for (auto&& t : dst) {
    const auto& tile = t.get();
    const auto& tile_range = tile.range();
    auto src_blk_range = TiledArray::BlockRange(
        trange.elements_range(), tile_range.lobound(), tile_range.upbound());
    using std::data;
    auto src_view = TiledArray::make_const_map(data(src), src_blk_range);

    for (const auto& i : tile_range) {
      BOOST_CHECK_EQUAL(src_view(i), tile(i));
    }
  }

  // convert to the replicated DistArray back to a btas::Tensor
  btas::Tensor<T> src_copy;
  BOOST_REQUIRE_NO_THROW(src_copy = array_to_btas_tensor(dst));
  for (const auto& i : src.range()) {
    BOOST_CHECK_EQUAL(src(i), src_copy(i));
  }
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_BLAS
