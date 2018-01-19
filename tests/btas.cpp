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

#ifdef TILEDARRAY_HAS_BTAS

#include "tiledarray.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include <TiledArray/external/btas.h>

using namespace TiledArray;

// test both bare (deep-copy) BTAS tensor as well as its shallow-copy wrap in Tile<>
typedef boost::mpl::list<
DistArray<Tile<btas::Tensor<double, TA::Range, btas::varray<double>>>, DensePolicy>,
DistArray<     btas::Tensor<double, TA::Range, btas::varray<double>> , DensePolicy>
> array_types;

struct BTASFixture : public TiledRangeFixture {

  BTASFixture() :
      a(*GlobalFixture::world, trange2),
      b(*GlobalFixture::world, trange3),
      c(*GlobalFixture::world, trange4),
      d(*GlobalFixture::world, trange2),
      e(*GlobalFixture::world, trange3),
      f(*GlobalFixture::world, trange4)
  {
    random_fill(a);
    random_fill(b);
    random_fill(c);
    random_fill(d);
    random_fill(e);
    random_fill(f);
    GlobalFixture::world->gop.fence();
  }

  template<typename Tile>
  static void random_fill(DistArray<Tile> &array) {
    using Range = typename DistArray<Tile>::range_type;
    typename DistArray<Tile>::pmap_interface::const_iterator it = array.pmap()->begin();
    typename DistArray<Tile>::pmap_interface::const_iterator end = array.pmap()->end();
    for (; it != end; ++it) {
      array.set(*it, array.world().taskq.add(BTASFixture::make_rand_tile<Tile, Range>,
                                             array.trange().make_tile_range(*it)));
    }
  }

  template<typename T>
  static void set_random(T &t) {
    // with 50% generate nonzero integer value in [0,101)
    auto rand_int = GlobalFixture::world->rand();
    t = (rand_int < 0x8ffffful) ? rand_int % 101 : 0;
  }

  template<typename T>
  static void set_random(std::complex<T> &t) {
    // with 50% generate nonzero value
    auto rand_int1 = GlobalFixture::world->rand();
    if (rand_int1 < 0x8ffffful) {
      t = std::complex<T>{T(rand_int1 % 101),
                          T(GlobalFixture::world->rand() % 101)};
    } else
      t = std::complex<T>{0, 0};
  }

  // Fill a tile with random data
  template<typename Tile, typename Range>
  static Tile
  make_rand_tile(const Range &r) {
    Tile tile(r);
    for(auto& v: tile) {
      set_random(v);
    }
    return tile;
  }

  ~BTASFixture() {
    GlobalFixture::world->gop.fence();
  }

  const static TiledRange trange1;
  const static TiledRange trange2;
  const static TiledRange trange3;
  const static TiledRange trange4;

  using TArrayDSB = DistArray<Tile<btas::Tensor<double, TA::Range, btas::varray<double>>>, DensePolicy>;
  TArrayDSB a, b, c;
  using TArrayDB = DistArray<btas::Tensor<double, TA::Range, btas::varray<double>>, DensePolicy>;
  TArrayDB d, e, f;

  template <typename Array>
  Array& array(size_t idx);

}; // BTASFixture

// Instantiate static variables for fixture
const TiledRange BTASFixture::trange1 =
    {{0, 2, 5, 10, 17, 28, 41}};
const TiledRange BTASFixture::trange2 =
    {{0, 2, 5, 10, 17, 28, 41},
     {0, 3, 6, 11, 18, 29, 42}};
const TiledRange BTASFixture::trange3 =
    {{0, 2, 5, 10, 17, 28, 41},
     {0, 3, 6, 11, 18, 29, 42},
     {0, 4, 5, 12, 17, 30, 41}};
const TiledRange BTASFixture::trange4 =
    {trange2.data()[0], trange2.data()[1], trange2.data()[0], trange2.data()[1]};

template<>
BTASFixture::TArrayDSB&
BTASFixture::array<BTASFixture::TArrayDSB>(size_t idx) {
  if (idx == 0)
    return a;
  else if (idx == 1)
    return b;
  else if (idx == 2)
    return c;
  else
    throw std::range_error("idx out of range");
}

template<>
BTASFixture::TArrayDB&
BTASFixture::array<BTASFixture::TArrayDB>(size_t idx) {
  if (idx == 0)
    return d;
  else if (idx == 1)
    return e;
  else if (idx == 2)
    return f;
  else
    throw std::range_error("idx out of range");
}

BOOST_FIXTURE_TEST_SUITE(btas_suite, BTASFixture)

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

  BOOST_CHECK(std::sqrt((c_copy("i,j,k") - c_ref("i,j,k")).squared_norm().get()) < 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_BLAS
