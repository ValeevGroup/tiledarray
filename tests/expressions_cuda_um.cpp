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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions.cpp
 *  Aug 08, 2018
 *
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/cuda/btas_um_tensor.h>
#include <range_fixture.h>
#include <tiledarray.h>
#include "unit_test_config.h"

using namespace TiledArray;

struct UMExpressionsFixture : public TiledRangeFixture {
  using UMTensor = TA::Tile<btasUMTensorVarray<double>>;
  using TArrayUMD = TiledArray::DistArray<UMTensor, TA::DensePolicy>;

  UMExpressionsFixture()
      : a(*GlobalFixture::world, tr),
        b(*GlobalFixture::world, tr),
        c(*GlobalFixture::world, tr),
        u(*GlobalFixture::world, trange1),
        v(*GlobalFixture::world, trange1),
        w(*GlobalFixture::world, trange2) {
    random_fill(a);
    random_fill(b);
    random_fill(u);
    random_fill(v);
    GlobalFixture::world->gop.fence();
  }

  template <typename Tile>
  static void random_fill(DistArray<Tile>& array) {
    typename DistArray<Tile>::pmap_interface::const_iterator it =
        array.pmap()->begin();
    typename DistArray<Tile>::pmap_interface::const_iterator end =
        array.pmap()->end();
    for (; it != end; ++it)
      array.set(*it, array.world().taskq.add(
                         &UMExpressionsFixture::template make_rand_tile<Tile>,
                         array.trange().make_tile_range(*it)));
  }

  template <typename T>
  static void set_random(T& t) {
    t = GlobalFixture::world->drand();
  }

  static UMTensor permute_task(const UMTensor& tensor,
                               const Permutation& perm) {
    return perm * tensor;
  }

  static UMTensor permute_fn(const madness::Future<UMTensor>& tensor_f,
                             const Permutation& perm) {
    return madness::add_cuda_task(*GlobalFixture::world, permute_task, tensor_f,
                                  perm)
        .get();
  }

  // Fill a tile with random data
  template <typename Tile>
  static Tile make_rand_tile(const typename TA::Range& r) {
    Tile tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    return tile;
  }

  template <typename M, typename A>
  static void rand_fill_matrix_and_array(M& matrix, A& array, int seed = 42) {
    TA_ASSERT(std::size_t(matrix.size()) ==
              array.trange().elements_range().volume());
    matrix.fill(0);

    GlobalFixture::world->srand(seed);

    // Iterate over local tiles
    for (typename A::iterator it = array.begin(); it != array.end(); ++it) {
      typename A::value_type tile(array.trange().make_tile_range(it.index()));
      for (Range::const_iterator rit = tile.range().begin();
           rit != tile.range().end(); ++rit) {
        const std::size_t elem_index = array.elements_range().ordinal(*rit);
        tile[*rit] =
            (matrix.array()(elem_index) = (GlobalFixture::world->drand()));
      }
      *it = tile;
    }
    GlobalFixture::world->gop.sum(&matrix(0, 0), matrix.size());
  }

  ~UMExpressionsFixture() { GlobalFixture::world->gop.fence(); }

  const static TiledRange trange1;
  const static TiledRange trange2;
  //  const static TiledRange trange3;

  TArrayUMD a;
  TArrayUMD b;
  TArrayUMD c;
  TArrayUMD u;
  TArrayUMD v;
  TArrayUMD w;
  static constexpr double tolerance = 5.0e-14;
};  // UMExpressionsFixture

// Instantiate static variables for fixture
const TiledRange UMExpressionsFixture::trange1 =
    TiledRange{{0, 2, 5, 10, 17, 28, 41}};
const TiledRange UMExpressionsFixture::trange2 =
    TiledRange{{0, 2, 5, 10, 17, 28, 41}, {0, 3, 6, 11, 18, 29, 42}};
// const TiledRange UMExpressionsFixture::trange3 = {{0,11,20}, {0,10,20},
// {0,15,20,30}};

BOOST_FIXTURE_TEST_SUITE(um_expressions_suite, UMExpressionsFixture)

BOOST_AUTO_TEST_CASE(tensor_factories) {
  const auto& ca = a;
  const std::array<int, 3> lobound{{3, 3, 3}};
  const std::array<int, 3> upbound{{5, 5, 5}};

  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") += a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = c("a,c,b") + a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") -= a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = c("a,c,b") - a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") *= a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = c("a,c,b") * a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a").conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}));
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("c,b,a").conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("a,b,c").block({3, 3, 3}, {5, 5, 5}));
}

BOOST_AUTO_TEST_CASE(block_tensor_factories) {
  const auto& ca = a;
  const std::array<int, 3> lobound{{3, 3, 3}};
  const std::array<int, 3> upbound{{5, 5, 5}};

  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           a("a,b,c").block({3, 3, 3}, {5, 5, 5}).conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") += a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           c("b,a,c") + a("b,a,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") -= a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           c("b,a,c") - a("b,a,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") *= a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           c("b,a,c") * a("b,a,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound).conj());
  BOOST_CHECK_NO_THROW(c("a,b,c") = ca("a,b,c").block(lobound, upbound).conj());

  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("a,b,c").block(lobound, upbound) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           2 * (2 * a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           (2 * a("a,b,c").block(lobound, upbound)) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = -a("a,b,c").block(lobound, upbound));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * a("a,b,c").block(lobound, upbound)));

  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(conj(a("a,b,c").block(lobound, upbound))));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(2 * a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(conj(2 * a("a,b,c").block(lobound, upbound))));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           2 * conj(a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(a("a,b,c").block(lobound, upbound)) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           2 * conj(2 * a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           conj(2 * a("a,b,c").block(lobound, upbound)) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("a,b,c").block(lobound, upbound)));
  BOOST_CHECK_NO_THROW(c("a,b,c") =
                           -conj(2 * a("a,b,c").block(lobound, upbound)));
}

BOOST_AUTO_TEST_CASE(scaled_tensor_factories) {
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -a("c,b,a"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * a("c,b,a"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * a("c,b,a")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * a("c,b,a")));
}
//
BOOST_AUTO_TEST_CASE(add_factories) {
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") + b("a,b,c"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (a("c,b,a") + b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * (a("c,b,a") + b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * (a("c,b,a") + b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (conj(a("c,b,a") + b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") + b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") + b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * (a("c,b,a") + b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a") + b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * (a("c,b,a") + b("a,b,c"))) * 2);
}
//
//
BOOST_AUTO_TEST_CASE(subt_factories) {
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") - b("a,b,c"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (a("c,b,a") - b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * (a("c,b,a") - b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * (a("c,b,a") - b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (conj(a("c,b,a") - b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") - b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") - b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * (a("c,b,a") - b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a") - b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * (a("c,b,a") - b("a,b,c"))) * 2);
}

BOOST_AUTO_TEST_CASE(mult_factories) {
  BOOST_CHECK_NO_THROW(c("a,b,c") = a("c,b,a") * b("a,b,c"));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (a("c,b,a") * b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = (2 * (a("c,b,a") * b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * (2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -(2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(conj(2 * (a("c,b,a") * b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (conj(a("c,b,a") * b("a,b,c")))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(a("c,b,a") * b("a,b,c")) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = conj(2 * (a("c,b,a") * b("a,b,c"))) * 2);
  BOOST_CHECK_NO_THROW(c("a,b,c") = 2 * conj(2 * (a("c,b,a") * b("a,b,c"))));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(a("c,b,a") * b("a,b,c")));
  BOOST_CHECK_NO_THROW(c("a,b,c") = -conj(2 * (a("c,b,a") * b("a,b,c"))) * 2);
}

BOOST_AUTO_TEST_CASE(reduce_factories) {
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").sum().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").product().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").squared_norm().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").norm().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").min().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").max().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").abs_min().get());
  BOOST_CHECK_NO_THROW(auto result = a("a,b,c").abs_max().get());
}

BOOST_AUTO_TEST_CASE(permute) {
  Permutation perm({2, 1, 0});
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = b("c,b,a"));

  for (std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (a.is_local(perm_index)) {
      TArrayUMD::value_type a_tile = a.find(perm_index).get();
      TArrayUMD::value_type perm_b_tile = permute_fn(b.find(i), perm);

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], perm_b_tile[j]);
    }
  }

  BOOST_REQUIRE_NO_THROW(b("a,b,c") = b("c,b,a"));

  for (std::size_t i = 0ul; i < b.size(); ++i) {
    if (a.is_local(i)) {
      TArrayUMD::value_type a_tile = a.find(i).get();
      TArrayUMD::value_type b_tile = b.find(i).get();

      BOOST_CHECK_EQUAL(a_tile.range(), b_tile.range());
      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], b_tile[j]);
    }
  }

  Permutation perm2({1, 2, 0});
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = b("b,c,a"));

  for (std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index =
        a.tiles_range().ordinal(perm2 * b.tiles_range().idx(i));
    if (a.is_local(perm_index)) {
      TArrayUMD::value_type a_tile = a.find(perm_index).get();
      TArrayUMD::value_type perm_b_tile = permute_fn(b.find(i), perm2);

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], perm_b_tile[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_permute) {
  Permutation perm({2, 1, 0});
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = 2 * b("c,b,a"));

  for (std::size_t i = 0ul; i < b.size(); ++i) {
    const std::size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (a.is_local(perm_index)) {
      TArrayUMD::value_type a_tile = a.find(perm_index).get();
      TArrayUMD::value_type perm_b_tile = permute_fn(b.find(i), perm);

      BOOST_CHECK_EQUAL(a_tile.range(), perm_b_tile.range());
      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        BOOST_CHECK_EQUAL(a_tile[j], 2 * perm_b_tile[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(block) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto arg_tile = a.find(block_range.ordinal(index)).get();
    auto result_tile = c.find(index).get();

    for (unsigned int r = 0u; r < arg_tile.range().rank(); ++r) {
      BOOST_CHECK_EQUAL(
          result_tile.range().lobound(r),
          arg_tile.range().lobound(r) - a.trange().data()[r].tile(3).first);

      BOOST_CHECK_EQUAL(
          result_tile.range().upbound(r),
          arg_tile.range().upbound(r) - a.trange().data()[r].tile(3).first);

      BOOST_CHECK_EQUAL(result_tile.range().extent(r),
                        arg_tile.range().extent(r));

      BOOST_CHECK_EQUAL(result_tile.range().stride(r),
                        arg_tile.range().stride(r));
    }
    BOOST_CHECK_EQUAL(result_tile.range().volume(), arg_tile.range().volume());

    // Check that the data is correct for the result array.
    for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
      BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                      b("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) &&
        !b.is_zero(block_range.ordinal(index))) {
      auto a_tile = a.find(block_range.ordinal(index)).get();
      auto b_tile = b.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(index).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], a_tile[j] + b_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_AUTO_TEST_CASE(const_block) {
  const TArrayUMD& ca = a;
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = ca("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto arg_tile = a.find(block_range.ordinal(index)).get();
    auto result_tile = c.find(index).get();

    for (unsigned int r = 0u; r < arg_tile.range().rank(); ++r) {
      BOOST_CHECK_EQUAL(
          result_tile.range().lobound(r),
          arg_tile.range().lobound(r) - a.trange().data()[r].tile(3).first);

      BOOST_CHECK_EQUAL(
          result_tile.range().upbound(r),
          arg_tile.range().upbound(r) - a.trange().data()[r].tile(3).first);

      BOOST_CHECK_EQUAL(result_tile.range().extent(r),
                        arg_tile.range().extent(r));

      BOOST_CHECK_EQUAL(result_tile.range().stride(r),
                        arg_tile.range().stride(r));
    }
    BOOST_CHECK_EQUAL(result_tile.range().volume(), arg_tile.range().volume());

    // Check that the data is correct for the result array.
    for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
      BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                      b("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) &&
        !b.is_zero(block_range.ordinal(index))) {
      auto a_tile = a.find(block_range.ordinal(index)).get();
      auto b_tile = b.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(index).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], a_tile[j] + b_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_AUTO_TEST_CASE(scal_block) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto arg_tile = a.find(block_range.ordinal(index)).get();
    auto result_tile = c.find(index).get();

    for (unsigned int r = 0u; r < arg_tile.range().rank(); ++r) {
      BOOST_CHECK_EQUAL(
          result_tile.range().lobound(r),
          arg_tile.range().lobound(r) - a.trange().data()[r].tile(3).first);

      BOOST_CHECK_EQUAL(
          result_tile.range().upbound(r),
          arg_tile.range().upbound(r) - a.trange().data()[r].tile(3).first);

      BOOST_CHECK_EQUAL(result_tile.range().extent(r),
                        arg_tile.range().extent(r));

      BOOST_CHECK_EQUAL(result_tile.range().stride(r),
                        arg_tile.range().stride(r));
    }
    BOOST_CHECK_EQUAL(result_tile.range().volume(), arg_tile.range().volume());

    for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
      BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                  b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) &&
        !b.is_zero(block_range.ordinal(index))) {
      auto a_tile = a.find(block_range.ordinal(index)).get();
      auto b_tile = b.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(index).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (a_tile[j] + b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_AUTO_TEST_CASE(permute_block) {
  Permutation perm({2, 1, 0});
  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto result_tile = c.find(index).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto result_tile = c.find(index).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(index).get();
      auto a_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto b_tile = b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             2 * (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("c,b,a").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    const size_t perm_index =
        c.tiles_range().ordinal(perm * c.tiles_range().idx(index));

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(perm_index))) {
      auto result_tile = c.find(index).get();
      auto a_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto b_tile = permute_fn(b.find(block_range.ordinal(perm_index)), perm);

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_AUTO_TEST_CASE(assign_sub_block) {
  c.fill_local(0.0);

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             2.0 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}));

  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto arg_tile = a.find(block_range.ordinal(index)).get();
    auto result_tile = c.find(block_range.ordinal(index)).get();

    BOOST_CHECK_EQUAL(result_tile.range(), arg_tile.range());

    for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
      BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             2 * (a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                  b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) &&
        !b.is_zero(block_range.ordinal(index))) {
      auto a_tile = a.find(block_range.ordinal(index)).get();
      auto b_tile = b.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (a_tile[j] + b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             2 * (3 * a("a,b,c").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    if (!a.is_zero(block_range.ordinal(index)) &&
        !b.is_zero(block_range.ordinal(index))) {
      auto a_tile = a.find(block_range.ordinal(index)).get();
      auto b_tile = b.find(block_range.ordinal(index)).get();
      auto result_tile = c.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(index));
    }
  }
}

BOOST_AUTO_TEST_CASE(assign_subblock_permute_block) {
  c.fill_local(0.0);

  Permutation perm({2, 1, 0});
  BlockRange block_range(a.trange().tiles_range(), {3, 3, 3}, {5, 5, 5});

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto result_tile = c.find(block_range.ordinal(index)).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             2 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index))) {
      auto arg_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto result_tile = c.find(block_range.ordinal(index)).get();

      // Check that the data is correct for the result array.
      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * arg_tile[j]);
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             2 * (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("a,b,c").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(index))) {
      auto result_tile = c.find(block_range.ordinal(index)).get();
      auto a_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto b_tile = b.find(block_range.ordinal(index)).get();

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c").block({3, 3, 3}, {5, 5, 5}) =
                             2 * (3 * a("c,b,a").block({3, 3, 3}, {5, 5, 5}) +
                                  4 * b("c,b,a").block({3, 3, 3}, {5, 5, 5})));

  for (std::size_t index = 0ul; index < block_range.volume(); ++index) {
    auto perm_index = perm * block_range.idx(index);

    if (!a.is_zero(block_range.ordinal(perm_index)) ||
        !b.is_zero(block_range.ordinal(perm_index))) {
      auto result_tile = c.find(block_range.ordinal(index)).get();
      auto a_tile = permute_fn(a.find(block_range.ordinal(perm_index)), perm);
      auto b_tile = permute_fn(b.find(block_range.ordinal(perm_index)), perm);

      for (std::size_t j = 0ul; j < result_tile.range().volume(); ++j) {
        BOOST_CHECK_EQUAL(result_tile[j], 2 * (3 * a_tile[j] + 4 * b_tile[j]));
      }
    } else {
      BOOST_CHECK(c.is_zero(block_range.ordinal(index)));
    }
  }
}

BOOST_AUTO_TEST_CASE(assign_subblock_block_contract) {
  w.fill_local(0.0);

  BOOST_REQUIRE_NO_THROW(w("a,b").block({3, 2}, {5, 5}) =
                             a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                             b("c,d,b").block({2, 3, 3}, {5, 5, 5}));
}

BOOST_AUTO_TEST_CASE(assign_subblock_block_permute_contract) {
  w.fill_local(0.0);

  BOOST_REQUIRE_NO_THROW(w("a,b").block({3, 2}, {5, 5}) =
                             a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                             b("d,c,b").block({3, 2, 3}, {5, 5, 5}));
}

BOOST_AUTO_TEST_CASE(block_contract) {
  BOOST_REQUIRE_NO_THROW(w("a,b") = a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                                    b("c,d,b").block({2, 3, 3}, {5, 5, 5}));
}

BOOST_AUTO_TEST_CASE(block_permute_contract) {
  BOOST_REQUIRE_NO_THROW(w("a,b") = a("a,c,d").block({3, 2, 3}, {5, 5, 5}) *
                                    b("d,c,b").block({3, 2, 3}, {5, 5, 5}));
}

BOOST_AUTO_TEST_CASE(add) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") + b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] + b_tile[j]);
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) + b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + b_tile[j]);
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] + (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c") + 3 * b("a,b,c")) +
                                      2 * (a("a,b,c") - b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], (4 * a_tile[j]) + (b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(add_to) {
  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(a("a,b,c") += b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(a_tile[j], c_tile[j] + b_tile[j]);
  }

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = a("a,b,c") + b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(a_tile[j], c_tile[j] + b_tile[j]);
  }
}

BOOST_AUTO_TEST_CASE(add_permute) {
  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) + (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) + (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(scale_add) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") + b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] + b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) + b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) + b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") + (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] + (3 * b_tile[j])));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5 * ((2 * a("a,b,c")) + (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) + (3 * b_tile[j])));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c") + 3 * b("a,b,c")) +
                                           2 * (a("a,b,c") - b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    if (!c.is_zero(i)) {
      auto c_tile = c.find(i).get();
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < c_tile.size(); ++j)
        BOOST_CHECK_EQUAL(c_tile[j], 5 * (4 * a_tile[j] + b_tile[j]));
    } else {
      BOOST_CHECK(a.is_zero(i) && b.is_zero(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_add_permute) {
  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) + (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) + (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) + (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) + (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(subt) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") - b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] - b_tile[j]);
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) - b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - b_tile[j]);
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] - (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(subt_to) {
  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(a("a,b,c") -= b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(a_tile[j], c_tile[j] - b_tile[j]);
  }

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = a("a,b,c") - b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(a_tile[j], c_tile[j] - b_tile[j]);
  }
}

BOOST_AUTO_TEST_CASE(subt_permute) {
  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) - (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) - (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(scale_subt) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") - b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] - b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) - b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) - b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") - (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] - (3 * b_tile[j])));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5 * ((2 * a("a,b,c")) - (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) - (3 * b_tile[j])));
  }
}

BOOST_AUTO_TEST_CASE(scale_subt_permute) {
  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) - (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) - (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) - (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) - (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(mult) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") * b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] * b_tile[j]);
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) * b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * b_tile[j]);
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = a("a,b,c") * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], a_tile[j] * (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("a,b,c")) * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(mult_to) {
  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(a("a,b,c") *= b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(a_tile[j], c_tile[j] * b_tile[j]);
  }

  c("a,b,c") = a("a,b,c");
  BOOST_REQUIRE_NO_THROW(a("a,b,c") = a("a,b,c") * b("a,b,c"));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(a_tile[j], c_tile[j] * b_tile[j]);
  }
}

BOOST_AUTO_TEST_CASE(mult_permute) {
  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = (2 * a("c,b,a")) * (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], (2 * a_tile[j]) * (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(scale_mult) {
  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * ((2 * a("a,b,c")) * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (a("a,b,c") * (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (a_tile[j] * (3 * b_tile[j])));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") =
                             5 * ((2 * a("a,b,c")) * (3 * b("a,b,c"))));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * ((2 * a_tile[j]) * (3 * b_tile[j])));
  }
}

BOOST_AUTO_TEST_CASE(scale_mult_permute) {
  Permutation perm({2, 1, 0});

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) * (3 * b("a,b,c")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) * (3 * b_tile[j]));
  }

  BOOST_REQUIRE_NO_THROW(c("a,b,c") = 5 * (2 * a("c,b,a")) * (3 * b("c,b,a")));

  for (std::size_t i = 0ul; i < c.size(); ++i) {
    TArrayUMD::value_type c_tile = c.find(i).get();
    const size_t perm_index =
        c.tiles_range().ordinal(perm * a.tiles_range().idx(i));
    TArrayUMD::value_type a_tile = permute_fn(a.find(perm_index), perm);
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < c_tile.size(); ++j)
      BOOST_CHECK_EQUAL(c_tile[j], 5 * (2 * a_tile[j]) * (3 * b_tile[j]));
  }
}

BOOST_AUTO_TEST_CASE(cont) {
  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  TiledArray::EigenMatrixXd left(m, k);
  left.fill(0);

  for (TArrayUMD::const_iterator it = a.begin(); it != a.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  TiledArray::EigenMatrixXd right(n, k);
  right.fill(0);

  for (TArrayUMD::const_iterator it = b.begin(); it != b.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          right(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  TiledArray::EigenMatrixXd result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * b("j,b,c"));
  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]),
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * b("j,b,c"));
  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 2,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * (3 * b("j,b,c")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 3,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * (3 * b("j,b,c")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 6,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") -= (3 * b("j,b,c")) * a("i,b,c"));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 3,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") += 2 * a("i,b,c") * b("j,b,c"));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 5,
                                   tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(cont_permute) {
  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  TiledArray::EigenMatrixXd left(m, k);
  left.fill(0);

  for (TArrayUMD::const_iterator it = a.begin(); it != a.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  TiledArray::EigenMatrixXd right(n, k);
  right.fill(0);

  for (TArrayUMD::const_iterator it = b.begin(); it != b.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  TiledArray::EigenMatrixXd result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * b("j,c,b"));
  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]),
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * b("j,c,b"));
  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 2,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("i,b,c") * (3 * b("j,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 3,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("i,b,c")) * (3 * b("j,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 6,
                                   tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(cont_permute_permute) {
  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  TiledArray::EigenMatrixXd left(m, k);
  left.fill(0);

  for (TArrayUMD::const_iterator it = a.begin(); it != a.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  TiledArray::EigenMatrixXd right(n, k);
  right.fill(0);

  for (TArrayUMD::const_iterator it = b.begin(); it != b.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  TiledArray::EigenMatrixXd result(m, n);

  result = right * left.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("j,b,c") * b("i,c,b"));
  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]),
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("j,b,c")) * b("i,c,b"));
  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 2,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = a("j,b,c") * (3 * b("i,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 3,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = (2 * a("j,b,c")) * (3 * b("i,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 6,
                                   tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_cont) {
  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  TiledArray::EigenMatrixXd left(m, k);
  left.fill(0);

  for (TArrayUMD::const_iterator it = a.begin(); it != a.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  TiledArray::EigenMatrixXd right(n, k);
  right.fill(0);

  for (TArrayUMD::const_iterator it = b.begin(); it != b.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          right(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  TiledArray::EigenMatrixXd result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * b("j,b,c")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 5,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * b("j,b,c")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 10,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * (3 * b("j,b,c"))));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 15,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * (3 * b("j,b,c"))));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 30,
                                   tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_cont_permute) {
  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  TiledArray::EigenMatrixXd left(m, k);
  left.fill(0);

  for (TArrayUMD::const_iterator it = a.begin(); it != a.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  TiledArray::EigenMatrixXd right(n, k);
  right.fill(0);

  for (TArrayUMD::const_iterator it = b.begin(); it != b.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  TiledArray::EigenMatrixXd result(m, n);

  result = left * right.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * b("j,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 5,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * b("j,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 10,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("i,b,c") * (3 * b("j,c,b"))));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 15,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("i,b,c")) * (3 * b("j,c,b"))));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 30,
                                   tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(scale_cont_permute_permute) {
  const std::size_t m = a.trange().elements_range().extent(0);
  const std::size_t k = a.trange().elements_range().extent(1) *
                        a.trange().elements_range().extent(2);
  const std::size_t n = b.trange().elements_range().extent(2);

  TiledArray::EigenMatrixXd left(m, k);
  left.fill(0);

  for (TArrayUMD::const_iterator it = a.begin(); it != a.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[1] * a.trange().elements_range().stride(1) +
                                i[2] * a.trange().elements_range().stride(2);

          left(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&left(0, 0), left.rows() * left.cols());

  TiledArray::EigenMatrixXd right(n, k);
  right.fill(0);

  for (TArrayUMD::const_iterator it = b.begin(); it != b.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 3> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      const std::size_t r = i[0];
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        for (i[2] = tile.range().lobound(2); i[2] < tile.range().upbound(2);
             ++i[2]) {
          const std::size_t c = i[2] * a.trange().elements_range().stride(1) +
                                i[1] * a.trange().elements_range().stride(2);

          right(r, c) = tile(i[0], i[1], i[2]);
        }
      }
    }
  }

  GlobalFixture::world->gop.sum(&right(0, 0), right.rows() * right.cols());

  TiledArray::EigenMatrixXd result(m, n);

  result = right * left.transpose();

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("j,b,c") * b("i,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 5,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("j,b,c")) * b("i,c,b")));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 10,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * (a("j,b,c") * (3 * b("i,c,b"))));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 15,
                                   tolerance);
      }
    }
  }

  BOOST_REQUIRE_NO_THROW(w("i,j") = 5 * ((2 * a("j,b,c")) * (3 * b("i,c,b"))));

  for (TArrayUMD::const_iterator it = w.begin(); it != w.end(); ++it) {
    TArrayUMD::value_type tile = *it;

    std::array<std::size_t, 2> i;

    for (i[0] = tile.range().lobound(0); i[0] < tile.range().upbound(0);
         ++i[0]) {
      for (i[1] = tile.range().lobound(1); i[1] < tile.range().upbound(1);
           ++i[1]) {
        BOOST_CHECK_CLOSE_FRACTION(tile(i[0], i[1]), result(i[0], i[1]) * 30,
                                   tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(cont_non_uniform1) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 40 * 5 * 5;
  const std::size_t n = 5;

  // Construct the test arguments
  TArrayUMD left(*GlobalFixture::world, trange);
  TArrayUMD right(*GlobalFixture::world, trange);

  // Construct the reference matrices
  TiledArray::EigenMatrixXd left_ref(m, k);
  TiledArray::EigenMatrixXd right_ref(n, k);

  // Initialize input
  rand_fill_matrix_and_array(left_ref, left, 23);
  rand_fill_matrix_and_array(right_ref, right, 42);

  // Compute the reference result
  TiledArray::EigenMatrixXd result_ref = 5 * left_ref * right_ref.transpose();

  // Compute the result to be tested
  TArrayUMD result;
  BOOST_REQUIRE_NO_THROW(result("x,y") =
                             5 * left("x,i,j,k") * right("y,i,j,k"));

  // Check the result
  for (TArrayUMD::iterator it = result.begin(); it != result.end(); ++it) {
    const TArrayUMD::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_CLOSE_FRACTION(result_ref.array()(elem_index), tile[*rit],
                                 tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE(cont_non_uniform2) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_1, tr1_2, tr1_2}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 5 * 40 * 40;
  const std::size_t n = 5;

  // Construct the test arguments
  TArrayUMD left(*GlobalFixture::world, trange);
  TArrayUMD right(*GlobalFixture::world, trange);

  // Construct the reference matrices
  TiledArray::EigenMatrixXd left_ref(m, k);
  TiledArray::EigenMatrixXd right_ref(n, k);

  // Initialize input
  rand_fill_matrix_and_array(left_ref, left, 23);
  rand_fill_matrix_and_array(right_ref, right, 42);

  // Compute the reference result
  TiledArray::EigenMatrixXd result_ref = 5 * left_ref * right_ref.transpose();

  // Compute the result to be tested
  TArrayUMD result;
  BOOST_REQUIRE_NO_THROW(result("x,y") =
                             5 * left("x,i,j,k") * right("y,i,j,k"));

  // Check the result
  for (TArrayUMD::iterator it = result.begin(); it != result.end(); ++it) {
    const TArrayUMD::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_CLOSE_FRACTION(result_ref.array()(elem_index), tile[*rit],
                                 tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE(cont_plus_reduce) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 40 * 5 * 5;
  const std::size_t n = 5;

  // Construct the test arrays
  TArrayUMD arg1(*GlobalFixture::world, trange);
  TArrayUMD arg2(*GlobalFixture::world, trange);
  TArrayUMD arg3(*GlobalFixture::world, trange);
  TArrayUMD arg4(*GlobalFixture::world, trange);

  // Construct the reference matrices
  TiledArray::EigenMatrixXd arg1_ref(m, k);
  TiledArray::EigenMatrixXd arg2_ref(n, k);
  TiledArray::EigenMatrixXd arg3_ref(m, k);
  TiledArray::EigenMatrixXd arg4_ref(n, k);

  // Initialize input
  rand_fill_matrix_and_array(arg1_ref, arg1, 23);
  rand_fill_matrix_and_array(arg2_ref, arg2, 42);
  rand_fill_matrix_and_array(arg3_ref, arg3, 79);
  rand_fill_matrix_and_array(arg4_ref, arg4, 19);

  // Compute the reference result
  TiledArray::EigenMatrixXd result_ref =
      2 * (arg1_ref * arg2_ref.transpose() + arg1_ref * arg4_ref.transpose() +
           arg3_ref * arg4_ref.transpose() + arg3_ref * arg2_ref.transpose());

  // Compute the result to be tested
  TArrayUMD result;
  result("x,y") = arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y") += arg1("x,i,j,k") * arg4("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y") += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y") += arg1("x,i,j,k") * arg4("y,i,j,k");

  // Check the result
  for (TArrayUMD::iterator it = result.begin(); it != result.end(); ++it) {
    const TArrayUMD::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_CLOSE_FRACTION(result_ref.array()(elem_index), tile[*rit],
                                 tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE(no_alias_plus_reduce) {
  // Construct the tiled range
  std::array<std::size_t, 6> tiling1 = {{0, 1, 2, 3, 4, 5}};
  std::array<std::size_t, 2> tiling2 = {{0, 40}};
  TiledRange1 tr1_1(tiling1.begin(), tiling1.end());
  TiledRange1 tr1_2(tiling2.begin(), tiling2.end());
  std::array<TiledRange1, 4> tiling4 = {{tr1_1, tr1_2, tr1_1, tr1_1}};
  TiledRange trange(tiling4.begin(), tiling4.end());

  const std::size_t m = 5;
  const std::size_t k = 40 * 5 * 5;
  const std::size_t n = 5;

  // Construct the test arrays
  TArrayUMD arg1(*GlobalFixture::world, trange);
  TArrayUMD arg2(*GlobalFixture::world, trange);
  TArrayUMD arg3(*GlobalFixture::world, trange);
  TArrayUMD arg4(*GlobalFixture::world, trange);

  // Construct the reference matrices
  TiledArray::EigenMatrixXd arg1_ref(m, k);
  TiledArray::EigenMatrixXd arg2_ref(n, k);
  TiledArray::EigenMatrixXd arg3_ref(m, k);
  TiledArray::EigenMatrixXd arg4_ref(n, k);

  // Initialize input
  rand_fill_matrix_and_array(arg1_ref, arg1, 23);
  rand_fill_matrix_and_array(arg2_ref, arg2, 42);
  rand_fill_matrix_and_array(arg3_ref, arg3, 79);
  rand_fill_matrix_and_array(arg4_ref, arg4, 19);

  // Compute the reference result
  TiledArray::EigenMatrixXd result_ref =
      2 * (arg1_ref * arg2_ref.transpose() + arg1_ref * arg4_ref.transpose() +
           arg3_ref * arg4_ref.transpose() + arg3_ref * arg2_ref.transpose());

  // Compute the result to be tested
  TArrayUMD result;
  result("x,y") = arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y").no_alias() += arg1("x,i,j,k") * arg4("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg1("x,i,j,k") * arg2("y,i,j,k");
  result("x,y").no_alias() += arg3("x,i,j,k") * arg4("y,i,j,k");
  result("x,y").no_alias() += arg1("x,i,j,k") * arg4("y,i,j,k");

  // Check the result
  for (TArrayUMD::iterator it = result.begin(); it != result.end(); ++it) {
    const TArrayUMD::value_type tile = *it;
    for (Range::const_iterator rit = tile.range().begin();
         rit != tile.range().end(); ++rit) {
      const std::size_t elem_index = result.elements_range().ordinal(*rit);
      BOOST_CHECK_CLOSE_FRACTION(result_ref.array()(elem_index), tile[*rit],
                                 tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_product) {
  // Test that outer product works
  BOOST_REQUIRE_NO_THROW(w("i,j") = u("i") * v("j"));

  v.make_replicated();
  u.make_replicated();
  // Generate Eigen matrices from input arrays.
  EigenMatrixXd ev = TA::array_to_eigen(v);
  EigenMatrixXd eu = TA::array_to_eigen(u);
  GlobalFixture::world->gop.fence();

  // Generate the expected result
  EigenMatrixXd ew_test = eu * ev.transpose();

  w.make_replicated();
  GlobalFixture::world->gop.fence();

  EigenMatrixXd ew = TA::array_to_eigen(w);

  BOOST_CHECK_EQUAL(ew, ew_test);
}

BOOST_AUTO_TEST_CASE(dot) {
  // Test the dot expression function
  double result = 0;
  BOOST_REQUIRE_NO_THROW(result = static_cast<double>(a("a,b,c") * b("a,b,c")));
  BOOST_REQUIRE_NO_THROW(result += a("a,b,c") * b("a,b,c"));
  BOOST_REQUIRE_NO_THROW(result -= a("a,b,c") * b("a,b,c"));
  BOOST_REQUIRE_NO_THROW(result *= a("a,b,c") * b("a,b,c"));
  BOOST_REQUIRE_NO_THROW(result = a("a,b,c").dot(b("a,b,c")).get());

  // Compute the expected value for the dot function.
  double expected = 0;
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    TArrayUMD::value_type a_tile = a.find(i).get();
    TArrayUMD::value_type b_tile = b.find(i).get();

    for (std::size_t j = 0ul; j < a_tile.size(); ++j)
      expected += a_tile[j] * b_tile[j];
  }

  // Check the result of dot
  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(
      result = (a("a,b,c") - b("a,b,c")).dot((a("a,b,c") + b("a,b,c"))).get());
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) && !b.is_zero(i)) {
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(result = (2 * a("a,b,c")).dot(3 * b("a,b,c")).get());
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) && !b.is_zero(i)) {
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * (a_tile[j] * b_tile[j]);
    }
  }

  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(
      result =
          2 *
          (a("a,b,c") - b("a,b,c")).dot(3 * (a("a,b,c") + b("a,b,c"))).get());
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    if (!a.is_zero(i) && !b.is_zero(i)) {
      auto a_tile = a.find(i).get();
      auto b_tile = b.find(i).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);
}

BOOST_AUTO_TEST_CASE(dot_permute) {
  Permutation perm({2, 1, 0});
  // Test the dot expression function
  double result = 0;
  BOOST_REQUIRE_NO_THROW(result = static_cast<double>(a("a,b,c") * b("c,b,a")));
  BOOST_REQUIRE_NO_THROW(result += a("a,b,c") * b("c,b,a"));
  BOOST_REQUIRE_NO_THROW(result -= a("a,b,c") * b("c,b,a"));
  BOOST_REQUIRE_NO_THROW(result *= a("a,b,c") * b("c,b,a"));
  BOOST_REQUIRE_NO_THROW(result = a("a,b,c").dot(b("c,b,a")).get());

  // Compute the expected value for the dot function.
  double expected = 0;
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    TArrayUMD::value_type a_tile = a.find(i).get();
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    TArrayUMD::value_type b_tile = permute_fn(b.find(perm_index), perm);

    for (std::size_t j = 0ul; j < a_tile.size(); ++j)
      expected += a_tile[j] * b_tile[j];
  }

  // Check the result of dot
  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(
      result = (a("a,b,c") - b("c,b,a")).dot(a("a,b,c") + b("c,b,a")).get());

  // Compute the expected value for the dot function.
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) && !b.is_zero(perm_index)) {
      auto a_tile = a.find(i).get();
      auto b_tile = perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  // Check the result of dot
  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(result = (2 * a("a,b,c")).dot(3 * b("c,b,a")).get());

  // Compute the expected value for the dot function.
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) && !b.is_zero(perm_index)) {
      auto a_tile = a.find(i).get();
      auto b_tile = perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * a_tile[j] * b_tile[j];
    }
  }

  // Check the result of dot
  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);

  result = 0;
  expected = 0;
  BOOST_REQUIRE_NO_THROW(result = (2 * (a("a,b,c") - b("c,b,a")))
                                      .dot(3 * (a("a,b,c") + b("c,b,a")))
                                      .get());

  // Compute the expected value for the dot function.
  for (std::size_t i = 0ul; i < a.size(); ++i) {
    const size_t perm_index =
        a.tiles_range().ordinal(perm * b.tiles_range().idx(i));
    if (!a.is_zero(i) && !b.is_zero(perm_index)) {
      auto a_tile = a.find(i).get();
      auto b_tile = perm * b.find(perm_index).get();

      for (std::size_t j = 0ul; j < a_tile.size(); ++j)
        expected += 6 * (a_tile[j] - b_tile[j]) * (a_tile[j] + b_tile[j]);
    }
  }

  // Check the result of dot
  BOOST_CHECK_CLOSE_FRACTION(result, expected, tolerance);
}

BOOST_AUTO_TEST_CASE(dot_contr) {
  for (int i = 0; i != 3; ++i)
    BOOST_REQUIRE_NO_THROW(
        (a("a,b,c") * b("d,b,c")).dot(b("d,e,f") * a("a,e,f")));
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_CUDA
