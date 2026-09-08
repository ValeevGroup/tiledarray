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
 */

#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct ForeachFixture : public TiledRangeFixture {
  ForeachFixture()
      : a(*GlobalFixture::world, tr,
          std::make_shared<detail::HashPmap>(*GlobalFixture::world,
                                             tr.tiles_range().volume(), 1)),
        b(*GlobalFixture::world, tr,
          std::make_shared<detail::HashPmap>(*GlobalFixture::world,
                                             tr.tiles_range().volume(), 2)),
        c(*GlobalFixture::world, tr, make_shape(tr, 0.50, 42),
          std::make_shared<detail::HashPmap>(*GlobalFixture::world,
                                             tr.tiles_range().volume(), 3)),
        d(*GlobalFixture::world, tr, make_shape(tr, 0.50, 16),
          std::make_shared<detail::HashPmap>(*GlobalFixture::world,
                                             tr.tiles_range().volume(), 4)) {
    random_fill(a);
    random_fill(b);
    random_fill(c);
    random_fill(d);
    GlobalFixture::world->gop.fence();
  }

  template <typename Tile, typename Policy>
  static void random_fill(DistArray<Tile, Policy>& array) {
    for (auto index : *array.pmap()) {
      if (array.is_zero(index)) continue;
      array.set(index, array.world().taskq.add(
                           &ForeachFixture::template make_rand_tile<
                               DistArray<Tile, Policy> >,
                           array.trange().make_tile_range(index)));
    }
  }

  template <typename T>
  static void set_random(T& t) {
    t = GlobalFixture::world->rand() % 101;
  }

  // Fill a tile with random data
  template <typename A>
  static typename A::value_type make_rand_tile(
      const typename A::value_type::range_type& r) {
    typename A::value_type tile(r);
    for (std::size_t i = 0ul; i < tile.size(); ++i) set_random(tile[i]);
    return tile;
  }

  static Tensor<float> make_norm_tensor(const TiledRange& trange,
                                        const float fill_percent,
                                        const int seed) {
    GlobalFixture::world->srand(seed);
    Tensor<float> norms(trange.tiles_range());
    for (Tensor<float>::size_type i = 0ul; i < norms.size(); ++i) {
      const Range range = trange.make_tile_range(i);
      norms[i] = (GlobalFixture::world->rand() % 101);
      norms[i] = std::sqrt(norms[i] * norms[i] * range.volume());
    }

    const std::size_t n = float(norms.size()) * (1.0 - fill_percent);
    for (std::size_t i = 0ul; i < n; ++i) {
      norms[GlobalFixture::world->rand() % norms.size()] =
          SparseShape<float>::threshold() * 0.1;
    }

    return norms;
  }

  static SparseShape<float> make_shape(const TiledRange& trange,
                                       const float fill_percent,
                                       const int seed) {
    Tensor<float> tile_norms = make_norm_tensor(trange, fill_percent, seed);
    return SparseShape<float>(tile_norms, trange);
  }

  ~ForeachFixture() { GlobalFixture::world->gop.fence(); }

  TArrayI a;
  TArrayI b;
  TSpArrayI c;
  TSpArrayI d;
};  // ForeachFixture

BOOST_FIXTURE_TEST_SUITE(foreach_suite, ForeachFixture)

BOOST_AUTO_TEST_CASE(foreach_unary) {
  TArrayI result = foreach (
      a, [](TensorI& result, const TensorI& arg) { result = arg.scale(2); });

  for (auto index : *result.pmap()) {
    TensorI tile0 = a.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], 2 * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_w_idx) {

  TArrayI result = a.clone();
  foreach_inplace(result, [](TensorI& tile, const Range::index_type &coord_idx) {
    long fac = (coord_idx[0] < coord_idx[1]) ? coord_idx[0] : coord_idx[1];
    tile[coord_idx] = fac * tile[coord_idx];
  }, true);

  for (auto index : *result.pmap()) {
    TensorI tile0 = a.find(index).get();
    TensorI tile = result.find(index).get();
    const Range &range = tile0.range();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      const Range::index_type &coord_idx = range.idx(i);
      long fac = coord_idx[0] < coord_idx[1] ? coord_idx[0] : coord_idx[1];
      BOOST_CHECK_EQUAL(tile[i], fac * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_unary_sparse) {
  TSpArrayI result =
      foreach (c, [](TensorI& result, const TensorI& arg) -> float {
        result = arg.scale(2);
        return result.norm();
      });

  for (auto index : *result.pmap()) {
    if (c.is_zero(index)) continue;

    TensorI tile0 = c.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], 2 * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_unary_to_double) {
  TArrayD result = foreach<TensorD>(a, [](TensorD& result, const TensorI& arg) {
    result = TensorD(arg, [](int val) -> double { return 2.0 * double(val); });
  });

  for (auto index : *result.pmap()) {
    TensorI tile0 = a.find(index).get();
    TensorD tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(int(tile[i]), 2 * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_unary_sparse_to_double) {
  TSpArrayD result =
      foreach<TensorD>(c, [](TensorD& result, const TensorI& arg) -> float {
        result =
            TensorD(arg, [](int val) -> double { return 2.0 * double(val); });
        return result.norm();
      });

  for (auto index : *result.pmap()) {
    if (c.is_zero(index)) continue;

    TensorI tile0 = c.find(index).get();
    TensorD tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(int(tile[i]), 2 * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_unary_inplace) {
  TArrayI result = a.clone();
  foreach_inplace(result, [](TensorI& arg) { arg.scale_to(2); });

  for (auto index : *result.pmap()) {
    TensorI tile0 = a.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], 2 * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_unary_sparse_inplace) {
  TSpArrayI result = c.clone();
  foreach_inplace(result, [](TensorI& arg) {
    arg.scale_to(2);
    return arg.norm<float>();
  });

  for (auto index : *result.pmap()) {
    if (c.is_zero(index)) continue;

    TensorI tile0 = c.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], 2 * tile0[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_binary) {
  TArrayI result =
      foreach (a, b, [](TensorI& result, const TensorI& l, const TensorI& r) {
        result = l.add(r);
      });

  for (auto index : *result.pmap()) {
    TensorI tilea = a.find(index).get();
    TensorI tileb = b.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], tilea[i] + tileb[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_binary_sparse) {
  TSpArrayI result = foreach (
      c, d, [](TensorI& result, const TensorI& l, const TensorI& r) -> float {
        result = l.add(r);
        return result.norm();
      });

  for (auto index : *result.pmap()) {
    if (result.is_zero(index)) continue;

    TensorI tilec = c.find(index).get();
    TensorI tiled = d.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], tilec[i] + tiled[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_binary_to_double) {
  TArrayD result = foreach<TensorD>(
      a, b, [](TensorD& result, const TensorI& l, const TensorI& r) {
        result = TensorD(l, r, [](int l, int r) -> double { return l + r; });
      });

  for (auto index : *result.pmap()) {
    TensorI tilea = a.find(index).get();
    TensorI tileb = b.find(index).get();
    auto tile = TensorI(result.find(index).get());
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], tilea[i] + tileb[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_binary_sparse_to_double) {
  TSpArrayD result = foreach<TensorD>(
      c, d, [](TensorD& result, const TensorI& l, const TensorI& r) -> float {
        result = TensorD(l, r, [](int l, int r) -> double { return l + r; });
        return result.norm();
      });

  for (auto index : *result.pmap()) {
    if (result.is_zero(index)) continue;

    TensorI tilec = c.find(index).get();
    TensorI tiled = d.find(index).get();
    TensorD tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], tilec[i] + tiled[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_binary_inplace) {
  TArrayI result = a.clone();
  foreach_inplace(result, b, [](TensorI& l, const TensorI& r) { l.add_to(r); });

  for (auto index : *result.pmap()) {
    TensorI tilea = a.find(index).get();
    TensorI tileb = b.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], tilea[i] + tileb[i]);
    }
  }
}

BOOST_AUTO_TEST_CASE(foreach_binary_sparse_inplace) {
  TSpArrayI result = c.clone();
  foreach_inplace(result, d, [](TensorI& l, const TensorI& r) -> float {
    l.add_to(r);
    return l.norm();
  });

  for (auto index : *result.pmap()) {
    if (result.is_zero(index)) continue;

    TensorI tilec = c.find(index).get();
    TensorI tiled = d.find(index).get();
    TensorI tile = result.find(index).get();
    for (std::size_t i = 0; i < tile.size(); ++i) {
      BOOST_CHECK_EQUAL(tile[i], tilec[i] + tiled[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
