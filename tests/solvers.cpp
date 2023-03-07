/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  utils.h
 *  May 20, 2013
 *
 */

#include <TiledArray/math/linalg/conjgrad.h>
#include <tiledarray.h>

#include "unit_test_config.h"

using namespace TiledArray;

typedef boost::mpl::list<TArrayD, TSpArrayD> array_types;

template <typename T>
struct make_Ax {
  struct Ax {
    T operator()(const T& x) const;
  };
  Ax operator()() const { return Ax{}(); }
};

template <typename T>
struct make_b /* {
  T operator()() const;
} */;

template <typename T>
struct make_pc /* {
  T operator()() const;
} */;

template <typename T>
struct validate /* {
  bool operator()() const;
} */;

template <typename Tile, typename Policy>
struct make_Ax<DistArray<Tile, Policy>> {
  using T = DistArray<Tile, Policy>;

  struct Ax {
    static auto make_tile(const Range& range) {
      // Construct a tile
      typename T::value_type tile(range);

      // Fill tile with data
      tile(0, 0) = 1;
      tile(0, 1) = 2;
      tile(0, 2) = 3;
      tile(1, 0) = 2;
      tile(1, 1) = 5;
      tile(1, 2) = 8;
      tile(2, 0) = 3;
      tile(2, 1) = 8;
      tile(2, 2) = 15;

      return tile;
    }

    Ax()
        : A_(TA::get_default_world(),
             TiledRange({TiledRange1{0, 3}, TiledRange1{0, 3}})) {
      if (A_.is_local(0))
        A_.set(0, A_.world().taskq.add(&Ax::make_tile,
                                       A_.trange().make_tile_range(0)));
    }
    void operator()(const T& x, T& result) const {
      T Ax;
      Ax("p") = A_("p,q") * x("q");
      result = Ax;
    }
    T A_;
  };
  Ax operator()() const { return Ax{}; }
};

template <typename Tile, typename Policy>
struct make_b<DistArray<Tile, Policy>> {
  using T = DistArray<Tile, Policy>;

  static auto make_tile(const Range& range) {
    // Construct a tile
    typename T::value_type tile(range);

    // Fill tile with data
    tile({0}) = 1;
    tile({1}) = 4;
    tile({2}) = 0;

    return tile;
  }

  T operator()() const {
    T result(get_default_world(), TiledRange{TiledRange1{0, 3}});
    if (result.is_local(0))
      result.set(0,
                 result.world().taskq.add(&make_b::make_tile,
                                          result.trange().make_tile_range(0)));
    return result;
  }
};

template <typename Tile, typename Policy>
struct make_pc<DistArray<Tile, Policy>> {
  using T = DistArray<Tile, Policy>;

  static auto make_tile(const Range& range) {
    // Construct a tile
    typename T::value_type tile(range);

    // Fill tile with data
    tile({0}) = 1;
    tile({1}) = 1;
    tile({2}) = 1;

    return tile;
  }

  T operator()() const {
    T result(get_default_world(), TiledRange{TiledRange1{0, 3}});
    if (result.is_local(0))
      result.set(0,
                 result.world().taskq.add(&make_pc::make_tile,
                                          result.trange().make_tile_range(0)));
    return result;
  }
};

template <typename Tile, typename Policy>
struct validate<DistArray<Tile, Policy>> {
  using T = DistArray<Tile, Policy>;

  bool operator()(const T& x) const {
    if (x.is_local({0})) {
      double tile_0_ref[] = {-6.5, 9., -3.5};
      const auto& tile_0 = x.find({0}).get();
      double error_2norm = 0.0;
      for (int i = 0; i != 3; ++i) {
        auto delta = tile_0({i}) - tile_0_ref[i];
        error_2norm += delta * delta;
      }
      error_2norm = std::sqrt(error_2norm);
      if (error_2norm > 1e-11) return false;
    }
    return true;
  }
};

BOOST_AUTO_TEST_SUITE(solvers)

BOOST_AUTO_TEST_CASE_TEMPLATE(conjugate_gradient, Array, array_types) {
  auto Ax = make_Ax<Array>{}();
  auto b = make_b<Array>{}();
  auto pc = make_pc<Array>{}();
  Array x;
  ConjugateGradientSolver<Array, decltype(Ax)>{}(Ax, b, x, pc, 1e-11);
  BOOST_CHECK(validate<Array>{}(x));
}

BOOST_AUTO_TEST_SUITE_END()
