/// Unified tensor-of-tensors construction: detail::make_nested_tile,
/// DistArray::init_tiles_nested, and the DistArray ToT range_fn constructor --
/// exercised identically for TA::Tensor and ArenaTensor inner tiles.

#include "TiledArray/einsum/tiledarray.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "tiledarray.h"

#include "global_fixture.h"
#include "unit_test_config.h"

#include <complex>
#include <cstddef>
#include <vector>

namespace {

namespace TA = TiledArray;

/// Deliberately non-uniform inner extent keyed on the outer element index.
inline long inner_extent(long e) { return 2 + (e % 3); }

/// Build a rank-1 inner range of the inner tile's range type.
template <typename InnerTile, typename Index>
auto inner_range_for(const Index& idx) {
  return
      typename InnerTile::range_type{inner_extent(static_cast<long>(idx[0]))};
}

/// Build a rank-2 (d0 x d1) inner range of the inner tile's range type.
/// Works for both TA::Range (TA::Tensor inner) and btas::zb::RangeNd
/// (ArenaTensor inner), which are both constructible from an extent vector.
template <typename InnerTile>
auto inner_range_2d(std::size_t d0, std::size_t d1) {
  return typename InnerTile::range_type(std::vector<std::size_t>{d0, d1});
}

// Assert an arena ToT outer tile is a single contiguous page: uniform-size
// cells at constant page-jump-free stride. classify_run returns 0 when clean,
// 3 on a page jump (multi-page build).
template <typename OuterTile>
static void check_single_page_uniform(const OuterTile& tile) {
  const std::size_t n = tile.range().volume();
  const int code = TA::detail::classify_run(
      [&](std::size_t i) -> decltype(auto) { return tile.data()[i]; }, n);
  BOOST_CHECK_EQUAL(code, 0);  // 0 == clean single-page constant stride
}

template <typename InnerTile>
void verify_cell(const InnerTile& cell, long e, bool expect_filled) {
  BOOST_REQUIRE(!cell.empty());
  BOOST_CHECK_EQUAL(static_cast<long>(cell.size()), inner_extent(e));
  for (std::size_t i = 0; i < cell.size(); ++i) {
    const double expect = expect_filled ? (100.0 * e + i) : 0.0;
    BOOST_CHECK_EQUAL(cell.data()[i], expect);
  }
}

/// Fill an inner cell so element i of outer element e holds 100*e + i.
template <typename Cell, typename Index>
void fill_cell(Cell& cell, const Index& idx) {
  const long e = static_cast<long>(idx[0]);
  for (std::size_t i = 0; i < cell.size(); ++i) cell.data()[i] = 100.0 * e + i;
}

template <typename InnerTile>
void test_make_nested_tile() {
  using OuterTile = TA::Tensor<InnerTile>;
  const TA::Range outer{4};
  OuterTile tile = TA::detail::make_nested_tile<OuterTile>(
      outer, [](const auto& idx) { return inner_range_for<InnerTile>(idx); },
      [](auto& cell, const auto& idx) { fill_cell(cell, idx); });
  BOOST_REQUIRE_EQUAL(tile.range().volume(), 4u);
  for (long e = 0; e < 4; ++e)
    verify_cell(tile.data()[e], e, /*expect_filled=*/true);
}

template <typename InnerTile, typename Policy>
void test_dist_array_tot_ctor() {
  using Array = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  // ToT range_fn ctor: shapes every inner cell, storage zero-initialized.
  Array a(world, trange,
          [](const auto& idx) { return inner_range_for<InnerTile>(idx); });
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      verify_cell(tile.data()[ord], e, /*expect_filled=*/false);
    }
  }
}

template <typename InnerTile, typename Policy>
void test_init_tiles_nested() {
  using Array = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  Array a(world, trange);
  a.init_tiles_nested(
      [](const auto& idx) { return inner_range_for<InnerTile>(idx); },
      [](auto& cell, const auto& idx) { fill_cell(cell, idx); });
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      verify_cell(tile.data()[ord], e, /*expect_filled=*/true);
    }
  }
}

/// fill_random on an already-shaped ToT array is an in-place scalar mutator:
/// it overwrites every inner scalar while leaving the inner ranges intact.
template <typename InnerTile, typename Policy>
void test_fill_random() {
  using Array = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  Array a(world, trange,
          [](const auto& idx) { return inner_range_for<InnerTile>(idx); });
  a.fill_random();
  double sum = 0.0;
  std::size_t ncells = 0;
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      const auto& cell = tile.data()[ord];
      // inner ranges must survive the in-place fill
      BOOST_REQUIRE(!cell.empty());
      BOOST_CHECK_EQUAL(static_cast<long>(cell.size()), inner_extent(e));
      for (std::size_t i = 0; i < cell.size(); ++i) sum += cell.data()[i];
      ++ncells;
    }
  }
  BOOST_REQUIRE_GT(ncells, 0u);
  // a random fill leaving every scalar exactly 0 is a measure-zero event
  BOOST_CHECK_NE(sum, 0.0);
}

/// init_elements drives the ToT constructor with an op that yields freestanding
/// owning inner tensors; for arena inners each outer tile collects the op
/// outputs, sizes one slab to fit, and deep-copies into the bound cells.
template <typename InnerTile, typename Policy>
void test_init_elements() {
  using Array = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  Array a(world, trange);
  a.init_elements([](const auto& idx) {
    const long e = static_cast<long>(idx[0]);
    TA::Tensor<double> t{TA::Range(inner_extent(e))};
    for (std::size_t i = 0; i < t.size(); ++i) t.data()[i] = 100.0 * e + i;
    return t;
  });
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      verify_cell(tile.data()[ord], e, /*expect_filled=*/true);
    }
  }
}

/// fill on an already-shaped (uniform-extent) arena ToT deep-copies a
/// freestanding owning tensor into every bound inner cell.
void test_fill_arena() {
  using InnerTile = TA::ArenaTensor<double>;
  using Array = TA::DistArray<TA::Tensor<InnerTile>, TA::DensePolicy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  const long ext = 3;
  Array a(world, trange,
          [ext](const auto&) { return typename InnerTile::range_type{ext}; });
  TA::Tensor<double> value{TA::Range(ext)};
  for (std::size_t i = 0; i < value.size(); ++i) value.data()[i] = 7.0 + i;
  a.fill(value);
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const auto& cell = tile.data()[ord];
      BOOST_REQUIRE(!cell.empty());
      BOOST_CHECK_EQUAL(static_cast<long>(cell.size()), ext);
      for (std::size_t i = 0; i < cell.size(); ++i)
        BOOST_CHECK_EQUAL(cell.data()[i], 7.0 + i);
    }
  }
}

/// set(i, value) populates a tile of an unshaped arena ToT array: every inner
/// cell is sized to `value`'s range and deep-copies its data.
void test_set_value_arena() {
  using InnerTile = TA::ArenaTensor<double>;
  using Array = TA::DistArray<TA::Tensor<InnerTile>, TA::DensePolicy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  const long ext = 3;
  // harvest a populated inner cell from a shaped source array
  Array src(world, trange,
            [ext](const auto&) { return typename InnerTile::range_type{ext}; });
  InnerTile value;
  for (const auto& tidx : src.trange().tiles_range()) {
    if (!src.is_local(tidx)) continue;
    auto tile = src.find(tidx).get();
    value = tile.data()[0];  // null -> rebind: view src's cell 0
    for (std::size_t i = 0; i < value.size(); ++i) value.data()[i] = 10.0 + i;
    break;
  }
  BOOST_REQUIRE(!value.empty());
  // populate a fresh, unshaped array tile-by-tile
  Array a(world, trange);
  for (const auto& tidx : a.trange().tiles_range())
    if (a.is_local(tidx)) a.set(tidx, value);
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const auto& cell = tile.data()[ord];
      BOOST_REQUIRE(!cell.empty());
      BOOST_CHECK_EQUAL(static_cast<long>(cell.size()), ext);
      for (std::size_t i = 0; i < cell.size(); ++i)
        BOOST_CHECK_EQUAL(cell.data()[i], 10.0 + i);
    }
  }
}

/// set(i, InIter) populates a tile from a sequence of freestanding owning
/// inner tensors; the slab is sized from their (possibly non-uniform) ranges.
void test_set_iter_arena() {
  using InnerTile = TA::ArenaTensor<double>;
  using Array = TA::DistArray<TA::Tensor<InnerTile>, TA::DensePolicy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  Array a(world, trange);
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    const auto tr = a.trange().make_tile_range(tidx);
    std::vector<TA::Tensor<double>> cells;
    for (std::size_t ord = 0; ord < tr.volume(); ++ord) {
      const long e = static_cast<long>(tr.idx(ord)[0]);
      TA::Tensor<double> c{TA::Range(inner_extent(e))};
      for (std::size_t i = 0; i < c.size(); ++i) c.data()[i] = 100.0 * e + i;
      cells.push_back(c);
    }
    a.set(tidx, cells.begin());
  }
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) continue;
    auto tile = a.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      verify_cell(tile.data()[ord], e, /*expect_filled=*/true);
    }
  }
}

/// Distributed: fetching an arena ToT tile owned by another rank transports
/// it via madness::archive, exercising Tensor<ArenaTensor>'s arena-aware
/// serialization end-to-end (slab marshalled out, rebuilt on the receiver).
void test_distributed_arena_tot() {
  using InnerTile = TA::ArenaTensor<double>;
  using Array = TA::DistArray<TA::Tensor<InnerTile>, TA::DensePolicy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4, 6, 8, 10, 12, 14}};  // 7 outer tiles
  Array a(world, trange);
  a.init_tiles_nested(
      [](const auto& idx) { return inner_range_for<InnerTile>(idx); },
      [](auto& cell, const auto& idx) { fill_cell(cell, idx); });
  world.gop.fence();
  std::size_t nremote = 0;
  for (const auto& tidx : a.trange().tiles_range()) {
    if (!a.is_local(tidx)) ++nremote;
    auto tile = a.find(tidx).get();  // remote tile -> serialized transfer
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      verify_cell(tile.data()[ord], e, /*expect_filled=*/true);
    }
  }
  // with >1 rank at least one tile must have been fetched (transported) here
  if (world.size() > 1) BOOST_CHECK_GT(nremote, 0u);
  world.gop.fence();
}

/// DistArray-level expression on a tensor-of-tensors, for plain and arena
/// inner tiles. `fill_a`/`fill_b` populate operands a/b; `expr(c, a, b)`
/// evaluates the expression under test; `expected(e, i)` is the reference
/// value for element i of outer element e.
template <typename InnerTile, typename Policy, typename FillA, typename FillB,
          typename Expr, typename Expected>
void run_tot_expr(FillA fill_a, FillB fill_b, Expr expr, Expected expected) {
  using Array = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}};
  Array a(world, trange), b(world, trange);
  auto range_fn = [](const auto& idx) {
    return inner_range_for<InnerTile>(idx);
  };
  a.init_tiles_nested(range_fn, fill_a);
  b.init_tiles_nested(range_fn, fill_b);
  world.gop.fence();
  Array c;
  expr(c, a, b);
  world.gop.fence();
  for (const auto& tidx : c.trange().tiles_range()) {
    if (!c.is_local(tidx)) continue;
    auto tile = c.find(tidx).get();
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const long e = static_cast<long>(tile.range().idx(ord)[0]);
      const auto& cell = tile.data()[ord];
      BOOST_REQUIRE(!cell.empty());
      BOOST_CHECK_EQUAL(static_cast<long>(cell.size()), inner_extent(e));
      for (std::size_t i = 0; i < cell.size(); ++i)
        BOOST_CHECK_EQUAL(cell.data()[i], expected(e, static_cast<long>(i)));
    }
  }
}

/// c = a + b, element-wise over matching inner cells.
template <typename InnerTile, typename Policy>
void test_tot_add() {
  auto fill = [](auto& cell, const auto& idx) { fill_cell(cell, idx); };
  run_tot_expr<InnerTile, Policy>(
      fill, fill,
      [](auto& c, auto& a, auto& b) { c("i;j") = a("i;j") + b("i;j"); },
      [](long e, long i) { return 2.0 * (100.0 * e + i); });
}

/// c = a - b, element-wise over matching inner cells.
template <typename InnerTile, typename Policy>
void test_tot_subt() {
  auto fill_a = [](auto& cell, const auto& idx) {
    const long e = static_cast<long>(idx[0]);
    for (std::size_t i = 0; i < cell.size(); ++i)
      cell.data()[i] = 300.0 * e + 2.0 * i;
  };
  auto fill_b = [](auto& cell, const auto& idx) { fill_cell(cell, idx); };
  run_tot_expr<InnerTile, Policy>(
      fill_a, fill_b,
      [](auto& c, auto& a, auto& b) { c("i;j") = a("i;j") - b("i;j"); },
      [](long e, long i) { return 200.0 * e + i; });
}

/// c = a * b, full Hadamard (outer and inner) over matching inner cells.
template <typename InnerTile, typename Policy>
void test_tot_mult() {
  auto fill_a = [](auto& cell, const auto& idx) {
    const long e = static_cast<long>(idx[0]);
    for (std::size_t i = 0; i < cell.size(); ++i)
      cell.data()[i] = static_cast<double>(e + static_cast<long>(i) + 1);
  };
  auto fill_b = [](auto& cell, const auto&) {
    for (std::size_t i = 0; i < cell.size(); ++i) cell.data()[i] = 3.0;
  };
  run_tot_expr<InnerTile, Policy>(
      fill_a, fill_b,
      [](auto& c, auto& a, auto& b) { c("i;j") = a("i;j") * b("i;j"); },
      [](long e, long i) { return 3.0 * (e + i + 1); });
}

/// c = 3 * a, scalar scaling over inner cells.
template <typename InnerTile, typename Policy>
void test_tot_scale() {
  auto fill = [](auto& cell, const auto& idx) { fill_cell(cell, idx); };
  run_tot_expr<InnerTile, Policy>(
      fill, fill, [](auto& c, auto& a, auto&) { c("i;j") = 3.0 * a("i;j"); },
      [](long e, long i) { return 3.0 * (100.0 * e + i); });
}

/// c = 3 * (a + b); exercises the scaled-add tile op (add with a factor).
template <typename InnerTile, typename Policy>
void test_tot_scaled_add() {
  auto fill = [](auto& cell, const auto& idx) { fill_cell(cell, idx); };
  run_tot_expr<InnerTile, Policy>(
      fill, fill,
      [](auto& c, auto& a, auto& b) { c("i;j") = 3.0 * (a("i;j") + b("i;j")); },
      [](long e, long i) { return 6.0 * (100.0 * e + i); });
}

/// c = 3 * (a - b); exercises the scaled-subt tile op (subt with a factor).
template <typename InnerTile, typename Policy>
void test_tot_scaled_subt() {
  auto fill_a = [](auto& cell, const auto& idx) {
    const long e = static_cast<long>(idx[0]);
    for (std::size_t i = 0; i < cell.size(); ++i)
      cell.data()[i] = 300.0 * e + 2.0 * i;
  };
  auto fill_b = [](auto& cell, const auto& idx) { fill_cell(cell, idx); };
  run_tot_expr<InnerTile, Policy>(
      fill_a, fill_b,
      [](auto& c, auto& a, auto& b) { c("i;j") = 3.0 * (a("i;j") - b("i;j")); },
      [](long e, long i) { return 3.0 * (200.0 * e + i); });
}

/// c = -a, negation over inner cells.
template <typename InnerTile, typename Policy>
void test_tot_neg() {
  auto fill = [](auto& cell, const auto& idx) { fill_cell(cell, idx); };
  run_tot_expr<InnerTile, Policy>(
      fill, fill, [](auto& c, auto& a, auto&) { c("i;j") = -a("i;j"); },
      [](long e, long i) { return -(100.0 * e + i); });
}

/// End-to-end ToT contraction through TA::einsum: outer Hadamard over i,j
/// with an outer contraction over k, plus an inner contraction. This routes
/// through the regime-A arena einsum path (the outer-Hadamard "hadamard
/// reduction" branch), not the expression-DSL delegation a pure-Hadamard
/// outer would take. `annot` is the einsum string; a's inner cells are
/// `a0 x a1`, b's are `b0 x b1`. A non-canonical inner annotation exercises
/// the inner-permutation hoist. The arena-inner result is checked against a
/// Tensor<Tensor<double>> reference run of the identical expression.
template <typename InnerTile, typename Policy>
void test_tot_einsum_contraction(const char* annot, std::size_t a0,
                                 std::size_t a1, std::size_t b0,
                                 std::size_t b1) {
  using Array = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  using RefArray = TA::DistArray<TA::Tensor<TA::Tensor<double>>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}, {0, 2, 4}, {0, 2}};

  auto fill_a = [](auto& cell, const auto& idx) {
    const long key = 7 * static_cast<long>(idx[0]) +
                     13 * static_cast<long>(idx[1]) +
                     31 * static_cast<long>(idx[2]);
    for (std::size_t p = 0; p < cell.size(); ++p)
      cell.data()[p] = static_cast<double>(1 + static_cast<long>(p) + key);
  };
  auto fill_b = [](auto& cell, const auto& idx) {
    const long key = 5 * static_cast<long>(idx[0]) +
                     3 * static_cast<long>(idx[1]) +
                     11 * static_cast<long>(idx[2]);
    for (std::size_t p = 0; p < cell.size(); ++p)
      cell.data()[p] = static_cast<double>(2 + static_cast<long>(p) + key);
  };

  Array a(world, trange), b(world, trange);
  a.init_tiles_nested(
      [a0, a1](const auto&) { return inner_range_2d<InnerTile>(a0, a1); },
      fill_a);
  b.init_tiles_nested(
      [b0, b1](const auto&) { return inner_range_2d<InnerTile>(b0, b1); },
      fill_b);
  RefArray a_ref(world, trange), b_ref(world, trange);
  a_ref.init_tiles_nested(
      [a0, a1](const auto&) {
        return inner_range_2d<TA::Tensor<double>>(a0, a1);
      },
      fill_a);
  b_ref.init_tiles_nested(
      [b0, b1](const auto&) {
        return inner_range_2d<TA::Tensor<double>>(b0, b1);
      },
      fill_b);
  world.gop.fence();

  auto c = TA::einsum(annot, a, b);
  auto c_ref = TA::einsum(annot, a_ref, b_ref);
  world.gop.fence();

  for (const auto& tidx : c.trange().tiles_range()) {
    if (!c.is_local(tidx)) continue;
    auto tile = c.find(tidx).get();
    auto ref_tile = c_ref.find(tidx).get();
    BOOST_REQUIRE_EQUAL(tile.range().volume(), ref_tile.range().volume());
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const auto& cell = tile.data()[ord];
      const auto& ref_cell = ref_tile.data()[ord];
      BOOST_REQUIRE(!cell.empty());
      BOOST_REQUIRE_EQUAL(cell.size(), ref_cell.size());
      for (std::size_t p = 0; p < cell.size(); ++p)
        BOOST_CHECK_EQUAL(cell.data()[p], ref_cell.data()[p]);
    }
  }
}

/// End-to-end T x ToT Hadamard through TA::einsum: a plain DistArray scales
/// each inner cell of a ToT array. A pure-Hadamard outer ("ij,ij;a->ij;a")
/// makes einsum delegate to the expression DSL, exercising the arena
/// `t x tot` Mult tile op. The arena-inner result is checked against an
/// identical Tensor<double>-inner reference run (the legacy `binary` path).
template <typename InnerTile, typename Policy>
void test_tot_einsum_t_x_tot() {
  using ToTArray = TA::DistArray<TA::Tensor<InnerTile>, Policy>;
  using RefArray = TA::DistArray<TA::Tensor<TA::Tensor<double>>, Policy>;
  using PlainArray = TA::DistArray<TA::Tensor<double>, Policy>;
  TA::World& world = *GlobalFixture::world;
  TA::TiledRange trange{{0, 2, 4}, {0, 2, 4}};

  auto plain_fill = [](const TA::Range& r) {
    TA::Tensor<double> t(r);
    for (std::size_t p = 0; p < t.size(); ++p)
      t.data()[p] = 1.0 + static_cast<double>(p);
    return t;
  };
  auto tot_fill = [](auto& cell, const auto& idx) {
    const long key =
        7 * static_cast<long>(idx[0]) + 13 * static_cast<long>(idx[1]);
    for (std::size_t p = 0; p < cell.size(); ++p)
      cell.data()[p] = static_cast<double>(2 + static_cast<long>(p) + key);
  };

  PlainArray a(world, trange);
  a.init_tiles(plain_fill);
  ToTArray b(world, trange);
  b.init_tiles_nested(
      [](const auto&) {
        return typename InnerTile::range_type(std::vector<std::size_t>{4});
      },
      tot_fill);
  RefArray b_ref(world, trange);
  b_ref.init_tiles_nested(
      [](const auto&) {
        return TA::Tensor<double>::range_type(std::vector<std::size_t>{4});
      },
      tot_fill);
  world.gop.fence();

  auto c = TA::einsum("ij,ij;a->ij;a", a, b);
  auto c_ref = TA::einsum("ij,ij;a->ij;a", a, b_ref);
  world.gop.fence();

  for (const auto& tidx : c.trange().tiles_range()) {
    if (!c.is_local(tidx)) continue;
    auto tile = c.find(tidx).get();
    auto ref_tile = c_ref.find(tidx).get();
    BOOST_REQUIRE_EQUAL(tile.range().volume(), ref_tile.range().volume());
    for (std::size_t ord = 0; ord < tile.range().volume(); ++ord) {
      const auto& cell = tile.data()[ord];
      const auto& ref_cell = ref_tile.data()[ord];
      BOOST_REQUIRE(!cell.empty());
      BOOST_REQUIRE_EQUAL(cell.size(), ref_cell.size());
      for (std::size_t p = 0; p < cell.size(); ++p)
        BOOST_CHECK_EQUAL(cell.data()[p], ref_cell.data()[p]);
    }
  }
}

/// Tensor<ArenaTensor>::permute with a bipartite permutation: outer cells
/// reorder shallowly, inner cells are permuted into a fresh slab. Here both
/// the outer and inner parts are transposes.
void test_arena_tile_permute() {
  using Inner = TA::ArenaTensor<double>;
  using Outer = TA::Tensor<Inner>;
  constexpr long OI = 2, OJ = 3, R = 4, C = 5;
  auto val = [](long oi, long oj, long ii, long ij) {
    return 1.0 + oi * 1000.0 + oj * 100.0 + ii * 10.0 + ij;
  };
  Outer tile = TA::detail::make_nested_tile<Outer>(
      TA::Range{OI, OJ},
      [](const auto&) { return inner_range_2d<Inner>(R, C); },
      [&val](auto& cell, const auto& idx) {
        const long oi = static_cast<long>(idx[0]);
        const long oj = static_cast<long>(idx[1]);
        for (long ii = 0; ii < R; ++ii)
          for (long ij = 0; ij < C; ++ij)
            cell.data()[ii * C + ij] = val(oi, oj, ii, ij);
      });

  // bipartite transpose over the combined index space {0,1 | 2,3}: outer
  // part transposes dims 0,1 and inner part transposes dims 2,3; the trailing
  // 2 marks the second (inner) partition size.
  TA::BipartitePermutation bperm(TA::Permutation{1, 0, 3, 2}, 2);
  Outer p = tile.permute(bperm);

  // outer range transposed: {OI,OJ} -> {OJ,OI}
  BOOST_REQUIRE_EQUAL(p.range().extent(0), OJ);
  BOOST_REQUIRE_EQUAL(p.range().extent(1), OI);
  for (long oi = 0; oi < OI; ++oi)
    for (long oj = 0; oj < OJ; ++oj) {
      // src outer (oi,oj) lands at result outer (oj,oi)
      const auto& cell = p.data()[oj * OI + oi];
      BOOST_REQUIRE(!cell.empty());
      BOOST_REQUIRE_EQUAL(static_cast<long>(cell.size()), R * C);
      // inner transposed: result cell range {R,C} -> {C,R}
      BOOST_CHECK_EQUAL(cell.range().extent(0), C);
      BOOST_CHECK_EQUAL(cell.range().extent(1), R);
      for (long ii = 0; ii < R; ++ii)
        for (long ij = 0; ij < C; ++ij)
          BOOST_CHECK_EQUAL(cell.data()[ij * R + ii], val(oi, oj, ii, ij));
    }
}

/// conj()/conj(perm)/conj_to() on a tensor-of-tensors with complex inner cells.
/// Exercises the arena-aware scale(factor) and scale(factor, perm) paths (the
/// latter previously threw for view/arena inner) and in-place conj_to. Real
/// conj would be a no-op, so inner cells carry a nonzero imaginary part.
template <typename InnerTile>
void test_conj_tot() {
  using OuterTile = TA::Tensor<InnerTile>;
  using cd = typename InnerTile::value_type;  // std::complex<double>

  // rank-2 outer so a non-identity outer permutation is meaningful
  const TA::Range outer{2, 2};
  auto rng = [](const auto&) { return typename InnerTile::range_type{3}; };
  auto fill = [](auto& cell, const auto& idx) {
    const long e = static_cast<long>(idx[0]) * 10 + static_cast<long>(idx[1]);
    for (std::size_t i = 0; i < cell.size(); ++i)
      cell.data()[i] = cd(100.0 * e + i, 1.0 + e - static_cast<double>(i));
  };
  OuterTile src = TA::detail::make_nested_tile<OuterTile>(outer, rng, fill);

  auto inner_is_conj = [](const InnerTile& got, const InnerTile& s) {
    BOOST_REQUIRE_EQUAL(got.size(), s.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      BOOST_CHECK_EQUAL(got.data()[i].real(), s.data()[i].real());
      BOOST_CHECK_EQUAL(got.data()[i].imag(), -s.data()[i].imag());
    }
  };

  // out-of-place conj() (scale(conj_op) -> arena_trivial_unary)
  OuterTile c = src.conj();
  for (std::size_t o = 0; o < src.range().volume(); ++o)
    inner_is_conj(c.data()[o], src.data()[o]);

  // permuted conj() (scale(conj_op, perm) -- the arena branch under test);
  // must agree with conj-then-permute.
  TA::Permutation perm({1, 0});  // swap the two outer modes
  OuterTile cp = src.conj(perm);
  OuterTile cp_ref = src.conj().permute(perm);
  BOOST_REQUIRE_EQUAL(cp.range(), cp_ref.range());
  for (std::size_t o = 0; o < cp.range().volume(); ++o) {
    const InnerTile& a = cp.data()[o];
    const InnerTile& b = cp_ref.data()[o];
    BOOST_REQUIRE_EQUAL(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      BOOST_CHECK_EQUAL(a.data()[i], b.data()[i]);
  }

  // in-place conj_to() (scale_to(conj_op) -> free arena operator*=)
  OuterTile t2 = TA::detail::make_nested_tile<OuterTile>(outer, rng, fill);
  t2.conj_to();
  for (std::size_t o = 0; o < src.range().volume(); ++o)
    inner_is_conj(t2.data()[o], src.data()[o]);
}

}  // namespace

BOOST_AUTO_TEST_SUITE(tot_construction_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(make_nested_tile_tensor_inner) {
  test_make_nested_tile<TA::Tensor<double>>();
}

BOOST_AUTO_TEST_CASE(make_nested_tile_arena_inner) {
  test_make_nested_tile<TA::ArenaTensor<double>>();
}

BOOST_AUTO_TEST_CASE(dist_array_tot_ctor_tensor_inner) {
  test_dist_array_tot_ctor<TA::Tensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(dist_array_tot_ctor_arena_inner) {
  test_dist_array_tot_ctor<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(init_tiles_nested_tensor_inner) {
  test_init_tiles_nested<TA::Tensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(init_tiles_nested_arena_inner) {
  test_init_tiles_nested<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(fill_random_tensor_inner) {
  test_fill_random<TA::Tensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(fill_random_arena_inner) {
  test_fill_random<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(init_elements_tensor_inner) {
  test_init_elements<TA::Tensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(init_elements_arena_inner) {
  test_init_elements<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(fill_arena_inner) { test_fill_arena(); }

BOOST_AUTO_TEST_CASE(set_value_arena_inner) { test_set_value_arena(); }

BOOST_AUTO_TEST_CASE(set_iter_arena_inner) { test_set_iter_arena(); }

BOOST_AUTO_TEST_CASE(add_tensor_inner) {
  test_tot_add<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(add_arena_inner) {
  test_tot_add<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(subt_tensor_inner) {
  test_tot_subt<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(subt_arena_inner) {
  test_tot_subt<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(mult_tensor_inner) {
  test_tot_mult<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(mult_arena_inner) {
  test_tot_mult<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(scaled_add_tensor_inner) {
  test_tot_scaled_add<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(scaled_add_arena_inner) {
  test_tot_scaled_add<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(scaled_subt_tensor_inner) {
  test_tot_scaled_subt<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(scaled_subt_arena_inner) {
  test_tot_scaled_subt<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(scale_tensor_inner) {
  test_tot_scale<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(scale_arena_inner) {
  test_tot_scale<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(neg_tensor_inner) {
  test_tot_neg<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(neg_arena_inner) {
  test_tot_neg<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(conj_tot_tensor_inner) {
  test_conj_tot<TA::Tensor<std::complex<double>>>();
}
BOOST_AUTO_TEST_CASE(conj_tot_arena_inner) {
  test_conj_tot<TA::ArenaTensor<std::complex<double>>>();
}

// canonical inner contraction: c(ij;mn) = sum_k sum_o a(ijk;mo) b(ijk;on)
BOOST_AUTO_TEST_CASE(einsum_contraction_tensor_inner) {
  test_tot_einsum_contraction<TA::Tensor<double>, TA::DensePolicy>(
      "ijk;mo,ijk;on->ij;mn", 2, 3, 3, 2);
}
BOOST_AUTO_TEST_CASE(einsum_contraction_arena_inner) {
  test_tot_einsum_contraction<TA::ArenaTensor<double>, TA::DensePolicy>(
      "ijk;mo,ijk;on->ij;mn", 2, 3, 3, 2);
}

// non-canonical inner annotations: operand A reordered (o,m) and the result
// reordered (n,m) -- exercises the regime-A inner-permutation hoist.
BOOST_AUTO_TEST_CASE(einsum_contraction_perm_tensor_inner) {
  test_tot_einsum_contraction<TA::Tensor<double>, TA::DensePolicy>(
      "ijk;om,ijk;on->ij;nm", 3, 2, 3, 2);
}
BOOST_AUTO_TEST_CASE(einsum_contraction_perm_arena_inner) {
  test_tot_einsum_contraction<TA::ArenaTensor<double>, TA::DensePolicy>(
      "ijk;om,ijk;on->ij;nm", 3, 2, 3, 2);
}

// inner Hadamard with a permuted operand: c(ij;mn) = sum_k a(ijk;mn) b(ijk;nm)
BOOST_AUTO_TEST_CASE(einsum_hadamard_perm_tensor_inner) {
  test_tot_einsum_contraction<TA::Tensor<double>, TA::DensePolicy>(
      "ijk;mn,ijk;nm->ij;mn", 2, 3, 3, 2);
}
BOOST_AUTO_TEST_CASE(einsum_hadamard_perm_arena_inner) {
  test_tot_einsum_contraction<TA::ArenaTensor<double>, TA::DensePolicy>(
      "ijk;mn,ijk;nm->ij;mn", 2, 3, 3, 2);
}

// plain T x ToT Hadamard: c(ij;a) = a(ij) * b(ij;a), routed through the
// expression-DSL Mult tile op.
BOOST_AUTO_TEST_CASE(einsum_t_x_tot_tensor_inner) {
  test_tot_einsum_t_x_tot<TA::Tensor<double>, TA::DensePolicy>();
}
BOOST_AUTO_TEST_CASE(einsum_t_x_tot_arena_inner) {
  test_tot_einsum_t_x_tot<TA::ArenaTensor<double>, TA::DensePolicy>();
}

BOOST_AUTO_TEST_CASE(arena_tile_bipartite_permute) {
  test_arena_tile_permute();
}

BOOST_AUTO_TEST_CASE(make_nested_tile_arena_single_page) {
  using Inner = TA::ArenaTensor<double>;
  using OuterTile = TA::Tensor<Inner>;
  // 256 cells x 64 doubles = 128 KiB > 64 KiB default page, so a multi-page
  // builder spills; a single-page allocator does not.
  const std::size_t ncells = 256, isize = 64;
  TA::Range outer(std::array<std::size_t, 1>{ncells});
  auto range_fn = [isize](const TA::Range::index_type&) {
    return Inner::range_type(std::vector<std::size_t>{isize});
  };
  auto fill = [](Inner& cell, const TA::Range::index_type&) {
    for (std::size_t p = 0; p < cell.size(); ++p) cell[p] = double(p);
  };
  OuterTile tile =
      TA::detail::make_nested_tile<OuterTile>(outer, range_fn, fill);
  BOOST_REQUIRE_EQUAL(ncells * isize * sizeof(double), std::size_t(128 * 1024));
  check_single_page_uniform(tile);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(tot_construction_dist_suite, TA_UT_LABEL_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(arena_tot_remote_tile_transport) {
  test_distributed_arena_tot();
}

BOOST_AUTO_TEST_SUITE_END()
