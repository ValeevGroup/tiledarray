/// Unified tensor-of-tensors construction: detail::make_nested_tile,
/// DistArray::init_tiles_nested, and the DistArray ToT range_fn constructor --
/// exercised identically for TA::Tensor and ArenaTensor inner tiles.

#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "tiledarray.h"

#include "global_fixture.h"
#include "unit_test_config.h"

#include <cstddef>

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

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(tot_construction_dist_suite, TA_UT_LABEL_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(arena_tot_remote_tile_transport) {
  test_distributed_arena_tot();
}

BOOST_AUTO_TEST_SUITE_END()
