/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 *  disk_tile.cpp
 *
 *  Tests of disk-backed (lazy) tiles and arrays.
 */

#include "TiledArray/disk/disk_tile.h"
#include "TiledArray/disk/disk_array.h"
#include "TiledArray/disk/store.h"
#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "tiledarray.h"

#include <madness/world/vector_archive.h>
#include "unit_test_config.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

namespace TA = TiledArray;

namespace test {

/// A lazy tile that simply holds its evaluated tile in memory: the minimal
/// tile type whose `eval_type` differs from itself. Reference point for the
/// behavior of lazy tiles that is independent of where the data lives.
template <typename Tile>
class InMemLazyTile {
 public:
  using eval_type = Tile;
  using value_type = typename Tile::value_type;
  using numeric_type = typename Tile::numeric_type;
  using scalar_type = typename Tile::scalar_type;
  using range_type = typename Tile::range_type;

  InMemLazyTile() = default;
  explicit InMemLazyTile(const Tile& tile)
      : tile_(std::make_shared<Tile>(tile)) {}

  explicit operator Tile() const { return *tile_; }

  template <typename Archive>
  void serialize(Archive& ar) {
    if constexpr (madness::is_output_archive_v<Archive>) {
      ar&* tile_;
    } else {
      Tile t;
      ar & t;
      tile_ = std::make_shared<Tile>(std::move(t));
    }
  }

 private:
  std::shared_ptr<Tile> tile_;
};

/// Names an eval_type but does not convert to it: not a lazy tile.
struct NamesEvalTypeOnly {
  using eval_type = TA::Tensor<TA::ArenaTensor<double, TA::Range>>;
};

}  // namespace test

namespace {

using ArenaInner = TA::ArenaTensor<double, TA::Range>;
using ArenaToT = TA::Tensor<ArenaInner>;
using ToTArray = TA::DistArray<ArenaToT, TA::SparsePolicy>;
using FlatArray = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;

/// Arena tensor-of-tensors array with deterministic, seed-shifted values.
ToTArray make_tot(TA::World& world, const TA::TiledRange& tr,
                  std::function<TA::Range(std::size_t)> inner, double seed) {
  ToTArray tot(world, tr);
  tot.init_tiles([=](TA::Range const& r) {
    ArenaToT t = TA::detail::arena_outer_init<ArenaToT>(r, 1, inner);
    for (std::size_t o = 0; o < t.range().volume(); ++o) {
      ArenaInner& c = t.data()[o];
      if (!c) continue;
      const auto n = c.range().volume();
      for (std::size_t a = 0; a < n; ++a)
        c.data()[a] = seed + 0.001 * static_cast<double>(
                                         (o * 7 + a + r.lobound()[0]) % 13);
    }
    return t;
  });
  world.gop.fence();
  return tot;
}

/// Copies \p array into an array of tiles wrapped by \p Lazy, with the same
/// TiledRange, shape and process map.
template <template <typename> class Lazy, typename Tile>
TA::DistArray<Lazy<Tile>, TA::SparsePolicy> wrap_tiles(
    const TA::DistArray<Tile, TA::SparsePolicy>& array) {
  TA::DistArray<Lazy<Tile>, TA::SparsePolicy> result(
      array.world(), array.trange(), array.shape(), array.pmap());
  for (auto it = array.pmap()->begin(); it != array.pmap()->end(); ++it) {
    if (array.is_zero(*it)) continue;
    result.set(*it, Lazy<Tile>(array.find_local(*it).get()));
  }
  array.world().gop.fence();
  return result;
}

/// A fresh, empty directory under the system temp directory, removed on
/// destruction; while alive it is the scratch directory.
struct ScratchDirGuard {
  std::filesystem::path prev = TA::disk_scratch_dir();
  std::filesystem::path dir;
  explicit ScratchDirGuard(const std::string& tag) {
    dir = std::filesystem::temp_directory_path() /
          ("ta-disk-test-" + tag + "-" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    TA::set_disk_scratch_dir(dir);
  }
  ~ScratchDirGuard() {
    TA::set_disk_scratch_dir(prev);
    std::filesystem::remove_all(dir);
  }
  bool empty() const { return std::filesystem::is_empty(dir); }
};

std::vector<char> pattern(std::size_t n, int seed) {
  std::vector<char> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = static_cast<char>((i * 31 + seed * 17) % 251);
  return v;
}

/// Element-wise equality of two flat tiles.
template <typename T>
bool same_tile(const TA::Tensor<T>& a, const TA::Tensor<T>& b) {
  if (a.empty() || b.empty()) return a.empty() == b.empty();
  return a.range() == b.range() &&
         std::equal(a.data(), a.data() + a.range().volume(), b.data());
}

/// Cell-by-cell equality of two arena tensor-of-tensors tiles.
bool same_tile(const ArenaToT& a, const ArenaToT& b) {
  if (a.empty() || b.empty()) return a.empty() == b.empty();
  if (a.range() != b.range()) return false;
  for (std::size_t o = 0; o < a.range().volume(); ++o) {
    const ArenaInner& x = a.data()[o];
    const ArenaInner& y = b.data()[o];
    if (static_cast<bool>(x) != static_cast<bool>(y)) return false;
    if (!x) continue;
    if (x.range() != y.range()) return false;
    if (!std::equal(x.data(), x.data() + x.range().volume(), y.data()))
      return false;
  }
  return true;
}

TA::Tensor<double> flat_tile(const TA::Range& r, double seed) {
  TA::Tensor<double> t(r);
  for (std::size_t o = 0; o < r.volume(); ++o)
    t.data()[o] = seed + 0.001 * static_cast<double>(o % 17);
  return t;
}

/// An arena ToT tile over \p r with inner Range{n}; the first cell is left
/// empty.
ArenaToT tot_tile(const TA::Range& r, std::size_t n, double seed) {
  ArenaToT t = TA::detail::arena_outer_init<ArenaToT>(
      r, 1, [n](std::size_t o) { return o == 0 ? TA::Range{} : TA::Range{n}; });
  for (std::size_t o = 0; o < t.range().volume(); ++o) {
    ArenaInner& c = t.data()[o];
    if (!c) continue;
    for (std::size_t a = 0; a < c.range().volume(); ++a)
      c.data()[a] = seed + 0.001 * static_cast<double>((o * 7 + a) % 13);
  }
  return t;
}

/// Flat array with deterministic, seed-shifted values.
template <typename Policy = TA::SparsePolicy>
TA::DistArray<TA::Tensor<double>, Policy> make_flat(TA::World& world,
                                                    const TA::TiledRange& tr,
                                                    double seed) {
  TA::DistArray<TA::Tensor<double>, Policy> flat(world, tr);
  flat.init_tiles([=](TA::Range const& r) {
    TA::Tensor<double> t(r);
    for (std::size_t o = 0; o < t.range().volume(); ++o)
      t.data()[o] =
          seed + 0.001 * static_cast<double>((o + 3 * r.lobound()[0]) % 17);
    return t;
  });
  world.gop.fence();
  return flat;
}

/// \return true if \p result equals \p ref (both annotated by \p annot) to
/// within rounding.
template <typename Array>
bool close(const Array& result, const Array& ref, const std::string& annot) {
  Array diff;
  diff(annot) = result(annot) - ref(annot);
  const double nref = TA::norm2(ref);
  return nref > 0.0 && TA::norm2(diff) <= 1e-12 * nref;
}

/// Disk I/O counters summed over all ranks.
TA::DiskIOStats global_disk_io_stats(TA::World& world) {
  auto const s = TA::disk_io_stats();
  std::size_t c[4] = {s.tiles_written, s.tiles_read, s.bytes_written,
                      s.bytes_read};
  world.gop.sum(c, 4);
  TA::DiskIOStats g;
  g.tiles_written = c[0];
  g.tiles_read = c[1];
  g.bytes_written = c[2];
  g.bytes_read = c[3];
  return g;
}

/// Zeroes the disk I/O counters on all ranks (collective).
void reset_global_disk_io_stats(TA::World& world) {
  world.gop.fence();
  TA::reset_disk_io_stats();
  world.gop.fence();
}

/// The arrays the expression tests share, in memory and on disk.
struct DiskArrayFixture {
  TA::World& world = TA::get_default_world();
  TA::TiledRange1 const tr_i{0, 2, 4, 6, 8}, tr_p{0, 8, 16, 24}, tr_k{0, 5, 10};
  ToTArray tot = make_tot(
      world, TA::TiledRange{tr_i, tr_i, tr_p},
      [](std::size_t) { return TA::Range{6}; }, 1.0);
  ToTArray tot2 = make_tot(
      world, TA::TiledRange{tr_i, tr_i, tr_p},
      [](std::size_t) { return TA::Range{6}; }, 0.3);
  ToTArray s = make_tot(
      world, TA::TiledRange{tr_i, tr_i},
      [](std::size_t) {
        return TA::Range{6, 6};
      },
      0.7);
  FlatArray g = make_flat(world, TA::TiledRange{tr_p, tr_p, tr_k}, 0.5);
  FlatArray x = make_flat(world, TA::TiledRange{tr_p, tr_k}, 0.2);
  TA::DiskArray<ArenaToT, TA::SparsePolicy> totd = TA::to_disk(tot),
                                            tot2d = TA::to_disk(tot2),
                                            sd = TA::to_disk(s);
  TA::DiskArray<TA::Tensor<double>, TA::SparsePolicy> gd = TA::to_disk(g),
                                                      xd = TA::to_disk(x);
};

}  // namespace

BOOST_AUTO_TEST_SUITE(disk_tile_suite)

BOOST_AUTO_TEST_CASE(lazy_tile_nestedness) {
  // a lazy tile is as nested as the tile it evaluates to ...
  static_assert(
      TA::detail::is_tensor_of_tensor_v<test::InMemLazyTile<ArenaToT>>);
  static_assert(!TA::detail::is_tensor_of_tensor_v<
                test::InMemLazyTile<TA::Tensor<double>>>);
  // ... but it is not itself a tensor
  static_assert(
      !TA::detail::is_tensor_v<test::InMemLazyTile<TA::Tensor<double>>>);
  // only a lazy TILE is looked through: an array (or any other type that
  // merely names an eval_type, without converting to it) is not a tensor of
  // tensors, whatever its tiles are
  static_assert(!TA::detail::is_tensor_of_tensor_v<ToTArray>);
  static_assert(
      !TA::detail::is_tensor_of_tensor_v<
          TA::DistArray<test::InMemLazyTile<ArenaToT>, TA::SparsePolicy>>);
  static_assert(!TA::detail::is_tensor_of_tensor_v<test::NamesEvalTypeOnly>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(lazy_tot_annotation) {
  auto& world = TA::get_default_world();
  TA::TiledRange const tr{{0, 2, 4}, {0, 3, 6}};
  auto const tot = make_tot(
      world, tr, [](std::size_t) { return TA::Range{3}; }, 1.0);
  auto const lazy = wrap_tiles<test::InMemLazyTile>(tot);

  // a nested annotation on an array of lazy nested tiles is accepted
  ToTArray r;
  BOOST_REQUIRE_NO_THROW(r("i,j;a") = lazy("i,j;a"));
  ToTArray d;
  d("i,j;a") = r("i,j;a") - tot("i,j;a");
  BOOST_CHECK_EQUAL(TA::norm2(d), 0.0);
  BOOST_CHECK(TA::norm2(tot) > 0.0);
}

BOOST_AUTO_TEST_CASE(store_roundtrip) {
  TA::detail::DiskStore store;
  auto const a = pattern(10, 1), b = pattern(1000, 2), c = pattern(3, 3);
  BOOST_CHECK_EQUAL(store.append(a.data(), a.size()), 0u);
  BOOST_CHECK_EQUAL(store.append(b.data(), b.size()), 10u);
  BOOST_CHECK_EQUAL(store.append(c.data(), c.size()), 1010u);
  BOOST_CHECK_EQUAL(store.size(), 1013u);
  std::vector<char> rb(b.size()), ra(a.size()), rc(c.size());
  store.read(10, rb.data(), rb.size());
  store.read(0, ra.data(), ra.size());
  store.read(1010, rc.data(), rc.size());
  BOOST_CHECK(ra == a);
  BOOST_CHECK(rb == b);
  BOOST_CHECK(rc == c);
}

BOOST_AUTO_TEST_CASE(store_chunked_io) {
  // requests larger than the chunk size (as a tile over 2 GiB would be
  // against the default) are split and reassembled
  auto const prev = TA::detail::disk_io_chunk();
  TA::detail::disk_io_chunk() = 7;
  {
    TA::detail::DiskStore store;
    auto const a = pattern(1000, 5);
    BOOST_CHECK_EQUAL(store.append(a.data(), a.size()), 0u);
    std::vector<char> r(a.size());
    store.read(0, r.data(), r.size());
    BOOST_CHECK(r == a);
    std::vector<char> mid(100);
    store.read(450, mid.data(), mid.size());
    BOOST_CHECK(std::equal(mid.begin(), mid.end(), a.begin() + 450));
  }
  TA::detail::disk_io_chunk() = prev;
}

BOOST_AUTO_TEST_CASE(store_concurrent) {
  TA::detail::DiskStore store;
  const int nthreads = 8, nper = 200;
  std::vector<std::vector<std::size_t>> offsets(nthreads,
                                                std::vector<std::size_t>(nper));
  auto size_of = [](int t, int i) { return std::size_t(1 + (t * 7 + i) % 97); };
  std::vector<std::thread> threads;
  for (int t = 0; t < nthreads; ++t)
    threads.emplace_back([&, t] {
      for (int i = 0; i < nper; ++i) {
        auto const p = pattern(size_of(t, i), t * 1000 + i);
        offsets[t][i] = store.append(p.data(), p.size());
        // read back something written earlier while others are appending
        if (i > 0) {
          auto const q = pattern(size_of(t, i - 1), t * 1000 + i - 1);
          std::vector<char> r(q.size());
          store.read(offsets[t][i - 1], r.data(), r.size());
          if (r != q) std::abort();
        }
      }
    });
  for (auto& th : threads) th.join();
  std::size_t total = 0;
  bool all_equal = true;
  for (int t = 0; t < nthreads; ++t)
    for (int i = 0; i < nper; ++i) {
      auto const p = pattern(size_of(t, i), t * 1000 + i);
      std::vector<char> r(p.size());
      store.read(offsets[t][i], r.data(), r.size());
      all_equal = all_equal && r == p;
      total += p.size();
    }
  BOOST_CHECK(all_equal);
  BOOST_CHECK_EQUAL(store.size(), total);
}

BOOST_AUTO_TEST_CASE(store_leaves_no_file) {
  ScratchDirGuard scratch("nofile");
  {
    TA::detail::DiskStore store;
    auto const a = pattern(100, 1);
    store.append(a.data(), a.size());
    // the file has no name: nothing to find, nothing to leave behind
    BOOST_CHECK(scratch.empty());
  }
  BOOST_CHECK(scratch.empty());
}

BOOST_AUTO_TEST_CASE(store_bad_dir_throws) {
  auto const prev = TA::disk_scratch_dir();
  const std::string bad = "/nonexistent-ta-disk-dir/scratch";
  TA::set_disk_scratch_dir(bad);
  bool threw = false;
  try {
    TA::detail::DiskStore store;
  } catch (TA::Exception& e) {
    threw = true;
    BOOST_CHECK(std::string(e.what()).find(bad) != std::string::npos);
  }
  BOOST_CHECK(threw);
  TA::set_disk_scratch_dir(prev);
}

BOOST_AUTO_TEST_CASE(scratch_dir_resolution) {
  auto const prev = TA::disk_scratch_dir();
  const char* const prev_env = std::getenv("TA_DISK_SCRATCH_DIR");
  const std::string saved_env = prev_env ? prev_env : "";

  // nothing set: the environment variable, else the system temp directory
  TA::set_disk_scratch_dir({});
  ::unsetenv("TA_DISK_SCRATCH_DIR");
  BOOST_CHECK(TA::disk_scratch_dir() == std::filesystem::temp_directory_path());
  ::setenv("TA_DISK_SCRATCH_DIR", "/tmp/ta-env-dir", 1);
  BOOST_CHECK(TA::disk_scratch_dir() ==
              std::filesystem::path("/tmp/ta-env-dir"));
  // an explicit setting wins over the environment
  TA::set_disk_scratch_dir("/tmp/ta-set-dir");
  BOOST_CHECK(TA::disk_scratch_dir() ==
              std::filesystem::path("/tmp/ta-set-dir"));

  if (prev_env)
    ::setenv("TA_DISK_SCRATCH_DIR", saved_env.c_str(), 1);
  else
    ::unsetenv("TA_DISK_SCRATCH_DIR");
  TA::set_disk_scratch_dir(prev);
}

BOOST_AUTO_TEST_CASE(io_stats) {
  TA::detail::DiskStore store;
  auto const a = pattern(100, 1);
  TA::reset_disk_io_stats();
  store.append(a.data(), a.size());
  std::vector<char> r(40);
  store.read(10, r.data(), r.size());
  auto const s = TA::disk_io_stats();
  BOOST_CHECK_EQUAL(s.bytes_written, 100u);
  BOOST_CHECK_EQUAL(s.bytes_read, 40u);
  BOOST_CHECK(s.seconds_writing >= 0.0);
  BOOST_CHECK(s.seconds_reading >= 0.0);
  BOOST_CHECK_EQUAL(s.tiles_written, 0u);  // counted by DiskTile, not the store
  BOOST_CHECK_EQUAL(s.tiles_read, 0u);
}

BOOST_AUTO_TEST_CASE(tile_roundtrip_flat) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = flat_tile(TA::Range{4, 5}, 1.0);
  TA::DiskTile<TA::Tensor<double>> const d(t, store);
  BOOST_CHECK(d.on_disk());
  BOOST_CHECK(!d.empty());
  BOOST_CHECK(same_tile(static_cast<TA::Tensor<double>>(d), t));
  // a second tile in the same store does not disturb the first
  auto const t2 = flat_tile(TA::Range{3, 3}, 2.0);
  TA::DiskTile<TA::Tensor<double>> const d2(t2, store);
  BOOST_CHECK(same_tile(static_cast<TA::Tensor<double>>(d), t));
  BOOST_CHECK(same_tile(static_cast<TA::Tensor<double>>(d2), t2));
  // default-constructed: empty, not on disk
  TA::DiskTile<TA::Tensor<double>> const e;
  BOOST_CHECK(e.empty());
  BOOST_CHECK(!e.on_disk());
}

BOOST_AUTO_TEST_CASE(tile_roundtrip_tot) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = tot_tile(TA::Range{2, 3}, 4, 1.0);
  BOOST_REQUIRE(!t.data()[0]);  // the fixture's empty inner cell
  TA::DiskTile<ArenaToT> const d(t, store);
  BOOST_CHECK(same_tile(static_cast<ArenaToT>(d), t));
}

BOOST_AUTO_TEST_CASE(tile_roundtrip_complex) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  TA::Tensor<std::complex<double>> t(TA::Range{3, 2});
  for (std::size_t o = 0; o < 6; ++o)
    t.data()[o] = std::complex<double>(1.0 + o, -0.5 * o);
  TA::DiskTile<TA::Tensor<std::complex<double>>> const d(t, store);
  BOOST_CHECK(same_tile(static_cast<TA::Tensor<std::complex<double>>>(d), t));
}

BOOST_AUTO_TEST_CASE(tile_is_lazy) {
  using D = TA::DiskTile<TA::Tensor<double>>;
  static_assert(TA::is_lazy_tile_v<D>);
  static_assert(std::is_same_v<TA::eval_trait<D>::type, TA::Tensor<double>>);
  static_assert(TA::detail::is_tensor_of_tensor_v<TA::DiskTile<ArenaToT>>);
  static_assert(!TA::detail::is_tensor_of_tensor_v<D>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(tile_no_caching) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = flat_tile(TA::Range{4, 5}, 1.0);
  TA::reset_disk_io_stats();
  TA::DiskTile<TA::Tensor<double>> const d(t, store);
  BOOST_CHECK_EQUAL(TA::disk_io_stats().tiles_written, 1u);
  for (int i = 0; i < 3; ++i)
    BOOST_CHECK(same_tile(static_cast<TA::Tensor<double>>(d), t));
  // every evaluation goes to disk: nothing is kept in the tile
  BOOST_CHECK_EQUAL(TA::disk_io_stats().tiles_read, 3u);
}

BOOST_AUTO_TEST_CASE(tile_serialize_counts_without_reading) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = tot_tile(TA::Range{2, 3}, 4, 1.0);
  TA::DiskTile<ArenaToT> d(t, store);
  TA::reset_disk_io_stats();
  madness::archive::BufferOutputArchive count;
  count & d;
  BOOST_CHECK(count.count_only());
  BOOST_CHECK_EQUAL(count.size(), d.serialized_size());
  // sizing a message costs no disk read
  BOOST_CHECK_EQUAL(TA::disk_io_stats().tiles_read, 0u);
  BOOST_CHECK_EQUAL(TA::disk_io_stats().bytes_read, 0u);
}

BOOST_AUTO_TEST_CASE(tile_serialize_roundtrip) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = tot_tile(TA::Range{2, 3}, 4, 1.0);
  TA::DiskTile<ArenaToT> d(t, store);
  std::vector<unsigned char> buf(d.serialized_size());
  TA::reset_disk_io_stats();
  madness::archive::BufferOutputArchive out(buf.data(), buf.size());
  out & d;
  BOOST_CHECK_EQUAL(out.size(), buf.size());
  // shipping a tile reads it from disk exactly once
  BOOST_CHECK_EQUAL(TA::disk_io_stats().tiles_read, 1u);

  // what arrives holds its data in memory: it needs no file
  madness::archive::BufferInputArchive in(buf.data(), buf.size());
  TA::DiskTile<ArenaToT> u;
  in & u;
  BOOST_CHECK(!u.on_disk());
  BOOST_CHECK(!u.empty());
  store.reset();
  d = TA::DiskTile<ArenaToT>{};
  TA::reset_disk_io_stats();
  BOOST_CHECK(same_tile(static_cast<ArenaToT>(u), t));
  BOOST_CHECK_EQUAL(TA::disk_io_stats().tiles_read, 0u);

  // and it can be shipped onward
  std::vector<unsigned char> buf2(u.serialized_size());
  BOOST_CHECK_EQUAL(buf2.size(), buf.size());
  madness::archive::BufferOutputArchive out2(buf2.data(), buf2.size());
  out2 & u;
  BOOST_CHECK(buf2 == buf);

  // an empty tile survives the trip as an empty tile
  TA::DiskTile<ArenaToT> e;
  std::vector<unsigned char> ebuf(e.serialized_size());
  madness::archive::BufferOutputArchive eout(ebuf.data(), ebuf.size());
  eout & e;
  madness::archive::BufferInputArchive ein(ebuf.data(), ebuf.size());
  TA::DiskTile<ArenaToT> e2(t, std::make_shared<TA::detail::DiskStore>());
  ein & e2;
  BOOST_CHECK(e2.empty());
}

BOOST_AUTO_TEST_CASE(tile_outlives_store_owner) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = flat_tile(TA::Range{4, 5}, 1.0);
  auto d = std::make_unique<TA::DiskTile<TA::Tensor<double>>>(t, store);
  TA::DiskTile<TA::Tensor<double>> const copy = *d;
  // the creator's handle and the original tile go away; the copy keeps the
  // scratch file alive
  store.reset();
  d.reset();
  BOOST_CHECK(copy.on_disk());
  BOOST_CHECK(same_tile(static_cast<TA::Tensor<double>>(copy), t));
}

BOOST_FIXTURE_TEST_CASE(expr_flat_contraction, DiskArrayFixture) {
  FlatArray ref, r;
  ref("q,l,k") = g("q,p,l") * x("p,k");
  r("q,l,k") = gd("q,p,l") * x("p,k");
  BOOST_CHECK(close(r, ref, "q,l,k"));
  r("q,l,k") = g("q,p,l") * xd("p,k");
  BOOST_CHECK(close(r, ref, "q,l,k"));
  r("q,l,k") = gd("q,p,l") * xd("p,k");
  BOOST_CHECK(close(r, ref, "q,l,k"));
}

BOOST_FIXTURE_TEST_CASE(expr_flat_general, DiskArrayFixture) {
  FlatArray ref, r;
  ref("q,r,k") = g("q,p,k") * g("p,r,k");
  r("q,r,k") = gd("q,p,k") * gd("p,r,k");
  BOOST_CHECK(close(r, ref, "q,r,k"));
}

BOOST_FIXTURE_TEST_CASE(expr_flat_binary, DiskArrayFixture) {
  FlatArray ref, r;
  ref("k,p") = 3.0 * x("p,k") - 2.0 * x("p,k");
  r("k,p") = 3.0 * xd("p,k") - 2.0 * x("p,k");
  BOOST_CHECK(close(r, ref, "k,p"));
}

BOOST_FIXTURE_TEST_CASE(expr_flat_materialize, DiskArrayFixture) {
  FlatArray r;
  r("p,k") = xd("p,k");
  BOOST_CHECK(close(r, x, "p,k"));
}

BOOST_FIXTURE_TEST_CASE(expr_mixed_half_transform, DiskArrayFixture) {
  const std::string tc = "i,j,q,k;a";
  ToTArray ref, r;
  ref(tc) = tot("i,j,p;a") * g("q,p,k");
  r(tc) = tot("i,j,p;a") * gd("q,p,k");
  BOOST_CHECK(close(r, ref, tc));
  r(tc) = totd("i,j,p;a") * g("q,p,k");
  BOOST_CHECK(close(r, ref, tc));
  r(tc) = totd("i,j,p;a") * gd("q,p,k");
  BOOST_CHECK(close(r, ref, tc));
  r(tc) = (gd("q,p,k") * totd("i,j,p;a")).set_shape(ref.shape());
  BOOST_CHECK(close(r, ref, tc));
}

BOOST_FIXTURE_TEST_CASE(expr_tot_materialize, DiskArrayFixture) {
  ToTArray r;
  r("i,j,p;a") = totd("i,j,p;a");
  BOOST_CHECK(close(r, tot, "i,j,p;a"));
}

BOOST_FIXTURE_TEST_CASE(expr_tot_binary_permuted, DiskArrayFixture) {
  ToTArray ref, r;
  ref("j,i,p;a") = 2.0 * tot("i,j,p;a") - tot2("i,j,p;a");
  r("j,i,p;a") = 2.0 * totd("i,j,p;a") - tot2("i,j,p;a");
  BOOST_CHECK(close(r, ref, "j,i,p;a"));
}

BOOST_FIXTURE_TEST_CASE(expr_dot_inner, DiskArrayFixture) {
  {  // outer indices fused
    FlatArray ref, r;
    ref("i,j,p") = tot("i,j,p;a").dot_inner(tot2("i,j,p;a"));
    r("i,j,p") = totd("i,j,p;a").dot_inner(tot2("i,j,p;a"));
    BOOST_CHECK(close(r, ref, "i,j,p"));
  }
  {  // one outer index contracted
    FlatArray ref, r;
    ref("i,j") = tot("i,j,p;a").dot_inner(tot2("i,j,p;a"));
    r("i,j") = totd("i,j,p;a").dot_inner(tot2d("i,j,p;a"));
    BOOST_CHECK(close(r, ref, "i,j"));
  }
  {  // outer contraction
    FlatArray ref, r;
    ref("i,k") = tot("i,j,p;a").dot_inner(tot2("j,k,p;a"));
    r("i,k") = totd("i,j,p;a").dot_inner(tot2d("j,k,p;a"));
    BOOST_CHECK(close(r, ref, "i,k"));
  }
}

BOOST_FIXTURE_TEST_CASE(expr_tot_tot, DiskArrayFixture) {
  ToTArray ref, r;
  ref("i,j,p;a") = tot("i,j,p;b") * s("i,j;a,b");
  r("i,j,p;a") = totd("i,j,p;b") * sd("i,j;a,b");
  BOOST_CHECK(close(r, ref, "i,j,p;a"));
}

BOOST_FIXTURE_TEST_CASE(materialize_block, DiskArrayFixture) {
  // tiles p in {1,2} x k in {0}: 2 of the 6 tiles
  FlatArray ref, r;
  ref("p,k") = x("p,k").block({1, 0}, {3, 1});
  reset_global_disk_io_stats(world);
  r("p,k") = xd("p,k").block({1, 0}, {3, 1});
  world.gop.fence();
  BOOST_CHECK(close(r, ref, "p,k"));
  // only the tiles of the block are read
  BOOST_CHECK_EQUAL(global_disk_io_stats(world).tiles_read, 2u);
}

BOOST_AUTO_TEST_CASE(dense_policy) {
  auto& world = TA::get_default_world();
  using DenseArray = TA::DistArray<TA::Tensor<double>, TA::DensePolicy>;
  TA::TiledRange1 const tr_p{0, 8, 16, 24}, tr_k{0, 5, 10};
  DenseArray const g =
      make_flat<TA::DensePolicy>(world, TA::TiledRange{tr_p, tr_p, tr_k}, 0.5);
  DenseArray const x =
      make_flat<TA::DensePolicy>(world, TA::TiledRange{tr_p, tr_k}, 0.2);
  auto const gd = TA::to_disk(g);
  DenseArray r, ref;
  r("q,p,k") = gd("q,p,k");
  BOOST_CHECK(close(r, g, "q,p,k"));
  ref("q,l,k") = g("q,p,l") * x("p,k");
  r("q,l,k") = gd("q,p,l") * x("p,k");
  BOOST_CHECK(close(r, ref, "q,l,k"));
}

BOOST_FIXTURE_TEST_CASE(writer_tile_at_a_time, DiskArrayFixture) {
  TA::DiskArrayWriter<TA::Tensor<double>, TA::SparsePolicy> writer(
      world, g.trange(), g.shape(), g.pmap());
  for (auto it = g.pmap()->begin(); it != g.pmap()->end(); ++it)
    if (!g.is_zero(*it)) writer.set(*it, g.find_local(*it).get());
  world.gop.fence();
  FlatArray r;
  r("q,p,k") = writer.array()("q,p,k");
  BOOST_CHECK(close(r, g, "q,p,k"));
}

BOOST_AUTO_TEST_CASE(rank_without_tiles) {
  // a sparse array all of whose nonzero tiles are owned by rank 0
  auto& world = TA::get_default_world();
  TA::TiledRange const tr{{0, 8, 16, 24}, {0, 5, 10}};
  FlatArray const dense = make_flat(world, tr, 0.2);
  TA::Tensor<float> norms(tr.tiles_range(), 0.0f);
  for (std::size_t ord = 0; ord < tr.tiles_range().volume(); ++ord)
    if (dense.pmap()->owner(ord) == 0) norms[ord] = 1.0f;
  TA::SparseShape<float> const shape(norms, tr, /* do_not_scale = */ true);
  FlatArray x(world, tr, shape, dense.pmap());
  for (auto it = x.pmap()->begin(); it != x.pmap()->end(); ++it)
    if (!x.is_zero(*it)) x.set(*it, dense.find_local(*it).get());
  world.gop.fence();

  reset_global_disk_io_stats(world);
  auto const xd = TA::to_disk(x);
  // a rank that owns nothing writes nothing
  if (world.rank() != 0) {
    BOOST_CHECK_EQUAL(TA::disk_io_stats().tiles_written, 0u);
    BOOST_CHECK_EQUAL(TA::disk_io_stats().bytes_written, 0u);
  }
  FlatArray r;
  r("p,k") = xd("p,k");
  BOOST_CHECK(close(r, x, "p,k"));
  FlatArray c, cref;
  cref("p,q") = x("p,k") * x("q,k");
  c("p,q") = xd("p,k") * xd("q,k");
  BOOST_CHECK(close(c, cref, "p,q"));
}

BOOST_FIXTURE_TEST_CASE(to_disk_preserves_structure, DiskArrayFixture) {
  BOOST_CHECK(gd.trange() == g.trange());
  BOOST_CHECK(gd.pmap() == g.pmap());
  bool same_shape = true;
  for (std::size_t ord = 0; ord < g.trange().tiles_range().volume(); ++ord)
    same_shape = same_shape && gd.is_zero(ord) == g.is_zero(ord) &&
                 gd.is_local(ord) == g.is_local(ord);
  BOOST_CHECK(same_shape);
  BOOST_CHECK(totd.trange() == tot.trange());
}

BOOST_AUTO_TEST_CASE(no_file_left) {
  auto& world = TA::get_default_world();
  ScratchDirGuard scratch("arrays-r" + std::to_string(world.rank()));
  {
    TA::TiledRange const tr{{0, 8, 16, 24}, {0, 5, 10}};
    FlatArray const x = make_flat(world, tr, 0.2);
    auto const xd = TA::to_disk(x);
    // scratch files have no name: nothing to find while the array is alive
    BOOST_CHECK(scratch.empty());
    FlatArray r;
    r("p,k") = xd("p,k");
    BOOST_CHECK(close(r, x, "p,k"));
    world.gop.fence();
  }
  world.gop.fence();
  BOOST_CHECK(scratch.empty());
}

// evaluating a tile that holds its data in memory (one received from another
// process) gives an independent tile each time, as evaluating an on-disk tile
// does: consumers of a lazy tile are entitled to modify what they are given
BOOST_AUTO_TEST_CASE(tile_in_memory_evaluations_independent) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = flat_tile(TA::Range{4, 5}, 1.0);
  TA::DiskTile<TA::Tensor<double>> d(t, store);
  std::vector<unsigned char> buf(d.serialized_size());
  madness::archive::BufferOutputArchive out(buf.data(), buf.size());
  out & d;
  madness::archive::BufferInputArchive in(buf.data(), buf.size());
  TA::DiskTile<TA::Tensor<double>> u;
  in & u;
  BOOST_REQUIRE(!u.on_disk());

  auto first = static_cast<TA::Tensor<double>>(u);
  first.data()[0] += 100.0;  // a consumer working in place
  auto const second = static_cast<TA::Tensor<double>>(u);
  BOOST_CHECK(same_tile(second, t));
  BOOST_CHECK(first.data() != second.data());
}

// serialization through an archive that is not a MADNESS buffer archive
BOOST_AUTO_TEST_CASE(tile_serialize_generic_archive) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  auto const t = tot_tile(TA::Range{2, 3}, 4, 1.0);
  TA::DiskTile<ArenaToT> d(t, store);
  std::vector<unsigned char> v;
  {
    madness::archive::VectorOutputArchive out(v);
    out & d;
  }
  madness::archive::VectorInputArchive in(v);
  TA::DiskTile<ArenaToT> u;
  in & u;
  BOOST_CHECK(!u.on_disk());
  BOOST_CHECK(same_tile(static_cast<ArenaToT>(u), t));
}

// misuse is an error in every build, not only where assertions are on
BOOST_AUTO_TEST_CASE(tile_misuse_throws) {
  using D = TA::DiskTile<TA::Tensor<double>>;
  auto const t = flat_tile(TA::Range{4, 5}, 1.0);
  BOOST_CHECK_THROW(static_cast<TA::Tensor<double>>(D{}), TA::Exception);
  BOOST_CHECK_THROW((void)D(t, nullptr), TA::Exception);
}

BOOST_FIXTURE_TEST_CASE(writer_misuse_throws, DiskArrayFixture) {
  TA::Tensor<float> norms(g.trange().tiles_range(), 1.0f);
  norms[0] = 0.0f;  // tile 0 is zero
  TA::SparseShape<float> const shape(norms, g.trange(), true);
  TA::DiskArrayWriter<TA::Tensor<double>, TA::SparsePolicy> writer(
      world, g.trange(), shape, g.pmap());
  auto const tile = flat_tile(g.trange().make_tile_range(0), 1.0);
  // a zero tile cannot be set
  if (writer.array().is_local(0))
    BOOST_CHECK_THROW(writer.set(0, tile), TA::Exception);
  // nor can a tile owned by another process
  for (std::size_t ord = 1; ord < g.trange().tiles_range().volume(); ++ord)
    if (!writer.array().is_local(ord)) {
      BOOST_CHECK_THROW(
          writer.set(ord, flat_tile(g.trange().make_tile_range(ord), 1.0)),
          TA::Exception);
      break;
    }
  // no tile was written by the rejected calls
  for (auto it = g.pmap()->begin(); it != g.pmap()->end(); ++it)
    if (*it != 0) writer.set(*it, g.find_local(*it).get());
  world.gop.fence();
}

// the scratch file is released when the last tile referring to it is gone
BOOST_AUTO_TEST_CASE(store_released_with_last_tile) {
  auto store = std::make_shared<TA::detail::DiskStore>();
  std::weak_ptr<TA::detail::DiskStore> const watch = store;
  auto const t = flat_tile(TA::Range{4, 5}, 1.0);
  auto d1 = std::make_unique<TA::DiskTile<TA::Tensor<double>>>(t, store);
  auto d2 = std::make_unique<TA::DiskTile<TA::Tensor<double>>>(*d1);
  store.reset();
  BOOST_CHECK(!watch.expired());
  d1.reset();
  BOOST_CHECK(!watch.expired());
  d2.reset();
  BOOST_CHECK(watch.expired());
}

// a tile requested by a process that does not own it is read from disk
// exactly once, by its owner
BOOST_FIXTURE_TEST_CASE(remote_fetch_reads_once, DiskArrayFixture,
                        TA_UT_LABEL_DISTRIBUTED) {
  reset_global_disk_io_stats(world);
  std::size_t nfetched = 0;
  bool all_equal = true;
  for (std::size_t ord = 0; ord < gd.trange().tiles_range().volume(); ++ord) {
    if (gd.is_zero(ord) || gd.is_local(ord)) continue;
    auto const fetched = gd.find(ord).get();
    ++nfetched;
    // what arrives holds its data in memory and evaluates to the source
    all_equal =
        all_equal && !fetched.on_disk() && !fetched.empty() &&
        same_tile(static_cast<TA::Tensor<double>>(fetched), g.find(ord).get());
  }
  world.gop.fence();
  BOOST_CHECK(all_equal);
  world.gop.sum(nfetched);
  if (world.size() > 1) BOOST_CHECK(nfetched > 0);
  BOOST_CHECK_EQUAL(global_disk_io_stats(world).tiles_read, nfetched);
}

BOOST_FIXTURE_TEST_CASE(remote_fetch_tot, DiskArrayFixture,
                        TA_UT_LABEL_DISTRIBUTED) {
  reset_global_disk_io_stats(world);
  std::size_t nfetched = 0;
  bool all_equal = true;
  for (std::size_t ord = 0; ord < totd.trange().tiles_range().volume(); ++ord) {
    if (totd.is_zero(ord) || totd.is_local(ord)) continue;
    auto const fetched = totd.find(ord).get();
    ++nfetched;
    all_equal = all_equal && !fetched.on_disk() &&
                same_tile(static_cast<ArenaToT>(fetched), tot.find(ord).get());
  }
  world.gop.fence();
  BOOST_CHECK(all_equal);
  world.gop.sum(nfetched);
  if (world.size() > 1) BOOST_CHECK(nfetched > 0);
  BOOST_CHECK_EQUAL(global_disk_io_stats(world).tiles_read, nfetched);
}

// a contraction reads each tile of a disk-backed argument once where it is
// used locally and once more if it is shipped: never more often
BOOST_FIXTURE_TEST_CASE(streaming_contraction, DiskArrayFixture,
                        TA_UT_LABEL_DISTRIBUTED) {
  FlatArray ref, r;
  ref("q,l,k") = g("q,p,l") * x("p,k");
  std::size_t ntiles = 0;
  for (std::size_t ord = 0; ord < gd.trange().tiles_range().volume(); ++ord)
    if (!gd.is_zero(ord)) ++ntiles;
  reset_global_disk_io_stats(world);
  r("q,l,k") = gd("q,p,l") * x("p,k");
  world.gop.fence();
  BOOST_CHECK(close(r, ref, "q,l,k"));
  auto const nread = global_disk_io_stats(world).tiles_read;
  BOOST_CHECK(nread >= ntiles);
  BOOST_CHECK(nread <= 2 * ntiles);
}

BOOST_AUTO_TEST_SUITE_END()
