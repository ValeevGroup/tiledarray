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
#include "TiledArray/disk/store.h"
#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_tensor.h"
#include "tiledarray.h"
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

BOOST_AUTO_TEST_SUITE_END()
