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
 */

#include "TiledArray/pmap/blocked_pmap.h"
#include "TiledArray/pmap/slabbed_pmap.h"
#include "global_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <madness/world/worldrmi.h>

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace TiledArray;

namespace {

TiledRange1 make_tr1(std::size_t ntiles, std::size_t extent) {
  std::vector<std::size_t> hashmarks;
  for (std::size_t i = 0; i <= ntiles; ++i) hashmarks.push_back(i * extent);
  return TiledRange1(hashmarks.begin(), hashmarks.end());
}

/// A process map whose owner of a tile is a given function of its ordinal
class FnPmap : public Pmap {
  std::function<size_type(size_type)> fn_;

 public:
  FnPmap(World& world, size_type size, std::function<size_type(size_type)> fn)
      : Pmap(world, size), fn_(std::move(fn)) {}
  size_type owner(const size_type tile) const override {
    return fn_(tile) % procs_;
  }
  bool is_local(const size_type tile) const override {
    return owner(tile) == size_type(rank_);
  }
  bool known_local_size() const override { return false; }
};

/// \return an array over \p tr on \p pmap with index-dependent elements
TArrayD make_array(World& world, const TiledRange& tr,
                   std::shared_ptr<const Pmap> pmap, const double seed) {
  TArrayD result(world, tr, std::move(pmap));
  result.init_elements([seed](const auto& idx) {
    double v = seed, scale = 1.0;
    for (auto x : idx) {
      v += scale * static_cast<double>(x + 1);
      scale *= 0.37;
    }
    return v;
  });
  return result;
}

/// \return the bytes all ranks have sent so far
std::size_t bytes_sent(World& world) {
  world.gop.fence();
  std::size_t n = madness::RMI::get_stats().nbyte_sent;
  world.gop.sum(n);
  return n;
}

/// \return the largest magnitude of an element of \p a - \p b
double max_abs_diff(const TArrayD& a, const TArrayD& b,
                    const std::string& annot) {
  TArrayD d;
  d(annot) = a(annot) - b(annot);
  return d(annot).abs_max().get();
}

/// the pair (h1,h2) of a slab ordinal counted as (h2,h1), and the reverse
struct Pairs {
  std::size_t n1, n2;
  std::size_t h1h2_of_h2h1(std::size_t o) const {
    return (o % n1) * n2 + o / n1;
  }
};

}  // namespace

BOOST_AUTO_TEST_SUITE(slab_pmap_suite)

BOOST_AUTO_TEST_CASE(table) {
  World& world = get_default_world();
  const std::size_t P = world.size();
  // 7 slabs of 3 tiles, every rank its own plane, slabs dealt by a table
  auto base = std::make_shared<FnPmap>(
      world, 3ul, [](std::size_t) { return std::size_t(0); });
  auto groups = std::make_shared<std::vector<std::size_t>>();
  for (std::size_t h = 0; h != 7; ++h) groups->push_back((3 * h + 1) % P);
  detail::SlabbedPmap pmap(world, base, 7ul, P, 1ul, groups);
  std::size_t nlocal = 0;
  for (std::size_t t = 0; t != pmap.size(); ++t) {
    BOOST_CHECK_EQUAL(pmap.owner(t), (*groups)[t / 3]);
    BOOST_CHECK_EQUAL(pmap.is_local(t),
                      (*groups)[t / 3] == std::size_t(world.rank()));
    if (pmap.is_local(t)) ++nlocal;
  }
  BOOST_CHECK(pmap.known_local_size());
  BOOST_CHECK_EQUAL(pmap.local_size(), nlocal);
  std::size_t niter = 0;
  for (auto it = pmap.begin(); it != pmap.end(); ++it) {
    BOOST_CHECK(pmap.is_local(*it));
    ++niter;
  }
  BOOST_CHECK_EQUAL(niter, nlocal);
}

// every free outer index is a single tile: a slab is one result tile and is
// computed, and left, on the rank the slab map names
BOOST_AUTO_TEST_CASE(single_tile_slabs) {
  World& world = get_default_world();
  const std::size_t P = world.size();
  const std::size_t n1 = 3, n2 = 5, nk = 4;
  const auto h1 = make_tr1(n1, 2), h2 = make_tr1(n2, 3), k = make_tr1(nk, 4),
             c = make_tr1(1, 6), r = make_tr1(1, 5);
  // a rule for the owner of a pair that is not round-robin in either order
  auto owner_of_pair = [](std::size_t p1, std::size_t p2) {
    return (5 * p1 + 3 * p2 * p2 + 1);
  };
  const Pairs pairs{n1, n2};
  // maps of the pairs in both orders, and of arrays keyed on the pair
  auto key_h1h2 = std::make_shared<FnPmap>(world, n1 * n2, [=](std::size_t o) {
    return owner_of_pair(o / n2, o % n2);
  });
  auto key_h2h1 = std::make_shared<FnPmap>(world, n1 * n2, [=](std::size_t o) {
    return owner_of_pair(o % n1, o / n1);
  });
  auto key_A = std::make_shared<FnPmap>(world, n1 * n2 * nk, [=](auto o) {
    return owner_of_pair(o / (n2 * nk), (o / nk) % n2);
  });  // (h1,h2,k,r): r has one tile
  auto key_B = std::make_shared<FnPmap>(world, nk * n1 * n2, [=](auto o) {
    return owner_of_pair((o / n2) % n1, o % n2);
  });  // (k,h1,h2,c)
  const TiledRange trA{h1, h2, k, r}, trB{k, h1, h2, c};
  auto A = make_array(world, trA, key_A, 0.5);
  auto B = make_array(world, trB, key_B, -1.5);
  auto A0 = make_array(world, trA, nullptr, 0.5);
  auto B0 = make_array(world, trB, nullptr, -1.5);

  TArrayD ref;
  ref("h1,h2,r,c") = A0("h1,h2,k,r") * B0("k,h1,h2,c");
  const double scale = ref("h1,h2,r,c").abs_max().get();

  // the result in the canonical order of the fused modes
  {
    TArrayD C;
    const auto before = bytes_sent(world);
    C("h1,h2,r,c") = (A("h1,h2,k,r") * B("k,h1,h2,c")).set_slab_pmap(key_h1h2);
    const auto after = bytes_sent(world);
    for (std::size_t t = 0; t != n1 * n2; ++t)
      BOOST_CHECK_EQUAL(C.pmap()->owner(t), key_h1h2->owner(t));
    BOOST_CHECK_LT(max_abs_diff(C, ref, "h1,h2,r,c"), 1e-12 * scale);
    // operands and result share the rule: nothing is sent
    BOOST_CHECK_EQUAL(after - before, 0ul);
  }
  // the fused modes swapped in the result: the slab map is over (h2,h1)
  {
    TArrayD C;
    const auto before = bytes_sent(world);
    C("h2,h1,c,r") = (A("h1,h2,k,r") * B("k,h1,h2,c")).set_slab_pmap(key_h2h1);
    const auto after = bytes_sent(world);
    for (std::size_t t = 0; t != n1 * n2; ++t)
      BOOST_CHECK_EQUAL(C.pmap()->owner(t), key_h2h1->owner(t));
    TArrayD Cp;
    Cp("h1,h2,r,c") = C("h2,h1,c,r");
    BOOST_CHECK_LT(max_abs_diff(Cp, ref, "h1,h2,r,c"), 1e-12 * scale);
    BOOST_CHECK_EQUAL(after - before, 0ul);
    (void)pairs;
  }
  // fewer slabs than ranks, and operands that are not on the map
  {
    const auto g1 = make_tr1(1, 2), g2 = make_tr1(2, 3);
    auto key = std::make_shared<FnPmap>(
        world, 2ul, [=](std::size_t o) { return P - 1 - (o % P); });
    auto A1 = make_array(world, TiledRange{g1, g2, k, r}, nullptr, 0.25);
    auto B1 = make_array(world, TiledRange{k, g1, g2, c}, nullptr, 2.0);
    TArrayD ref1, C;
    ref1("h1,h2,r,c") = A1("h1,h2,k,r") * B1("k,h1,h2,c");
    C("h1,h2,r,c") = (A1("h1,h2,k,r") * B1("k,h1,h2,c")).set_slab_pmap(key);
    for (std::size_t t = 0; t != 2; ++t)
      BOOST_CHECK_EQUAL(C.pmap()->owner(t), key->owner(t));
    BOOST_CHECK_LT(max_abs_diff(C, ref1, "h1,h2,r,c"),
                   1e-12 * ref1("h1,h2,r,c").abs_max().get());
  }
  // a requested result map still decides where the result is left
  {
    auto blocked = std::make_shared<detail::BlockedPmap>(world, n1 * n2);
    TArrayD C;
    C("h1,h2,r,c") = (A("h1,h2,k,r") * B("k,h1,h2,c"))
                         .set_slab_pmap(key_h1h2)
                         .set_pmap(blocked);
    for (std::size_t t = 0; t != n1 * n2; ++t)
      BOOST_CHECK_EQUAL(C.pmap()->owner(t), blocked->owner(t));
    BOOST_CHECK_LT(max_abs_diff(C, ref, "h1,h2,r,c"), 1e-12 * scale);
  }
}

// slabs of several result tiles: the slab map picks the group of ranks, the
// tiles of a slab are laid out by the group's process grid
BOOST_AUTO_TEST_CASE(multi_tile_slabs) {
  World& world = get_default_world();
  const std::size_t n1 = 6, nk = 3, nr = 2, nc = 2;
  const auto h1 = make_tr1(n1, 2), k = make_tr1(nk, 4), r = make_tr1(nr, 3),
             c = make_tr1(nc, 5);
  auto key = std::make_shared<FnPmap>(world, n1,
                                      [](std::size_t o) { return 7 * o + 2; });
  auto A = make_array(world, TiledRange{h1, r, k}, nullptr, 0.5);
  auto B = make_array(world, TiledRange{h1, k, c}, nullptr, -1.5);
  TArrayD ref;
  ref("h1,r,c") = A("h1,r,k") * B("h1,k,c");
  const double scale = ref("h1,r,c").abs_max().get();
  for (const char* target : {"h1,r,c", "r,h1,c", "c,r,h1"}) {
    TArrayD C, Cp;
    C(target) = (A("h1,r,k") * B("h1,k,c")).set_slab_pmap(key);
    Cp("h1,r,c") = C(target);
    BOOST_CHECK_LT(max_abs_diff(Cp, ref, "h1,r,c"), 1e-12 * scale);
  }
  // operands kept by the same rule: each slab, though it has several result
  // tiles, is computed by the rank that holds it, and nothing is sent
  {
    auto key_A = std::make_shared<FnPmap>(
        world, n1 * nr * nk,
        [=](std::size_t o) { return 7 * (o / (nr * nk)) + 2; });
    auto key_B = std::make_shared<FnPmap>(
        world, n1 * nk * nc,
        [=](std::size_t o) { return 7 * (o / (nk * nc)) + 2; });
    auto Ak = make_array(world, TiledRange{h1, r, k}, key_A, 0.5);
    auto Bk = make_array(world, TiledRange{h1, k, c}, key_B, -1.5);
    TArrayD C;
    const auto before = bytes_sent(world);
    C("h1,r,c") = (Ak("h1,r,k") * Bk("h1,k,c")).set_slab_pmap(key);
    const auto after = bytes_sent(world);
    for (std::size_t t = 0; t != n1 * nr * nc; ++t)
      BOOST_CHECK_EQUAL(C.pmap()->owner(t), key->owner(t / (nr * nc)));
    BOOST_CHECK_LT(max_abs_diff(C, ref, "h1,r,c"), 1e-12 * scale);
    BOOST_CHECK_EQUAL(after - before, 0ul);
  }
  // a slab map of the wrong size is ignored
  {
    auto bad = std::make_shared<FnPmap>(world, n1 + 1,
                                        [](std::size_t o) { return o; });
    TArrayD C;
    C("h1,r,c") = (A("h1,r,k") * B("h1,k,c")).set_slab_pmap(bad);
    BOOST_CHECK_LT(max_abs_diff(C, ref, "h1,r,c"), 1e-12 * scale);
  }
}

BOOST_AUTO_TEST_SUITE_END()
