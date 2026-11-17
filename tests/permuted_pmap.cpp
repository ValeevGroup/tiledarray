/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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

#include "TiledArray/pmap/permuted_pmap.h"
#include "TiledArray/pmap/blocked_pmap.h"
#include "global_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <string>
#include <vector>

using namespace TiledArray;

namespace {

TiledRange1 make_tr1(std::size_t ntiles, std::size_t extent) {
  std::vector<std::size_t> hashmarks;
  for (std::size_t i = 0; i <= ntiles; ++i) hashmarks.push_back(i * extent);
  return TiledRange1(hashmarks.begin(), hashmarks.end());
}

/// \return whether every tile of \p result , whose modes are those of \p ref
/// permuted by \p ref_to_result , has the owner and the contents of its
/// counterpart in \p ref
bool same_owners_and_values(const TArrayD& ref, const TArrayD& result,
                            const Permutation& ref_to_result) {
  const auto& ref_tiles = ref.trange().tiles_range();
  const auto& result_tiles = result.trange().tiles_range();
  bool ok = true;
  for (auto&& idx : ref_tiles) {
    const auto ref_ord = ref_tiles.ordinal(idx);
    const auto result_ord = result_tiles.ordinal(ref_to_result * idx);
    ok = ok && (result.pmap()->owner(result_ord) == ref.pmap()->owner(ref_ord));
    if (ref.is_local(ref_ord) && result.is_local(result_ord)) {
      const Tensor<double> expected =
          ref.find(ref_ord).get().permute(ref_to_result);
      const Tensor<double> actual = result.find(result_ord).get();
      ok = ok && (expected.range() == actual.range());
      if (ok)
        for (std::size_t i = 0; i < expected.size(); ++i)
          ok = ok && (expected[i] == actual[i]);
    }
  }
  return ok;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(permuted_pmap_suite)

BOOST_AUTO_TEST_CASE(owner) {
  auto& world = *GlobalFixture::world;
  const Range source_tiles(std::vector<std::size_t>{2, 3, 4});
  auto base =
      std::make_shared<detail::BlockedPmap>(world, source_tiles.volume());
  for (const Permutation& perm :
       {Permutation{1, 0, 2}, Permutation{2, 0, 1}, Permutation{1, 2, 0},
        Permutation{0, 2, 1}, Permutation{2, 1, 0}}) {
    const Range target_tiles = perm * source_tiles;
    detail::PermutedPmap pmap(world, base, target_tiles, perm);
    BOOST_CHECK_EQUAL(pmap.size(), source_tiles.volume());
    BOOST_CHECK_EQUAL(pmap.procs(), base->procs());
    BOOST_CHECK_EQUAL(pmap.rank(), base->rank());
    BOOST_CHECK_EQUAL(pmap.local_size(), base->local_size());

    std::vector<std::size_t> local;
    for (auto&& idx : source_tiles) {
      const auto source_ord = source_tiles.ordinal(idx);
      const auto target_ord = target_tiles.ordinal(perm * idx);
      BOOST_CHECK_EQUAL(pmap.owner(target_ord), base->owner(source_ord));
      BOOST_CHECK_EQUAL(pmap.is_local(target_ord), base->is_local(source_ord));
      if (base->is_local(source_ord)) local.push_back(target_ord);
    }
    std::sort(local.begin(), local.end());
    const std::vector<std::size_t> iterated(pmap.begin(), pmap.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(iterated.begin(), iterated.end(),
                                  local.begin(), local.end());
  }
}

// The result of a product, in whatever mode order it is asked for, is owned
// tile for tile by the ranks that own it in the order the product computes
// it in: delivering it moves no tile to another rank.

BOOST_AUTO_TEST_CASE(general_product_result) {
  auto& world = *GlobalFixture::world;
  // h fused, k contracted, r and c free
  TArrayD A(world, TiledRange{make_tr1(5, 2), make_tr1(3, 3), make_tr1(2, 4)});
  TArrayD B(world, TiledRange{make_tr1(5, 2), make_tr1(2, 4), make_tr1(4, 2)});
  A.fill_random();
  B.fill_random();
  TArrayD ref;
  ref("h,r,c") = A("h,r,k") * B("h,k,c");
  const std::vector<std::pair<std::string, Permutation>> targets = {
      {"r,h,c", Permutation{1, 0, 2}},
      {"r,c,h", Permutation{2, 0, 1}},
      {"c,h,r", Permutation{1, 2, 0}}};
  for (auto&& [target, ref_to_result] : targets) {
    BOOST_TEST_CONTEXT("result(" << target << ")") {
      TArrayD C;
      C(target) = A("h,r,k") * B("h,k,c");
      world.gop.fence();
      BOOST_CHECK(same_owners_and_values(ref, C, ref_to_result));
    }
  }
  // free indices of one tile each: every slab is computed on one rank
  TArrayD A1(world, TiledRange{make_tr1(3, 2), make_tr1(4, 2), make_tr1(2, 4),
                               make_tr1(1, 5)});
  TArrayD B1(world, TiledRange{make_tr1(3, 2), make_tr1(4, 2), make_tr1(2, 4),
                               make_tr1(1, 6)});
  A1.fill_random();
  B1.fill_random();
  TArrayD ref1, C1;
  ref1("g,h,r,c") = A1("g,h,k,r") * B1("g,h,k,c");
  C1("g,r,h,c") = A1("g,h,k,r") * B1("g,h,k,c");
  world.gop.fence();
  BOOST_CHECK(same_owners_and_values(ref1, C1, Permutation{0, 2, 1, 3}));
}

BOOST_AUTO_TEST_CASE(contraction_result) {
  auto& world = *GlobalFixture::world;
  TArrayD A(world, TiledRange{make_tr1(4, 3), make_tr1(3, 2), make_tr1(2, 4)});
  TArrayD B(world, TiledRange{make_tr1(2, 4), make_tr1(5, 2)});
  A.fill_random();
  B.fill_random();
  TArrayD ref;
  ref("r,s,c") = A("r,s,k") * B("k,c");
  const std::vector<std::pair<std::string, Permutation>> targets = {
      {"s,r,c", Permutation{1, 0, 2}},
      {"c,r,s", Permutation{1, 2, 0}},
      {"r,c,s", Permutation{0, 2, 1}}};
  for (auto&& [target, ref_to_result] : targets) {
    BOOST_TEST_CONTEXT("result(" << target << ")") {
      TArrayD C;
      C(target) = A("r,s,k") * B("k,c");
      world.gop.fence();
      BOOST_CHECK(same_owners_and_values(ref, C, ref_to_result));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
