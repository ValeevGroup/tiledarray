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

#include "TiledArray/pmap/permuted_pmap.h"
#include "TiledArray/pmap/blocked_pmap.h"
#include "global_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace TiledArray;

namespace {

using TSpArrayToT = DistArray<Tensor<Tensor<double>>, SparsePolicy>;

TiledRange1 make_tr1(std::size_t ntiles, std::size_t extent) {
  std::vector<std::size_t> hashmarks;
  for (std::size_t i = 0; i <= ntiles; ++i) hashmarks.push_back(i * extent);
  return TiledRange1(hashmarks.begin(), hashmarks.end());
}

/// \return a deterministic, index-dependent value
template <typename Index>
double pattern(const Index& idx, const double seed) {
  double v = seed;
  double scale = 1.0;
  for (auto x : idx) {
    v += scale * static_cast<double>(x + 1);
    scale *= 0.37;
  }
  return v;
}

/// \return a sparse array over \p tr whose every \p zero_stride -th tile is
/// zero and whose other tiles hold an index-dependent pattern
TSpArrayD make_sparse(World& world, const TiledRange& tr, const double seed,
                      const std::size_t zero_stride) {
  Tensor<float> norms(tr.tiles_range(), 1.0f);
  for (std::size_t ord = 0; ord < norms.size(); ord += zero_stride)
    norms.data()[ord] = 0.0f;
  TSpArrayD result(world, tr, SparseShape<float>(norms, tr));
  result.init_elements([seed](const auto& idx) { return pattern(idx, seed); });
  return result;
}

/// \return a sparse tensor-of-tensors array over \p tr whose every
/// \p zero_stride -th tile is zero; cells have extents \p inner_extents
TSpArrayToT make_sparse_tot(World& world, const TiledRange& tr,
                            const std::vector<std::size_t>& inner_extents,
                            const double seed, const std::size_t zero_stride) {
  Tensor<float> norms(tr.tiles_range(), 1.0f);
  for (std::size_t ord = 0; ord < norms.size(); ord += zero_stride)
    norms.data()[ord] = 0.0f;
  TSpArrayToT result(world, tr, SparseShape<float>(norms, tr));
  result.init_tiles([inner_extents, seed](const Range& outer_range) {
    Tensor<Tensor<double>> tile(outer_range);
    for (auto&& oix : outer_range) {
      Tensor<double> cell{Range(inner_extents)};
      const double cell_seed = pattern(oix, seed);
      for (auto&& iix : cell.range()) cell[iix] = pattern(iix, cell_seed);
      tile[oix] = cell;
    }
    return tile;
  });
  return result;
}

/// \return whether \p a and \p b agree to within \p tol in every element
bool close(const Tensor<double>& a, const Tensor<double>& b, double tol) {
  if (a.range() != b.range()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::abs(a.data()[i] - b.data()[i]) > tol * (1 + std::abs(a.data()[i])))
      return false;
  return true;
}

bool close(const Tensor<Tensor<double>>& a, const Tensor<Tensor<double>>& b,
           double tol) {
  if (a.range() != b.range()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!close(a.data()[i], b.data()[i], tol)) return false;
  return true;
}

/// \return whether every tile of \p result , whose modes are those of \p ref
/// permuted by \p ref_to_result , has the owner, the zero/nonzero status and
/// (to within a relative tolerance: the two products may be computed with
/// different GEMM operand layouts) the contents of its counterpart in \p ref
template <typename Array>
bool same_owners_and_values(const Array& ref, const Array& result,
                            const Permutation& ref_to_result) {
  constexpr double tol = 1e-12;
  const auto& ref_tiles = ref.trange().tiles_range();
  const auto& result_tiles = result.trange().tiles_range();
  bool ok = true;
  for (auto&& idx : ref_tiles) {
    const auto ref_ord = ref_tiles.ordinal(idx);
    const auto result_ord = result_tiles.ordinal(ref_to_result * idx);
    ok = ok && (result.pmap()->owner(result_ord) == ref.pmap()->owner(ref_ord));
    ok = ok && (result.is_zero(result_ord) == ref.is_zero(ref_ord));
    if (ok && ref.is_local(ref_ord) && !ref.is_zero(ref_ord)) {
      const auto expected = ref.find(ref_ord).get().permute(ref_to_result);
      const auto actual = result.find(result_ord).get();
      ok = close(expected, actual, tol);
    }
  }
  return ok;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(permuted_pmap_suite)

BOOST_AUTO_TEST_CASE(owner) {
  auto& world = *GlobalFixture::world;
  // the second range has fewer tiles than a 4-rank world has ranks, so some
  // ranks own none
  for (const Range& source_tiles : {Range(std::vector<std::size_t>{2, 3, 4}),
                                    Range(std::vector<std::size_t>{1, 1, 3})}) {
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
      BOOST_CHECK(pmap.known_local_size());
      BOOST_CHECK_EQUAL(pmap.local_size(), base->local_size());

      std::vector<std::size_t> local;
      for (auto&& idx : source_tiles) {
        const auto source_ord = source_tiles.ordinal(idx);
        const auto target_ord = target_tiles.ordinal(perm * idx);
        BOOST_CHECK_EQUAL(pmap.owner(target_ord), base->owner(source_ord));
        BOOST_CHECK_EQUAL(pmap.is_local(target_ord),
                          base->is_local(source_ord));
        if (base->is_local(source_ord)) local.push_back(target_ord);
      }
      std::sort(local.begin(), local.end());
      const std::vector<std::size_t> iterated(pmap.begin(), pmap.end());
      BOOST_CHECK_EQUAL_COLLECTIONS(iterated.begin(), iterated.end(),
                                    local.begin(), local.end());
    }
  }
}

BOOST_AUTO_TEST_CASE(invalid_arguments) {
  auto& world = *GlobalFixture::world;
  const Range tiles(std::vector<std::size_t>{2, 3, 4});
  auto base = std::make_shared<detail::BlockedPmap>(world, tiles.volume());
  // null base map
  BOOST_CHECK_THROW(
      detail::PermutedPmap(world, nullptr, tiles, Permutation{1, 0, 2}),
      TiledArray::Exception);
  // null permutation
  BOOST_CHECK_THROW(detail::PermutedPmap(world, base, tiles, Permutation{}),
                    TiledArray::Exception);
  // permutation of the wrong rank
  BOOST_CHECK_THROW(detail::PermutedPmap(world, base, tiles, Permutation{1, 0}),
                    TiledArray::Exception);
  // base map of a different number of tiles
  auto other = std::make_shared<detail::BlockedPmap>(world, tiles.volume() + 1);
  BOOST_CHECK_THROW(
      detail::PermutedPmap(world, other, tiles, Permutation{1, 0, 2}),
      TiledArray::Exception);
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

BOOST_AUTO_TEST_CASE(general_product_result_sparse) {
  auto& world = *GlobalFixture::world;
  const TiledRange tr_a{make_tr1(5, 2), make_tr1(3, 3), make_tr1(2, 4)};
  const TiledRange tr_b{make_tr1(5, 2), make_tr1(2, 4), make_tr1(4, 2)};
  auto A = make_sparse(world, tr_a, 1.0, 3);
  auto B = make_sparse(world, tr_b, 2.0, 4);
  TSpArrayD ref;
  ref("h,r,c") = A("h,r,k") * B("h,k,c");
  const std::vector<std::pair<std::string, Permutation>> targets = {
      {"r,h,c", Permutation{1, 0, 2}}, {"c,h,r", Permutation{1, 2, 0}}};
  for (auto&& [target, ref_to_result] : targets) {
    BOOST_TEST_CONTEXT("result(" << target << ")") {
      TSpArrayD C;
      C(target) = A("h,r,k") * B("h,k,c");
      world.gop.fence();
      BOOST_CHECK(same_owners_and_values(ref, C, ref_to_result));
    }
  }
}

BOOST_AUTO_TEST_CASE(general_product_result_tot) {
  auto& world = *GlobalFixture::world;
  // outer: h fused, k contracted, r and c free; inner: m fused (Hadamard)
  const TiledRange tr_a{make_tr1(4, 2), make_tr1(3, 2), make_tr1(2, 3)};
  const TiledRange tr_b{make_tr1(4, 2), make_tr1(2, 3), make_tr1(3, 2)};
  auto A = make_sparse_tot(world, tr_a, {3}, 1.0, 5);
  auto B = make_sparse_tot(world, tr_b, {3}, 2.0, 7);
  TSpArrayToT ref;
  ref("h,r,c;m") = A("h,r,k;m") * B("h,k,c;m");
  const std::vector<std::pair<std::string, Permutation>> targets = {
      {"r,h,c;m", Permutation{1, 0, 2}}, {"r,c,h;m", Permutation{2, 0, 1}}};
  for (auto&& [target, ref_to_result] : targets) {
    BOOST_TEST_CONTEXT("result(" << target << ")") {
      TSpArrayToT C;
      C(target) = A("h,r,k;m") * B("h,k,c;m");
      world.gop.fence();
      BOOST_CHECK(same_owners_and_values(ref, C, ref_to_result));
    }
  }
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

BOOST_AUTO_TEST_CASE(contraction_result_sparse) {
  auto& world = *GlobalFixture::world;
  const TiledRange tr_a{make_tr1(4, 3), make_tr1(3, 2), make_tr1(2, 4)};
  const TiledRange tr_b{make_tr1(2, 4), make_tr1(5, 2)};
  auto A = make_sparse(world, tr_a, 1.0, 3);
  auto B = make_sparse(world, tr_b, 2.0, 4);
  TSpArrayD ref;
  ref("r,s,c") = A("r,s,k") * B("k,c");
  const std::vector<std::pair<std::string, Permutation>> targets = {
      {"s,r,c", Permutation{1, 0, 2}}, {"c,r,s", Permutation{1, 2, 0}}};
  for (auto&& [target, ref_to_result] : targets) {
    BOOST_TEST_CONTEXT("result(" << target << ")") {
      TSpArrayD C;
      C(target) = A("r,s,k") * B("k,c");
      world.gop.fence();
      BOOST_CHECK(same_owners_and_values(ref, C, ref_to_result));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
