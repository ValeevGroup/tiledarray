/*
 * This file is a part of TiledArray.
 * Copyright (C) 2015  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_SPECIAL_KRONECKER_DELTA_H__INCLUDED
#define TILEDARRAY_SPECIAL_KRONECKER_DELTA_H__INCLUDED

#include <memory>
#include <tuple>

#include <tiledarray_fwd.h>

#include <TiledArray/external/madness.h>

// Array class
#include <TiledArray/permutation.h>
#include <TiledArray/tensor.h>
#include <TiledArray/tile.h>
#include <TiledArray/tile_op/tile_interface.h>

// Array policy classes
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/policies/sparse_policy.h>

namespace TiledArray {

// clang-format off
/// *generalized* (asymmetric) Kronecker delta

/// *generalized* (asymmetric) Kronecker delta is a product of `N` ordinary
/// Kronecker deltas
/// Definition: `KroneckerDeltaTile(b,k) = (b==k) ? 1 : 0` and
/// `KroneckerDeltaTile(b1,b2,...bN,k1,k2,...kN) = KroneckerDeltaTile(b0,k0) KroneckerDeltaTile(b1,k1) ... KroneckerDeltaTile(bN,kN)`.
/// The implicit layout is hardwired to `b0,b1,b2,...,bN,k0,k1,k2,...,kN` since the intended use is for taking slices.
// clang-format on
class KroneckerDeltaTile {
 public:
  // Concept typedefs
  typedef Range range_type;  // range type
  typedef int value_type;    // Element type
  typedef value_type
      numeric_type;  // The scalar type that is compatible with value_type
  typedef size_t size_type;  // Size type

 private:
  range_type range_;  // range_.rank() = 2*N
  bool empty_;

 public:
  /// default constructor makes an empty tile
  KroneckerDeltaTile() : empty_(true) {}

  /// Productive ctor 1
  /// \param[in] range the range of the tile, by definition must be even-order
  /// such that the number of Kronecker deltas `N` is `range.rank() / 2` \pre
  /// `range.rank() % 2 == 1`
  KroneckerDeltaTile(const range_type& range)
      : range_(range), empty_(is_empty(range_)) {
    TA_ASSERT(range.rank() % 2 == 0);
  }

  /// copy constructor (= deep copy)
  KroneckerDeltaTile(const KroneckerDeltaTile&) = default;

  /// assignment
  KroneckerDeltaTile& operator=(const KroneckerDeltaTile& other) = default;

  /// clone = copy
  KroneckerDeltaTile clone() const {
    KroneckerDeltaTile result(*this);
    return result;
  }

  range_type range() const { return range_; }

  bool empty() const { return empty_; }

  /// \return the number of Kronecker deltas in the product
  unsigned int N() const { return range_.rank() / 2; }

  /// MADNESS compliant serialization
  template <typename Archive>
  void serialize(Archive& ar) {
    std::cout << "KroneckerDelta::serialize not implemented by design!"
              << std::endl;
    abort();  // should never travel
  }

 private:
  /// @return false if contains any nonzeros
  static bool is_empty(const range_type& range) {
    bool empty = false;
    TA_ASSERT(range.rank() % 2 == 0);
    const auto N = range.rank() / 2;
    auto lobound = range.lobound_data();
    auto upbound = range.upbound_data();
    for (auto i = 0; i != N && not empty; ++i) {
      const auto lo = std::max(lobound[i], lobound[i + N]);
      const auto up = std::min(upbound[i], upbound[i + N]);
      empty = lo >= up;
    }
    return empty;
  }

};  // class KroneckerDeltaTile

// these are to satisfy interfaces, but not needed, actually

// Sum of hyper diagonal elements
typename KroneckerDeltaTile::numeric_type trace(const KroneckerDeltaTile& arg);
// foreach(i) result += arg[i]
typename KroneckerDeltaTile::numeric_type sum(const KroneckerDeltaTile& arg);
// foreach(i) result *= arg[i]
typename KroneckerDeltaTile::numeric_type product(
    const KroneckerDeltaTile& arg);
// foreach(i) result += arg[i] * arg[i]
typename KroneckerDeltaTile::numeric_type squared_norm(
    const KroneckerDeltaTile& arg);
// foreach(i) result = min(result, arg[i])
typename KroneckerDeltaTile::numeric_type min(const KroneckerDeltaTile& arg);
// foreach(i) result = max(result, arg[i])
typename KroneckerDeltaTile::numeric_type max(const KroneckerDeltaTile& arg);
// foreach(i) result = abs_min(result, arg[i])
typename KroneckerDeltaTile::numeric_type abs_min(
    const KroneckerDeltaTile& arg);
// foreach(i) result = abs_max(result, arg[i])
typename KroneckerDeltaTile::numeric_type abs_max(
    const KroneckerDeltaTile& arg);

// Permutation operation

// returns a tile for which result[perm ^ i] = tile[i]
template <typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
KroneckerDeltaTile permute(const KroneckerDeltaTile& tile, const Perm& perm) {
  abort();
}

// dense_result[i] = dense_arg1[i] * sparse_arg2[i]
template <typename T>
Tensor<T> mult(const KroneckerDeltaTile& arg1, const Tensor<T>& arg2) {
  abort();
}
// dense_result[perm ^ i] = dense_arg1[i] * sparse_arg2[i]
template <typename T, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
Tensor<T> mult(const KroneckerDeltaTile& arg1, const Tensor<T>& arg2,
               const Perm& perm) {
  abort();
}

// dense_result[i] *= sparse_arg1[i]
template <typename T>
Tensor<T>& mult_to(Tensor<T>& result, const KroneckerDeltaTile& arg1) {
  abort();
}

template <typename T>
Tensor<T>&& mult_to(Tensor<T>&& result, const KroneckerDeltaTile& arg1) {
  abort();
}

// dense_result[i] = binary(dense_arg1[i], sparse_arg2[i], op)
template <typename T, typename Op>
Tensor<T> binary(const KroneckerDeltaTile& arg1, const Tensor<T>& arg2,
                 Op&& op) {
  abort();
}
// dense_result[perm ^ i] = binary(dense_arg1[i], sparse_arg2[i], op)
template <typename T, typename Op, typename Perm,
          typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
Tensor<T> binary(const KroneckerDeltaTile& arg1, const Tensor<T>& arg2, Op&& op,
                 const Perm& perm) {
  abort();
}

// Contraction operations

// GEMM operation with fused indices as defined by gemm_config:
// dense_result[i,j] += dense_arg1[i,k] * sparse_arg2[k,j]
template <typename T>
void gemm(Tensor<T>& result, const KroneckerDeltaTile& arg1,
          const Tensor<T>& arg2, const typename Tensor<T>::numeric_type factor,
          const math::GemmHelper& gemm_config) {
  // preconditions:
  // 1. implemented only kronecker transform (every mode of arg2 is contracted
  // with the matching mode of arg1)
  TA_ASSERT((gemm_config.result_rank() == gemm_config.right_rank() &&
             gemm_config.left_rank() ==
                 gemm_config.result_rank() + gemm_config.right_rank()));

  auto arg1_range = arg1.range();
  auto arg2_range = arg2.range();
  // if result is empty, initialize it
  const auto& result_range =
      result.empty()
          ? gemm_config.make_result_range<Range>(arg1_range, arg2_range)
          : result.range();
  if (result.empty()) result = Tensor<T>(result_range, 0);

  auto result_data = result.data();
  auto arg1_extents = arg1_range.extent_data();
  auto arg2_data = arg2.data();
  auto arg2_volume = arg2_range.volume();

  TA_ASSERT(!arg1.empty());
  const auto N = arg1.N();
  auto max = [&](const auto* v1, const auto* v2) {
    TA::Index result(N);
    for (auto i = 0; i != N; ++i) result[i] = std::max(v1[i], v2[i]);
    return result;
  };
  auto min = [&](const auto* v1, const auto* v2) {
    TA::Index result(N);
    for (auto i = 0; i != N; ++i) result[i] = std::min(v1[i], v2[i]);
    return result;
  };
  const auto read_lobound =
      max(result_range.lobound_data(), arg2_range.lobound_data());
  const auto read_upbound =
      min(result_range.upbound_data(), arg2_range.upbound_data());
  result.block(read_lobound, read_upbound) =
      arg2.block(read_lobound, read_upbound);
}

// GEMM operation with fused indices as defined by gemm_config:
// dense_result[b0,..bN] = kronecker_arg1[b1,...bN,k1,...kN] *
// dense_arg2[k1,...kN]
template <typename T>
Tensor<T> gemm(const KroneckerDeltaTile& arg1, const Tensor<T>& arg2,
               const typename Tensor<T>::numeric_type factor,
               const math::GemmHelper& gemm_config) {
  Tensor<T> result;
  gemm(result, arg1, arg2, factor, gemm_config);
  return result;
}

namespace detail {

/// \brief computes shape data (i.e. Frobenius norms of the tiles) for a
/// DistArray of KroneckerDeltaTile
/// \param trange a TiledRange of the result
/// \return a Tensor<float> containing the Frobenius norms of
///         the tiles of a DistArray of KroneckerDeltaTile's
/// \note Unlike diagonal_shape() which works for hyperdiagonal tensor with
/// `N` modes (`t(i,i,...i) = 1`), this works for product of `N`
/// Kroneckers (`t(i1,...iN,i1,...iN) = 1`, with `N` = `trange.rank() / 2`).
inline Tensor<float> kronecker_shape(TiledRange const& trange) {
  // preconditions
  TA_ASSERT(trange.rank() % 2 == 0);

  Tensor<float> shape(trange.tiles_range(), 0.0);
  const auto N = trange.rank() / 2;

  // for every bra-ket pair of modes compute list of
  // {bra tile index, ket tile index, number of nonzeros}
  using bkn_type = std::tuple<std::size_t, std::size_t, std::size_t>;
  std::vector<std::vector<bkn_type>> bkns(N);
  for (auto d = 0; d != N; ++d) {
    auto& bkn = bkns[d];
    auto& bra_tr1 = trange.dim(d);
    auto& ket_tr1 = trange.dim(d + N);
    auto eidx = std::max(bra_tr1.elements_range().lobound(),
                         ket_tr1.elements_range().lobound());
    const auto eidx_fence = std::min(bra_tr1.elements_range().upbound(),
                                     ket_tr1.elements_range().upbound());
    while (eidx < eidx_fence) {
      const auto bra_tile_idx = bra_tr1.element_to_tile(eidx);
      const auto& bra_tile = bra_tr1.tile(bra_tile_idx);
      auto ket_tile_idx = ket_tr1.element_to_tile(eidx);
      const auto& ket_tile = ket_tr1.tile(ket_tile_idx);
      // closest tile boundary
      const auto next_eidx = std::min(bra_tile.upbound(), ket_tile.upbound());
      bkn.emplace_back(bra_tile_idx, ket_tile_idx, next_eidx - eidx);
      eidx = next_eidx;
    }
  }

  // number of nonzero tiles per mode
  TA::Index nnz_tiles(N);
  for (auto d = 0; d != N; ++d) nnz_tiles[d] = bkns[d].size();
  TA::Range nztiles_range(nnz_tiles);
  TA::Index tile_idx(2 * N);
  for (auto&& nztile : nztiles_range) {
    std::size_t nnz_elements = 1;
    for (auto d = 0; d != N; ++d) {
      tile_idx[d] = std::get<0>(bkns[d][nztile[d]]);
      tile_idx[d + N] = std::get<1>(bkns[d][nztile[d]]);
      nnz_elements *= std::get<2>(bkns[d][nztile[d]]);
    }
    shape(tile_idx) = std::sqrt(nnz_elements);
  }

  return shape;
}

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_TEST_SPARSE_TILE_H__INCLUDED
