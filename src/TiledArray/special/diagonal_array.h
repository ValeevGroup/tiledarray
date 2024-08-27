/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2016  Virginia Tech
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
 *  Drew Lewis
 *  Department of Chemistry, Virginia Tech
 *
 *  diagonal_array.h
 *  Nov 30, 2016
 *
 */

#ifndef TILEDARRAY_SPECIALARRAYS_DIAGONAL_ARRAY_H__INCLUDED
#define TILEDARRAY_SPECIALARRAYS_DIAGONAL_ARRAY_H__INCLUDED

#include <TiledArray/dist_array.h>
#include <TiledArray/range.h>
#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>

#include <range/v3/view/repeat.hpp>

#include <vector>

namespace TiledArray {
namespace detail {

/// \brief Computes a range of the diagonal elements (if any) in a rank-d Range

/// computes [min,max) describing the diagonal elements that rank-d Range
/// contains; if the input Range contains no diagonal elements this return an
/// empty Range
/// \param[in] rng an input (rank-d) Range
/// \return the range of diagonal elements, as a rank-1 Range
inline Range1 diagonal_range(Range const &rng) {
  const auto rank = rng.rank();
  TA_ASSERT(rng.rank() > 0);
  auto lo = rng.lobound_data();
  auto up = rng.upbound_data();

  // Determine the largest lower index and the smallest upper index
  auto max_low = *std::max_element(lo, lo + rank);
  auto min_up = *std::min_element(up, up + rank);

  // If the max small elem is less than the min large elem then a diagonal
  // elem is in this tile;
  if (max_low < min_up) {
    return Range1{max_low, min_up};
  } else {
    return Range1{};
  }
}

/// \brief computes shape data (i.e. Frobenius norms of the tiles) for a
/// non-constant diagonal tensor
/// \tparam RandomAccessIterator an iterator over
/// the range of diagonal elements
/// \tparam Sentinel sentinel type for the range of diagonal elements
/// \param[in] trange a TiledRange of the result
/// \param[in] diagonals_begin the begin iterator of the range of the diagonals
/// \param[in] diagonals_end the end iterator of the range of the diagonals; if
/// not given, default initialized and thus will not be checked
/// \return a Tensor<float> containing the Frobenius norms of the tiles of
/// a DistArray with \p val on the diagonal and zeroes elsewhere
template <typename RandomAccessIterator, typename Sentinel>
std::enable_if_t<is_iterator<RandomAccessIterator>::value, Tensor<float>>
diagonal_shape(TiledRange const &trange, RandomAccessIterator diagonals_begin,
               Sentinel diagonals_end = {}) {
  bool have_end = false;
  if constexpr (detail::is_equality_comparable_v<Sentinel>) {
    have_end = diagonals_end != Sentinel{};
  }

  Tensor<float> shape(trange.tiles_range(), 0.0);

  const auto rank = trange.rank();
  TA_ASSERT(rank > 0);
  const auto *lobound = trange.elements_range().lobound_data();
  const auto diag_lobound = *std::max_element(lobound, lobound + rank);
  const auto *upbound = trange.elements_range().upbound_data();
  const auto diag_upbound = *std::min_element(upbound, upbound + rank);

  auto diag_elem = diag_lobound;
  // the diagonal elements will never be larger than the length of the
  // shortest dimension
  while (diag_elem < diag_upbound) {
    // Get the tile index corresponding to the current diagonal_elem
    auto tile_idx = trange.element_to_tile(Index(rank, diag_elem));
    auto tile_range = trange.make_tile_range(tile_idx);

    // Compute the range of diagonal elements in the tile
    auto d_range = diagonal_range(tile_range);
    TA_ASSERT(d_range != Range1{});
    TA_ASSERT(diag_elem == d_range.lobound());
    const auto beg = d_range.lobound();
    const auto end = d_range.upbound();
    if (have_end) {
      if constexpr (detail::are_less_than_comparable_v<RandomAccessIterator,
                                                       Sentinel>) {
        TA_ASSERT(diagonals_begin + beg < diagonals_end);
      }
      if constexpr (detail::are_less_than_or_equal_comparable_v<
                        RandomAccessIterator, Sentinel>) {
        TA_ASSERT(diagonals_begin + end <= diagonals_end);
      }
    }

    auto t_norm = std::accumulate(diagonals_begin + beg, diagonals_begin + end,
                                  0.0, [](const auto &sum, const auto &val) {
                                    const auto abs_val = std::abs(val);
                                    return sum + abs_val * abs_val;
                                  });
    shape(tile_idx) = std::sqrt(static_cast<float>(t_norm));

    // Update diag_elem to the next elem not in this tile
    diag_elem = end;
  }

  return shape;
}

/// \brief computes shape data (i.e. Frobenius norms of the tiles) for a
/// constant diagonal tensor
/// \tparam T a numeric type
/// \param trange a TiledRange of the result
/// \param val value of the diagonal elements
/// \return a Tensor<float> containing the Frobenius norms of
///         the tiles of a DistArray with \p val on the diagonal and
///         zeroes elsewhere
template <typename T>
Tensor<float> diagonal_shape(TiledRange const &trange, T val) {
  auto val_range = ranges::views::repeat(val);
  return diagonal_shape(trange, val_range.begin(), val_range.end());
}

/// \brief Writes tiles of a nonconstant diagonal array

/// \tparam Array a DistArray type
/// \tparam RandomAccessIterator an iterator over the range of diagonal elements
/// \param[in] A an Array object
/// \param[in] diagonals_begin the begin iterator of the range of the diagonals
template <typename Array, typename RandomAccessIterator>
std::enable_if_t<is_iterator<RandomAccessIterator>::value, void>
write_diag_tiles_to_array_rng(Array &A, RandomAccessIterator diagonals_begin) {
  using Tile = typename Array::value_type;

  A.init_tiles(
      // Task to create each tile
      [diagonals_begin](const Range &rng) {
        // Compute range of diagonal elements in the tile
        auto diags = detail::diagonal_range(rng);
        const auto rank = rng.rank();

        Tile tile(rng, 0.0);

        if (diags.volume() > 0) {  // If the tile has diagonal elems
          // Loop over the elements and write val into them
          auto diag_lo = diags.lobound();
          auto diag_hi = diags.upbound();
          auto elem_it = diagonals_begin + diag_lo;
          for (auto elem = diag_lo; elem < diag_hi; ++elem, ++elem_it) {
            tile(Index(rank, elem)) = *elem_it;
          }
        }

        return tile;
      });
}

}  // namespace detail

/// \brief Creates a non-constant diagonal DistArray

/// Creates an array whose only nonzero values are the (hyper)diagonal elements
/// (i.e. (n,n,n, ..., n) ); the values of the diagonal elements are given by an
/// input range
/// \tparam Array a DistArray type
/// \tparam RandomAccessIterator an iterator over the range of diagonal elements
/// \tparam Sentinel sentinel type for the range of diagonal elements
/// \param world The world for the array
/// \param[in] trange The trange for the array
/// \param[in] diagonals_begin the begin iterator of the range of the diagonals
/// \param[in] diagonals_end the end iterator of the range of the diagonals;
///            if not given, default initialized and thus will not be checked
/// \return a diagonal DistArray
template <typename Array, typename RandomAccessIterator, typename Sentinel>
std::enable_if_t<detail::is_iterator<RandomAccessIterator>::value, Array>
diagonal_array(World &world, TiledRange const &trange,
               RandomAccessIterator diagonals_begin,
               Sentinel diagonals_end = {}) {
  using Policy = typename Array::policy_type;

  if constexpr (detail::is_equality_comparable_v<Sentinel>) {
    if (diagonals_end != Sentinel{}) {
      auto diagonals_range = detail::diagonal_range(trange.elements_range());
      if constexpr (detail::are_less_than_comparable_v<RandomAccessIterator,
                                                       Sentinel>) {
        TA_ASSERT(diagonals_begin + diagonals_range.lobound() < diagonals_end);
      }
      if constexpr (detail::are_less_than_or_equal_comparable_v<
                        RandomAccessIterator, Sentinel>) {
        TA_ASSERT(diagonals_begin + diagonals_range.upbound() <= diagonals_end);
      }
    }
  }

  // Init the array
  if constexpr (is_dense_v<Policy>) {
    Array A(world, trange);
    detail::write_diag_tiles_to_array_rng(A, diagonals_begin);
    A.world().taskq.fence();  // ensure tasks outlive the diagonals_begin view
    return A;
  } else {
    // Compute shape and init the Array
    auto shape_norm =
        detail::diagonal_shape(trange, diagonals_begin, diagonals_end);
    using ShapeType = typename Policy::shape_type;
    ShapeType shape(shape_norm, trange);
    Array A(world, trange, shape);
    detail::write_diag_tiles_to_array_rng(A, diagonals_begin);
    A.world().taskq.fence();  // ensure tasks outlive the diagonals_begin view
    return A;
  }
  abort();  // unreachable
}

/// \brief Creates a constant diagonal DistArray

/// Creates an array whose only nonzero values are the (hyper)diagonal elements
/// (i.e. (n,n,n, ..., n) ), and they are all have the same value
/// \tparam Policy the policy type of the resulting DistArray
/// \tparam T a numeric type
/// \param world The world for the array
/// \param[in] trange The trange for the array
/// \param[in] val The value of the diagonal elements
/// \return a constant diagonal DistArray
template <typename Array, typename T = double>
Array diagonal_array(World &world, TiledRange const &trange, T val = 1) {
  auto val_range = ranges::views::repeat(val);
  return diagonal_array<Array>(world, trange, val_range.begin(),
                               val_range.end());
}

}  // namespace TiledArray

#endif  // TILEDARRAY_SPECIALARRAYS_DIAGONAL_ARRAY_H__INCLUDED
