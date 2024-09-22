/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  btas.h
 *  January 19, 2018
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_BTAS_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_BTAS_H__INCLUDED

#include <limits>

#include <TiledArray/block_range.h>
#include <TiledArray/dense_shape.h>
#include <TiledArray/external/btas.h>
#include <TiledArray/pmap/replicated_pmap.h>
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/policies/sparse_policy.h>
#include <TiledArray/shape.h>
#include <TiledArray/sparse_shape.h>
#include <TiledArray/tensor.h>
#include <TiledArray/tensor/tensor_map.h>

#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

namespace TiledArray {

// clang-format off
/// Copy a block of a btas::Tensor into a TiledArray::Tensor

/// A block of btas::Tensor \c src will be copied into TiledArray::Tensor \c
/// dst. The block dimensions will be determined by the dimensions of the range
/// of \c dst .
/// \tparam T The tensor element type
/// \tparam Range_ The range type of the source btas::Tensor object
/// \tparam Storage_ The storage type of the source btas::Tensor object
/// \tparam Tensor_ A tensor type (e.g., TiledArray::Tensor or btas::Tensor,
///         optionally wrapped into TiledArray::Tile)
/// \param[in] src The source object; its subblock
///            `{dst.lobound(),dst.upbound()}`
///            will be copied to \c dst
/// \param[out] dst The object that will contain the contents of the
///             corresponding subblock of src
/// \throw TiledArray::Exception When the dimensions of \p src and \p dst do not
///        match.
// clang-format on
template <typename T, typename Range_, typename Storage_, typename Tensor_>
inline void btas_subtensor_to_tensor(
    const btas::Tensor<T, Range_, Storage_>& src, Tensor_& dst) {
  TA_ASSERT(dst.range().rank() == src.range().rank());

  const auto& src_range = src.range();
  const auto& dst_range = dst.range();
  auto src_blk_range =
      TiledArray::BlockRange(detail::make_ta_range(src_range),
                             dst_range.lobound(), dst_range.upbound());
  using std::data;
  auto src_view = TiledArray::make_const_map(data(src), src_blk_range);
  auto dst_view = TiledArray::make_map(data(dst), dst_range);

  dst_view = src_view;
}

// clang-format off
/// Copy a block of a btas::Tensor into a TiledArray::Tensor

/// A block of btas::Tensor \c src will be copied into TiledArray::Tensor \c
/// dst. The block dimensions will be determined by the dimensions of the range
/// of \c dst .
/// \tparam T The tensor element type
/// \tparam Range_ The range type of the source btas::Tensor object
/// \tparam Storage_ The storage type of the source btas::Tensor object
/// \tparam Tensor_ A tensor type (e.g., TiledArray::Tensor or btas::Tensor,
///         optionally wrapped into TiledArray::Tile)
/// \param[in] src The source object; its subblock
///            `{dst.lobound() + offset,dst.upbound() + offset}`
///            will be copied to \c dst
/// \param[out] dst The object that will contain the contents of the
///             corresponding subblock of src
/// \param[out] offset the offset to be applied to the coordinates of `dst.range()` to determine the block in \p src to be copied; this is needed if the DistArray that will contain \p dst will have a range whose lobound is different from `src.lobound()`
/// \throw TiledArray::Exception When the dimensions of \p src and \p dst do not
///        match.
// clang-format on
template <
    typename T, typename Range_, typename Storage_, typename Tensor_,
    typename IntegerRange,
    typename = std::enable_if_t<detail::is_integral_range_v<IntegerRange>>>
inline void btas_subtensor_to_tensor(
    const btas::Tensor<T, Range_, Storage_>& src, Tensor_& dst,
    IntegerRange&& offset) {
  TA_ASSERT(dst.range().rank() == src.range().rank());
  TA_ASSERT(ranges::size(offset) == src.range().rank());

  const auto& src_range = src.range();
  const auto& dst_range = dst.range();
  auto src_blk_range =
      TiledArray::BlockRange(detail::make_ta_range(src_range),
                             ranges::views::zip(dst_range.lobound(), offset) |
                                 ranges::views::transform([](auto&& i_j) {
                                   auto&& [i, j] = i_j;
                                   return i + j;
                                 }),
                             ranges::views::zip(dst_range.upbound(), offset) |
                                 ranges::views::transform([](auto&& i_j) {
                                   auto&& [i, j] = i_j;
                                   return i + j;
                                 }));
  using std::data;
  auto src_view = TiledArray::make_const_map(data(src), src_blk_range);
  auto dst_view = TiledArray::make_map(data(dst), dst_range);

  dst_view = src_view;
}

// clang-format off
/// Copy a TiledArray::Tensor into a block of a btas::Tensor

/// TiledArray::Tensor \c src will be copied into a block of btas::Tensor
/// \c dst. The block dimensions will be determined by the dimensions of the range
/// of \c src .
/// \tparam Tensor_ A tensor type (e.g., TiledArray::Tensor or btas::Tensor,
///         optionally wrapped into TiledArray::Tile)
/// \tparam T The tensor element type
/// \tparam Range_ The range type of the destination btas::Tensor object
/// \tparam Storage_ The storage type of the destination btas::Tensor object
/// \param[in] src The source object whose contents will be copied into
///            a subblock of \c dst
/// \param[out] dst The destination object; its subblock
///             `{src.lobound(),src.upbound()}` will be
///             overwritten with the content of \c src
/// \throw TiledArray::Exception When the dimensions
///        of \c src and \c dst do not match.
// clang-format on
template <typename Tensor_, typename T, typename Range_, typename Storage_>
inline void tensor_to_btas_subtensor(const Tensor_& src,
                                     btas::Tensor<T, Range_, Storage_>& dst) {
  TA_ASSERT(dst.range().rank() == src.range().rank());

  const auto& src_range = src.range();
  const auto& dst_range = dst.range();
  auto dst_blk_range =
      TiledArray::BlockRange(detail::make_ta_range(dst_range),
                             src_range.lobound(), src_range.upbound());
  using std::data;
  auto src_view = TiledArray::make_const_map(data(src), src_range);
  auto dst_view = TiledArray::make_map(data(dst), dst_blk_range);

  dst_view = src_view;
}

// clang-format off
/// Copy a TiledArray::Tensor into a block of a btas::Tensor

/// TiledArray::Tensor \c src will be copied into a block of btas::Tensor
/// \c dst. The block dimensions will be determined by the dimensions of the range
/// of \c src .
/// \tparam Tensor_ A tensor type (e.g., TiledArray::Tensor or btas::Tensor,
///         optionally wrapped into TiledArray::Tile)
/// \tparam T The tensor element type
/// \tparam Range_ The range type of the destination btas::Tensor object
/// \tparam Storage_ The storage type of the destination btas::Tensor object
/// \param[in] src The source object whose contents will be copied into
///            a subblock of \c dst
/// \param[out] dst The destination object; its subblock
///             `{src.lobound()+offset,src.upbound()+offset}` will be
///             overwritten with the content of \c src
/// \param[out] offset the offset to be applied to the coordinates of `src.range()` to determine the block in \p dst to be copied; this is needed if the DistArray that contains \p src has a range whose lobound is different from `dst.lobound()`
/// \throw TiledArray::Exception When the dimensions
///        of \c src and \c dst do not match.
// clang-format on
template <
    typename Tensor_, typename T, typename Range_, typename Storage_,
    typename IntegerRange,
    typename = std::enable_if_t<detail::is_integral_range_v<IntegerRange>>>
inline void tensor_to_btas_subtensor(const Tensor_& src,
                                     btas::Tensor<T, Range_, Storage_>& dst,
                                     IntegerRange&& offset) {
  TA_ASSERT(dst.range().rank() == src.range().rank());
  TA_ASSERT(ranges::size(offset) == src.range().rank());

  const auto& src_range = src.range();
  const auto& dst_range = dst.range();
  auto dst_blk_range =
      TiledArray::BlockRange(detail::make_ta_range(dst_range),
                             ranges::views::zip(src_range.lobound(), offset) |
                                 ranges::views::transform([](auto&& i_j) {
                                   auto&& [i, j] = i_j;
                                   return i + j;
                                 }),
                             ranges::views::zip(src_range.upbound(), offset) |
                                 ranges::views::transform([](auto&& i_j) {
                                   auto&& [i, j] = i_j;
                                   return i + j;
                                 }));
  using std::data;
  auto src_view = TiledArray::make_const_map(data(src), src_range);
  auto dst_view = TiledArray::make_map(data(dst), dst_blk_range);

  dst_view = src_view;
}

namespace detail {

/// Task function for converting btas::Tensor subblock to a
/// TiledArray::DistArray

/// \tparam DistArray_ a TiledArray::DistArray type
/// \tparam BTAS_Tensor_ a btas::Tensor type
/// \param src The btas::Tensor object whose block will be copied
/// \param dst The array that will hold the result
/// \param i The index of the tile to be copied
/// \param counter The task counter
/// \internal OK to use bare ptrs as args as long as the user blocks on the
/// counter.
template <typename DistArray_, typename BTAS_Tensor_>
void counted_btas_subtensor_to_tensor(const BTAS_Tensor_* src, DistArray_* dst,
                                      const typename Range::index_type i,
                                      madness::AtomicInt* counter) {
  typename DistArray_::value_type tensor(dst->trange().make_tile_range(i));
  auto offset = ranges::views::zip(ranges::views::all(src->range().lobound()),
                                   dst->trange().elements_range().lobound()) |
                ranges::views::transform([](const auto& s_d) {
                  auto&& [s, d] = s_d;
                  return s - d;
                });
  btas_subtensor_to_tensor(*src, tensor, offset);
  dst->set(i, tensor);
  (*counter)++;
}

/// Task function for assigning a tensor to an Eigen submatrix

/// \tparam TA_Tensor_ a TiledArray::Tensor type
/// \tparam BTAS_Tensor_ a btas::Tensor type
/// \param src The source tensor
/// \param src_array_lobound the lobound of the DistArrany that contains src,
/// used to compute the offset to be applied to the coordinates of `src.range()`
/// to determine the block in \p dst to be copied into \param dst The
/// destination tensor \param counter The task counter
template <
    typename TA_Tensor_, typename BTAS_Tensor_, typename IntegerRange,
    typename = std::enable_if_t<detail::is_integral_range_v<IntegerRange>>>
void counted_tensor_to_btas_subtensor(const TA_Tensor_& src,
                                      IntegerRange src_array_lobound,
                                      BTAS_Tensor_* dst,
                                      madness::AtomicInt* counter) {
  auto offset = ranges::views::zip(ranges::views::all(dst->range().lobound()),
                                   src_array_lobound) |
                ranges::views::transform([](const auto& d_s) {
                  auto&& [d, s] = d_s;
                  return d - s;
                });
  tensor_to_btas_subtensor(src, *dst, offset);
  (*counter)++;
}

template <bool sparse>
auto make_shape(World& world, const TiledArray::TiledRange& trange);

template <>
inline auto make_shape<true>(World& world,
                             const TiledArray::TiledRange& trange) {
  TiledArray::Tensor<float> tile_norms(trange.tiles_range(),
                                       std::numeric_limits<float>::max());
  return TiledArray::SparseShape<float>(world, tile_norms, trange);
}

template <>
inline auto make_shape<false>(World&, const TiledArray::TiledRange&) {
  return TiledArray::DenseShape{};
}

}  // namespace detail

/// Convert a btas::Tensor object into a TiledArray::DistArray object

/// This function will copy the contents of \c src into a \c DistArray_ object
/// that is tiled according to the \c trange object. If the \c DistArray_ object
/// has sparse policy, a sparse map with large norm is created to ensure all the
/// values from \c src copy to the \c DistArray_ object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c src have been copied into the result array tiles.
/// Each tile is created
/// using the local contents of \c src, hence
/// it is your responsibility to ensure that the data in \c src
/// is distributed correctly among the ranks. If in doubt, you should replicate
/// \c src among the ranks prior to calling this.
///
/// Upon completion,
/// if the \c DistArray_ object has sparse policy truncate() is called.\n
/// Usage:
/// \code
/// btas::Tensor<double> src(100, 100, 100);
/// // Fill src with data ...
///
/// // Create a range for the new array object
/// std::vector<std::size_t> blocks;
/// for(std::size_t i = 0ul; i <= 100ul; i += 10ul)
///   blocks.push_back(i);
/// std::array<TiledArray::TiledRange1, 3> blocks3 =
///     {{ TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()),
///        TiledArray::TiledRange1(blocks.begin(), blocks.end()) }};
/// TiledArray::TiledRange trange(blocks3.begin(), blocks3.end());
///
/// // Create an Array from the source btas::Tensor object
/// TiledArray::TArrayD array =
///     btas_tensor_to_array<decltype(array)>(world, trange, src);
/// \endcode
/// \tparam DistArray_ a TiledArray::DistArray type
/// \tparam TArgs the type pack in type btas::Tensor<TArgs...> of \c src
/// \param[in,out] world The world where the result array will live
/// \param[in] trange The tiled range of the new array
/// \param[in] src The btas::Tensor<TArgs..> object whose contents will be
/// copied to the result.
/// \param replicated if true, the result will be replicated
///        [default = false].
/// \param pmap the process map object [default=null]; initialized to the
/// default if \p replicated is false, or a replicated pmap if \p replicated
/// is true; ignored if \p replicated is true and \c world.size()>1
/// \return A \c DistArray_ object that is a copy of \c src
/// \throw TiledArray::Exception When world size is greater than 1
/// \note If using 2 or more World ranks, set \c replicated=true and make sure
/// \c matrix is the same on each rank!
template <typename DistArray_, typename T, typename Range, typename Storage>
DistArray_ btas_tensor_to_array(
    World& world, const TiledArray::TiledRange& trange,
    const btas::Tensor<T, Range, Storage>& src, bool replicated = false,
    std::shared_ptr<typename DistArray_::pmap_interface> pmap = {}) {
  // Test preconditions
  const auto rank = trange.tiles_range().rank();
  TA_ASSERT(rank == src.range().rank() &&
            "TiledArray::btas_tensor_to_array(): rank of destination "
            "trange does not match the rank of source BTAS tensor.");
  auto dst_range_extents = trange.elements_range().extent();
  for (std::remove_const_t<decltype(rank)> d = 0; d != rank; ++d) {
    TA_ASSERT(dst_range_extents[d] == src.range().extent(d) &&
              "TiledArray::btas_tensor_to_array(): source dimension does "
              "not match destination dimension.");
  }

  using Tensor_ = btas::Tensor<T, Range, Storage>;
  using Policy_ = typename DistArray_::policy_type;
  const auto is_sparse = !is_dense_v<Policy_>;

  // Make a shape, only used if making a sparse array
  using Shape_ = typename DistArray_::shape_type;
  Shape_ shape = detail::make_shape<is_sparse>(world, trange);

  // Create a new tensor
  if (replicated && (world.size() > 1))
    pmap = std::static_pointer_cast<typename DistArray_::pmap_interface>(
        std::make_shared<detail::ReplicatedPmap>(
            world, trange.tiles_range().volume()));
  DistArray_ array = (pmap ? DistArray_(world, trange, shape, pmap)
                           : DistArray_(world, trange, shape));

  // Spawn copy tasks
  madness::AtomicInt counter;
  counter = 0;
  std::int64_t n = 0;
  for (auto&& acc : array) {
    world.taskq.add(
        &detail::counted_btas_subtensor_to_tensor<DistArray_, Tensor_>, &src,
        &array, acc.index(), &counter);
    ++n;
  }

  // Wait until the write tasks are complete
  array.world().await([&counter, n]() { return counter == n; });

  // Analyze tiles norms and truncate based on sparse policy
  if (is_sparse) truncate(array);

  return array;
}

namespace detail {

/// \sa TiledArray::array_to_btas_tensor()
template <typename Tile, typename Policy, typename Range_ = TiledArray::Range,
          typename Storage_ = btas::DEFAULT::storage<typename Tile::value_type>>
btas::Tensor<typename Tile::value_type, Range_, Storage_>
array_to_btas_tensor_impl(const TiledArray::DistArray<Tile, Policy>& src,
                          const Range_& result_range, int target_rank) {
  // Test preconditions
  if (target_rank == -1 && src.world().size() > 1 &&
      !src.pmap()->is_replicated())
    TA_ASSERT(
        src.world().size() == 1 &&
        "TiledArray::array_to_btas_tensor(): a non-replicated array can only "
        "be converted to a btas::Tensor on every rank if the number of World "
        "ranks is 1.");

  using result_type =
      btas::Tensor<typename TiledArray::DistArray<Tile, Policy>::element_type,
                   Range_, Storage_>;

  // Construct the result
  if (target_rank == -1 || src.world().rank() == target_rank) {
    // if array is sparse must initialize to zero
    result_type result(result_range, 0.0);

    // Spawn tasks to copy array tiles to btas::Tensor
    madness::AtomicInt counter;
    counter = 0;
    int n = 0;
    for (std::size_t i = 0; i < src.size(); ++i) {
      if (!src.is_zero(i)) {
        src.world().taskq.add(
            &detail::counted_tensor_to_btas_subtensor<
                Tile, result_type,
                std::decay_t<
                    decltype(src.trange().elements_range().lobound())>>,
            src.find(i), src.trange().elements_range().lobound(), &result,
            &counter);
        ++n;
      }
    }

    // Wait until the write tasks are complete
    src.world().await([&counter, n]() { return counter == n; });

    return result;
  } else  // else
    return result_type{};
}

}  // namespace detail

/// Convert a TiledArray::DistArray object into a btas::Tensor object

/// This function will copy the contents of \c src into a \c btas::Tensor
/// object. The copy operation is done in parallel, and this function will block
/// until all elements of \c src have been copied into the result array tiles.
/// The size of \c src.world().size() must be equal to 1 or \c src must be a
/// replicated TiledArray::DistArray. Usage:
/// \code
/// TiledArray::TArrayD
/// array(world, trange);
/// // Set tiles of array ...
///
/// auto t = array_to_btas_tensor(array);
/// \endcode
/// \tparam Tile the tile type of \c src
/// \tparam Policy the policy type of \c src
/// \tparam Range_ the range type of the result (either, btas::RangeNd or
///         TiledArray::Range)
/// \tparam Storage_ the storage type of the result
/// \param[in] src The TiledArray::DistArray<Tile,Policy> object whose contents
/// will be copied to the result.
/// \param[in] target_rank the rank on which to create the BTAS tensor
///            containing the data of \c src ; if \c target_rank=-1 then
///            create the BTAS tensor on every rank (this requires
///            that \c src.is_replicated()==true )
/// \return BTAS tensor object containing the data of \c src , if my rank equals
///         \c target_rank or \c target_rank==-1 ,
///         default-initialized BTAS tensor otherwise.
/// \warning The range of \c src is
///         not preserved, i.e. the lobound of the result is zero. Use the
///         variant of this function tagged with preserve_lobound_t to
///          preserve the range.
/// \throw TiledArray::Exception When world size is greater than
///        1 and \c src is not replicated
template <typename Tile, typename Policy, typename Range_ = TiledArray::Range,
          typename Storage_ = btas::DEFAULT::storage<typename Tile::value_type>>
btas::Tensor<typename Tile::value_type, Range_, Storage_> array_to_btas_tensor(
    const TiledArray::DistArray<Tile, Policy>& src, int target_rank = -1) {
  return detail::array_to_btas_tensor_impl(
      src, Range_(src.trange().elements_range().extent()), target_rank);
}

template <typename Tile, typename Policy, typename Range_ = TiledArray::Range,
          typename Storage_ = btas::DEFAULT::storage<typename Tile::value_type>>
btas::Tensor<typename Tile::value_type, Range_, Storage_> array_to_btas_tensor(
    const TiledArray::DistArray<Tile, Policy>& src, preserve_lobound_t,
    int target_rank = -1) {
  return detail::array_to_btas_tensor_impl(src, src.trange().elements_range(),
                                           target_rank);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_BTAS_H__INCLUDED
