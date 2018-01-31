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

#include <TiledArray/external/btas.h>

namespace TiledArray {

  /// Copy a block of a btas::Tensor into a TiledArray::Tensor

  /// A block of btas::Tensor \c src will be copied into TiledArray::Tensor \c dst. The block
  /// dimensions will be determined by the dimensions of the range of \c dst .
  /// \tparam T The tensor element type
  /// \tparam Range_ The range type of the source btas::Tensor object
  /// \tparam Storage_ The storage type of the source btas::Tensor object
  /// \tparam Allocator_ The allocator type of the destination TiledArray::Tensor object
  /// \param[in] src The source object; its subblock defined by the {lower,upper} bounds \c {dst.lobound(),dst.upbound()} will be copied to \c dst
  /// \param[out] dst The object that will contain the contents of the corresponding subblock of src
  /// \throw TiledArray::Exception When the dimensions of \c src and \c dst do not match.
  template <typename T, typename Range_, typename Storage_, typename Allocator_>
  inline void btas_subtensor_to_tensor(const btas::Tensor<T,Range_,Storage_>& src, Tensor<T, Allocator_>& dst) {
    TA_ASSERT(dst.range().rank() == src.range().rank());

    const auto& src_range = src.range();
    const auto& dst_range = dst.range();
    auto src_blk_range = TiledArray::BlockRange(detail::make_ta_range(src_range), dst_range.lobound(), dst_range.upbound());
    using std::data;
    auto src_view = TiledArray::make_const_map(data(src), src_blk_range);

    dst = src_view;
  }

  /// Copy a block of a btas::Tensor into a TiledArray::Tensor

  /// TiledArray::Tensor \c src will be copied into a block of btas::Tensor \c dst. The block
  /// dimensions will be determined by the dimensions of the range of \c src .
  /// \tparam T The tensor element type
  /// \tparam Allocator_ The allocator type of the source TiledArray::Tensor object
  /// \tparam Range_ The range type of the destination btas::Tensor object
  /// \tparam Storage_ The storage type of the destination btas::Tensor object
  /// \param[in] src The source object whose contents will be copied into a subblock of \c dst
  /// \param[out] dst The destination object; its subblock defined by the {lower,upper} bounds \c {src.lobound(),src.upbound()} will be overwritten with the content of \c src
  /// \throw TiledArray::Exception When the dimensions of \c src and \c dst do not match.
  template <typename T, typename Allocator_, typename Range_, typename Storage_>
  inline void tensor_to_btas_subtensor(const Tensor<T, Allocator_>& src, btas::Tensor<T,Range_,Storage_>& dst) {
    TA_ASSERT(dst.range().rank() == src.range().rank());

    const auto& src_range = src.range();
    const auto& dst_range = dst.range();
    auto dst_blk_range = TiledArray::BlockRange(detail::make_ta_range(dst_range), src_range.lobound(), src_range.upbound());
    using std::data;
    auto dst_view = TiledArray::make_map(data(dst), dst_blk_range);

    dst_view = src;
  }

namespace detail {

/// Task function for converting btas::Tensor subblock to a TiledArray::DistArray

/// \tparam DistArray_ a TiledArray::DistArray type
/// \tparam TArgs the type pack in btas::Tensor<TArgs...> type
/// \param src The btas::Tensor object whose block will be copied
/// \param dst The array that will hold the result
/// \param i The index of the tile to be copied
/// \param counter The task counter
template <typename DistArray_, typename BTAS_Tensor_>
void counted_btas_subtensor_to_tensor(const BTAS_Tensor_* src,
                                      DistArray_* dst, const typename DistArray_::size_type i, madness::AtomicInt* counter)
{
  typename DistArray_::value_type tensor(dst->trange().make_tile_range(i));
  btas_subtensor_to_tensor(*src, tensor);
  dst->set(i, tensor);
  (*counter)++;
}

/// Task function for assigning a tensor to an Eigen submatrix

/// \tparam Tensor_ a TiledArray::Tensor type
/// \tparam TArgs the type pack in btas::Tensor<TArgs...> type
/// \param src The source tensor
/// \param dst The destination tensor
/// \param counter The task counter
template <typename TA_Tensor_, typename BTAS_Tensor_>
void counted_tensor_to_btas_subtensor(const TA_Tensor_& src,
                                      BTAS_Tensor_* dst, madness::AtomicInt* counter)
{
  tensor_to_btas_subtensor(src, *dst);
  (*counter)++;
}

} // namespace detail

/// Convert a btas::Tensor object into a TiledArray::DistArray object

/// This function will copy the contents of \c src into a \c DistArray_ object
/// that is tiled according to the \c trange object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c src have been copied into the result array tiles. The size of
/// \c world.size() must be equal to 1 or \c replicate must be equal to
/// \c true . If \c replicate is \c true, it is your responsibility to ensure
/// that the data in \c src is identical on all nodes.
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
/// \param[in] src The btas::Tensor<TArgs..> object whose contents will be copied to the result.
/// \param[in] replicated \c true indicates that the result array should be a
///            replicated array [default = false].
/// \return A \c DistArray_ object that is a copy of \c src
/// \throw TiledArray::Exception When world size is greater than 1
/// \note If using 2 or more World ranks, set \c replicated=true and make sure \c matrix
/// is the same on each rank!
template <typename DistArray_, typename ... TArgs>
DistArray_ btas_tensor_to_array(World& world, const typename DistArray_::trange_type& trange,
                                const btas::Tensor<TArgs...>& src, bool replicated = false)
{
  // Test preconditions
  const auto rank = trange.tiles_range().rank();
  TA_USER_ASSERT(rank == src.range().rank(),
                 "TiledArray::btas_tensor_to_array(): rank of destination trange does not match the rank of source BTAS tensor.");
  auto dst_range_extents = trange.elements_range().extent();
  for(auto d=0; d!=rank; ++d) {
    TA_USER_ASSERT(dst_range_extents[d] == src.range().extent(d),
                   "TiledArray::btas_tensor_to_array(): source dimension does not match destination dimension.");
  }

  // Check that this is not a distributed computing environment
  if(! replicated)
    TA_USER_ASSERT(world.size() == 1,
                   "An array can be created from a btas::Tensor if the number of World ranks is greater than 1 only when replicated=true.");

  // Create a new tensor
  DistArray_ array = (replicated && (world.size() > 1)
                     ? DistArray_(world, trange,
                     std::static_pointer_cast<typename DistArray_::pmap_interface>(
                     std::make_shared<detail::ReplicatedPmap>(
                         world, trange.tiles_range().volume())))
             : DistArray_(world, trange));

  // Spawn copy tasks
  madness::AtomicInt counter;
  counter = 0;
  std::int64_t n = 0;
  for(std::size_t i = 0; i < array.size(); ++i) {
    world.taskq.add(& detail::counted_btas_subtensor_to_tensor<DistArray_, btas::Tensor<TArgs...>>,
                    &src, &array, i, &counter);
    ++n;
  }

  // Wait until the write tasks are complete
  array.world().await([&counter,n] () { return counter == n; });

  return array;
}

/// Convert a TiledArray::DistArray object into a btas::Tensor object

/// This function will copy the contents of \c src into a \c btas::Tensor object.
/// The copy operation is done in parallel, and this function will block until all elements of
/// \c src have been copied into the result array tiles. The size of
/// \c src.world().size() must be equal to 1 or \c src must be a replicated TiledArray::DistArray.
/// Usage:
/// \code
/// TiledArray::TArrayD array(world, trange);
/// // Set tiles of array ...
///
/// auto t = array_to_btas_tensor(array);
/// \endcode
/// \tparam Tile the tile type of \c src
/// \tparam Policy the policy type of \c src
/// \param[in] src The TiledArray::DistArray<Tile,Policy> object whose contents will be copied to the result.
/// \return A \c btas::Tensor object that is a copy of \c src
/// \throw TiledArray::Exception When world size is greater than 1 and \c src is not replicated
template <typename Tile, typename Policy>
btas::Tensor<typename Tile::value_type>
array_to_btas_tensor(const TiledArray::DistArray<Tile,Policy>& src)
{
  // Test preconditions
  if(! src.pmap()->is_replicated())
    TA_USER_ASSERT(src.world().size() == 1,
                   "TiledArray::array_to_btas_tensor(): a non-replicated array can only be converted to a btas::Tensor if the number of World ranks is 1.");

  // Construct the result
  using result_type = btas::Tensor<typename TiledArray::DistArray<Tile,Policy>::element_type>;
  using result_range_type = typename result_type::range_type;
  // if array is sparse must initialize to zero
  result_type result(result_range_type(src.trange().elements_range().extent()), 0.0);

  // Spawn tasks to copy array tiles to btas::Tensor
  madness::AtomicInt counter;
  counter = 0;
  std::size_t n = 0;
  for(std::size_t i = 0; i < src.size(); ++i) {
    if(! src.is_zero(i)) {
      src.world().taskq.add(
          & detail::counted_tensor_to_btas_subtensor<Tile, result_type>,
          src.find(i), &result, &counter);
      ++n;
    }
  }

  // Wait until the write tasks are complete
  src.world().await([&counter,n] () { return counter == n; });

  return result;
}


/// Convert a btas::Tensor object into a TiledArray::DistArray object

/// This function will copy the contents of \c src into a sparse policy \c DistArray_ object
/// that is tiled according to the \c trange object. The copy operation is
/// done in parallel, and this function will block until all elements of
/// \c src have been copied into the result array tiles. The size of
/// \c world.size() must be equal to 1 or \c replicate must be equal to
/// \c true . If \c replicate is \c true, it is your responsibility to ensure
/// that the data in \c src is identical on all nodes.  After the copy is complete
/// sparse policy truncation is carried out on the new \c DistArray_ object
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
///     btas_tensor_to_sparse_array<decltype(array)>(world, trange, src);
/// \endcode
/// \tparam DistArray_ a TiledArray::DistArray type with policy=sparse
/// \tparam TArgs the type pack in type btas::Tensor<TArgs...> of \c src
/// \param[in,out] world The world where the result array will live
/// \param[in] trange The tiled range of the new array
/// \param[in] src The btas::Tensor<TArgs..> object whose contents will be copied to the result.
/// \param[in] replicated \c true indicates that the result array should be a
///            replicated array [default = false].
/// \return A \c DistArray_ object that is a copy of \c src
/// \throw TiledArray::Exception When world size is greater than 1
/// \note If using 2 or more World ranks, set \c replicated=true and make sure \c matrix
/// is the same on each rank!
template <typename DistArray_, typename ... TArgs>
DistArray_ btas_tensor_to_sparse_array(World& world, const typename DistArray_::trange_type& trange,
                                const btas::Tensor<TArgs...>& src, bool replicated = false)
  {
    // Test preconditions
    const auto rank = trange.tiles_range().rank();
    TA_USER_ASSERT(rank == src.range().rank(),
                   "TiledArray::btas_tensor_to_array(): rank of destination trange does not match the rank of source BTAS tensor.");

    auto dst_range_extents = trange.elements_range().extent();
    for(auto d=0; d!=rank; ++d) {
      TA_USER_ASSERT(dst_range_extents[d] == src.range().extent(d),
                     "TiledArray::btas_tensor_to_array(): source dimension does not match destination dimension.");
    }

    // Check that this is not a distributed computing environment
    if(! replicated)
      TA_USER_ASSERT(world.size() == 1,
                     "An array can be created from a btas::Tensor if the number of World ranks is greater than 1 only when replicated=true.");

    // Making a norm, so all data is copied to TA
    auto norm = src.size();
    TensorF tile_norms(trange.tiles_range(), norm);
    TA::TSpArrayD::shape_type shape(world, tile_norms, trange);
    
    // Create a new tensor
    DistArray_ array = (replicated && (world.size() > 1))
                        ? DistArray_(world, trange, shape,
                                     std::static_pointer_cast<typename DistArray_::pmap_interface>(
                                             std::make_shared<detail::ReplicatedPmap>(
                                                     world, trange.tiles_range().volume())))
                        : DistArray_(world, trange, shape);

    // Spawn copy tasks
    madness::AtomicInt counter;
    counter = 0;
    std::int64_t n = 0;
    for(std::size_t i = 0; i < array.size(); ++i) {
      world.taskq.add(& detail::counted_btas_subtensor_to_tensor<DistArray_, btas::Tensor<TArgs...>>,
                      &src, &array, i, &counter);
      ++n;
    }

    // Wait until the write tasks are complete
    array.world().await([&counter,n] () { return counter == n; });

    // Analyze tiles norms and truncate based on sparse policy
    truncate(array);
    return array;
  }
}  // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_BTAS_H__INCLUDED
