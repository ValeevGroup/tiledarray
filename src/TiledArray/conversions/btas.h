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

  /// A block of btas::Tensor \c src will be copied into TiledArray::Tensor \c dest. The block
  /// dimensions will be determined by the dimensions of the tensor's range.
  /// \tparam T The tensor element type
  /// \tparam Range The range type of the source btas::Tensor object
  /// \tparam Storage The storage type of the source btas::Tensor object
  /// \tparam Allocator The allocator type for the destination TA::Tensor object
  /// \param[in] src The source object; its subblock defined by the {lower,upper} bounds \c {dst.lobound(),dst.upbound()} will be copied to \c dst
  /// \param[out] dst The object that will contain the contents of the corresponding subblock of src
  /// \throw TiledArray::Exception When the dimensions of \c src and \c dst do not match.
  template <typename T, typename Range_, typename Storage, typename Allocator>
  inline void btas_subtensor_to_tensor(const btas::Tensor<T,Range_,Storage>& src, Tensor<T, Allocator>& dst) {
    typedef typename Tensor<T, Allocator>::size_type size_type;
    TA_ASSERT(dst.range().rank() == src.range().rank());

    const auto& src_range = src.range();
    const auto& dst_range = dst.range();
    auto src_blk_range = TA::BlockRange(detail::make_ta_range(src_range), dst_range.lobound(), dst_range.upbound());
    using std::data;
    auto src_view = TiledArray::make_const_map(data(src), src_blk_range);

    // Copy tensor subblock
    dst = src_view;
  }

}  // namespace TiledArray

#endif // TILEDARRAY_CONVERSIONS_BTAS_H__INCLUDED
