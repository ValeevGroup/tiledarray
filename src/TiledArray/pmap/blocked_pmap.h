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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  add.h
 *  April 24, 2012
 *
 */

#ifndef TILEDARRAY_PMAP_BLOCKED_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_BLOCKED_PMAP_H__INCLUDED

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {
namespace detail {

/// A blocked process map

/// Map N elements among P processes into blocks that are approximately N/P
/// elements in size. A minimum block size may also be specified.
class BlockedPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::procs_;  ///< The number of processes
  using Pmap::rank_;   ///< The rank of this process
  using Pmap::size_;   ///< The number of tiles mapped among all processes

 private:
  const size_type block_size_;         ///< block size (= size_ / procs_)
  const size_type remainder_;          ///< tile remainder (= size_ % procs_)
  const size_type block_size_plus_1_;  ///< Cashed value
  const size_type block_size_plus_1_times_remainder_;  ///< Cached value
  const size_type local_first_;  ///< First tile of this process's block
  const size_type local_last_;   ///< Last tile + 1 of this process's block

 public:
  typedef Pmap::size_type size_type;  ///< Key type

  /// Construct Blocked map

  /// \param world The world where the tiles will be mapped
  /// \param size The number of tiles to be mapped
  BlockedPmap(World& world, size_type size)
      : Pmap(world, size),
        block_size_(size_ / procs_),
        remainder_(size_ % procs_),
        block_size_plus_1_(block_size_ + 1),
        block_size_plus_1_times_remainder_(remainder_ * block_size_plus_1_),
        local_first_(rank_ * block_size_ +
                     std::min<size_type>(rank_, remainder_)),
        local_last_((rank_ + 1) * block_size_ +
                    std::min<size_type>((rank_ + 1), remainder_)) {
    this->local_size_ = local_last_ - local_first_;
  }

  virtual ~BlockedPmap() {}

  /// Maps \c tile to the processor that owns it

  /// \param tile The tile to be queried
  /// \return Processor that logically owns \c tile
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return (tile < block_size_plus_1_times_remainder_
                ? tile / block_size_plus_1_
                : ((tile - block_size_plus_1_times_remainder_) / block_size_) +
                      remainder_);
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false .
  virtual bool is_local(const size_type tile) const {
    return ((tile >= local_first_) && (tile < local_last_));
  }

  virtual const_iterator begin() const {
    return Iterator(*this, local_first_, local_last_, local_first_, false);
  }
  virtual const_iterator end() const {
    return Iterator(*this, local_first_, local_last_, local_last_, false);
  }

};  // class BlockedPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_BLOCKED_PMAP_H__INCLUDED
