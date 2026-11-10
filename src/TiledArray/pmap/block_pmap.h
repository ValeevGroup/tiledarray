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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  block_pmap.h
 *  October 30, 2026
 *
 */

#ifndef TILEDARRAY_PMAP_BLOCK_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_BLOCK_PMAP_H__INCLUDED

#include <TiledArray/permutation.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/range.h>

#include <memory>

namespace TiledArray {
namespace detail {

/// Process map of a block of an array's tiles that follows the array's map

/// Each tile of the block is owned by the rank that owns its source tile in
/// the parent array, so a block expression evaluated on this map fetches no
/// tile from another rank. The block's tiles may be permuted relative to the
/// parent's. The parent's map must be distributed (not replicated) and belong
/// to the World this map is constructed in: the ranks it gives are ranks of
/// that World only, which the constructor cannot check.
///
/// This map is meant for evaluators only: an array assigned from an
/// expression evaluated on it gets the default map for its tiled range
/// instead (see Expr::eval_to()), so that no array is left on a map that
/// depends on another array's.
class BlockPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::local_;       ///< The local tiles
  using Pmap::local_size_;  ///< The number of local tiles
  using Pmap::procs_;       ///< The number of processes
  using Pmap::rank_;        ///< The rank of this process
  using Pmap::size_;        ///< The number of tiles mapped among all processes

 public:
  typedef Pmap::size_type size_type;  ///< Size type

  /// \param world The world where the block's tiles are mapped, the parent's
  /// \param parent_pmap The parent array's process map
  /// \param parent_tiles The parent array's tiles range
  /// \param block_tiles The block's tiles range
  /// \param block_lobound The parent tile index of the block's first tile
  /// \param block_to_parent The permutation that maps a block tile index to
  ///        the parent's mode order (the inverse of the permutation applied
  ///        to the block, if any; the identity if none)
  BlockPmap(World& world, std::shared_ptr<const Pmap> parent_pmap,
            Range parent_tiles, Range block_tiles,
            Range::index_type block_lobound, Permutation block_to_parent)
      : Pmap(world, block_tiles.volume()),
        parent_pmap_(std::move(parent_pmap)),
        parent_tiles_(std::move(parent_tiles)),
        block_tiles_(std::move(block_tiles)),
        block_lobound_(std::move(block_lobound)),
        block_to_parent_(std::move(block_to_parent)) {
    TA_ASSERT(parent_pmap_);
    TA_ASSERT(!parent_pmap_->is_replicated());
    TA_ASSERT(parent_pmap_->procs() == procs_);
    TA_ASSERT(block_lobound_.size() == parent_tiles_.rank());
    for (size_type i = 0; i < size_; ++i)
      if (owner(i) == rank_) local_.push_back(i);
    local_size_ = local_.size();
  }

  virtual ~BlockPmap() {}

  /// Maps \c tile to the processor that owns it

  /// \param tile The tile to be queried
  /// \return The owner of \c tile 's source tile in the parent array
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    auto index = block_tiles_.idx(tile);
    if (block_to_parent_) index = block_to_parent_ * index;
    for (std::size_t d = 0; d < index.size(); ++d)
      index[d] += block_lobound_[d];
    return parent_pmap_->owner(parent_tiles_.ordinal(index));
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false .
  virtual bool is_local(const size_type tile) const {
    return BlockPmap::owner(tile) == rank_;
  }

  virtual const_iterator begin() const {
    return Iterator(*this, local_.begin());
  }
  virtual const_iterator end() const { return Iterator(*this, local_.end()); }

 private:
  std::shared_ptr<const Pmap> parent_pmap_;  ///< The parent array's map
  Range parent_tiles_;                       ///< The parent array's tiles
  Range block_tiles_;                        ///< The block's tiles
  Range::index_type block_lobound_;  ///< Parent index of block's first tile
  Permutation block_to_parent_;      ///< Block -> parent mode order
};                                   // class BlockPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_BLOCK_PMAP_H__INCLUDED
