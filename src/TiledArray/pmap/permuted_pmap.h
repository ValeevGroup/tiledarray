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
 *  permuted_pmap.h
 *  November 16, 2026
 *
 */

#ifndef TILEDARRAY_PMAP_PERMUTED_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_PERMUTED_PMAP_H__INCLUDED

#include <TiledArray/perm_index.h>
#include <TiledArray/permutation.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/range.h>

#include <memory>

namespace TiledArray {
namespace detail {

/// The image of a process map under a permutation of the tile index modes

/// Maps the tiles of a permuted ("target") tiles range so that each is owned
/// by the rank that a base map gives its preimage in the unpermuted
/// ("source") range: `owner(t) == base.owner(source(t))`.
///
/// This is the map of the result of an evaluation that computes its tiles in
/// one mode order, on the base map, and delivers them in another: with it
/// every tile is delivered to the rank that computed it. The base map applied
/// to the target ordinals directly would instead send most tiles to another
/// rank, since the base maps of contractions (see ProcGrid and SlabbedPmap)
/// are functions of the source mode order.
class PermutedPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::local_size_;  ///< The number of local tiles
  using Pmap::procs_;       ///< The number of processes
  using Pmap::rank_;        ///< The rank of this process
  using Pmap::size_;        ///< The number of tiles mapped among all processes

 public:
  typedef Pmap::size_type size_type;  ///< Size type

  /// \param world The world where the tiles are mapped
  /// \param base The map of the tiles in the source mode order
  /// \param target_tiles The tiles range in the target mode order
  /// \param source_to_target The permutation that takes a source tile index
  ///        to its target tile index
  PermutedPmap(World& world, std::shared_ptr<const Pmap> base,
               const Range& target_tiles, const Permutation& source_to_target)
      : Pmap(world, target_tiles.volume()),
        base_(std::move(base)),
        target_to_source_(target_tiles, source_to_target.inv()) {
    TA_ASSERT(base_);
    TA_ASSERT(source_to_target);
    TA_ASSERT(base_->size() == size_);
    TA_ASSERT(base_->procs() == procs_);
    if (base_->known_local_size()) this->local_size_ = base_->local_size();
  }

  virtual ~PermutedPmap() {}

  /// \return the map of the tiles in the source mode order
  const std::shared_ptr<const Pmap>& base() const { return base_; }

  /// Maps \c tile to the processor that owns it

  /// \param tile The (target) tile to be queried
  /// \return The owner of the source tile of \c tile in the base map
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return base_->owner(target_to_source_(tile));
  }

  /// Check that the tile is owned by this process

  /// \param tile The (target) tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false .
  virtual bool is_local(const size_type tile) const {
    return base_->is_local(target_to_source_(tile));
  }

  virtual bool known_local_size() const { return base_->known_local_size(); }

  virtual bool is_replicated() const { return base_->is_replicated(); }

  virtual const_iterator begin() const {
    return Iterator(*this, 0ul, size_, 0ul, /*checking=*/true);
  }
  virtual const_iterator end() const {
    return Iterator(*this, 0ul, size_, size_, /*checking=*/true);
  }

 private:
  const std::shared_ptr<const Pmap> base_;  ///< The map of the source tiles
  const PermIndex target_to_source_;  ///< Maps a target ordinal to its source

};  // class PermutedPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_PERMUTED_PMAP_H__INCLUDED
