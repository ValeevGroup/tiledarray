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
 *  slabbed_pmap.h
 *  Jun 11, 2026
 *
 */

#ifndef TILEDARRAY_PMAP_SLABBED_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_SLABBED_PMAP_H__INCLUDED

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {
namespace detail {

/// Replicates a base process map over a leading "slab" dimension

/// Consider a sequence of indices \f$ \{ o | o \in [0, N_{\rm slab} S) \} \f$
/// organized into \f$ N_{\rm slab} \f$ contiguous slabs of \f$ S \f$ indices
/// each. SlabbedPmap maps index \f$ o \f$ to the process that a base map
/// (defined over a single slab, i.e. over \f$ [0, S) \f$) assigns to
/// \f$ o \% S \f$ -- i.e. the owner of an index is independent of its slab.
///
/// This is the distribution of a *batched* contraction (see
/// TiledArray::detail::Summa): the operands and result carry the fused
/// (Hadamard/batch) modes as their leading dimensions, every slab is
/// distributed identically over the same 2-d process grid, and the slab
/// index never participates in inter-process communication patterns.
class SlabbedPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::local_size_;  ///< The number of local tiles
  using Pmap::procs_;       ///< The number of processes
  using Pmap::rank_;        ///< The rank of this process
  using Pmap::size_;        ///< The number of tiles mapped among all processes

 private:
  const std::shared_ptr<const Pmap> base_;  ///< The per-slab base map
  const size_type slab_size_;               ///< The number of indices per slab
  const size_type nslabs_;                  ///< The number of slabs

 public:
  typedef Pmap::size_type size_type;  ///< Size type

  /// Construct a slab-replicated process map

  /// \param world The world where the tiles will be mapped
  /// \param base The base process map, defined over one slab
  /// \param nslabs The number of slabs
  SlabbedPmap(World& world, std::shared_ptr<const Pmap> base,
              const size_type nslabs)
      : Pmap(world, base->size() * nslabs),
        base_(std::move(base)),
        slab_size_(base_->size()),
        nslabs_(nslabs) {
    TA_ASSERT(base_);
    TA_ASSERT(nslabs_ > 0ul);
    this->local_size_ = base_->local_size() * nslabs_;
  }

  virtual ~SlabbedPmap() {}

  /// \return the per-slab base map
  const std::shared_ptr<const Pmap>& base() const { return base_; }
  /// \return the number of indices per slab
  size_type slab_size() const { return slab_size_; }
  /// \return the number of slabs
  size_type nslabs() const { return nslabs_; }

  /// Maps \c tile to the process that owns it

  /// \param tile The tile to be queried
  /// \return Process that logically owns \c tile
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return base_->owner(tile % slab_size_);
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false
  virtual bool is_local(const size_type tile) const {
    return base_->is_local(tile % slab_size_);
  }

  virtual bool known_local_size() const { return base_->known_local_size(); }

  virtual const_iterator begin() const {
    return Iterator(*this, 0ul, size_, 0ul, /*checking=*/true);
  }
  virtual const_iterator end() const {
    return Iterator(*this, 0ul, size_, size_, /*checking=*/true);
  }

};  // class SlabbedPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_SLABBED_PMAP_H__INCLUDED
