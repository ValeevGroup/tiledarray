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

#include <memory>
#include <utility>

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
/// The 3-d-grid variant adds a process dimension \c proc_h along the slab
/// (h) axis: the world's first `proc_h * proc_h_stride` ranks are partitioned
/// into `proc_h` contiguous h-planes of `proc_h_stride` ranks each; slab
/// \f$ h \f$ belongs to plane \f$ h \% proc\_h \f$, and within its plane is
/// distributed by the base map, whose owners must then be PLANE-LOCAL ranks
/// in \f$ [0, proc\_h\_stride) \f$: the owner of index \f$ o \f$ is
/// \f$ (h \% proc\_h) \cdot proc\_h\_stride + base(o \% S) \f$.
/// `proc_h == 1` reduces to the slab-replicated map above (and is constructed
/// via the 2-argument-base constructor, which delegates locality to the base
/// map).
///
/// COARSEN-H (two-trange retile): when the array is physically tiled into
/// `nslabs` FINE (U) slabs but the SUMMA process grid rides COARSE slabs --
/// each coarse slab covering \c u_per_coarse_slab contiguous U slabs -- the
/// h-plane of a U slab is that of its COVERING coarse slab, so the group of
/// U slab \f$ h \f$ is \f$ \lfloor h / u\_per\_coarse\_slab \rfloor \% proc\_h
/// \f$. This co-locates every U tile of a coarse slab on the plane that runs
/// that coarse slab's SUMMA work, matching the coarse step grouping
/// (Summa::next_step / first_slab_) and the coarse result_tile_owner. With
/// `u_per_coarse_slab == 1` (identity-H) this is exactly the per-U-slab
/// grouping above and is byte-for-byte the prior behavior.
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
  const size_type proc_h_ = 1;         ///< Process-grid extent along the slab
                                       ///< (h) axis (the number of h-planes)
  const size_type proc_h_stride_ = 0;  ///< Ranks per h-plane (0 = the whole
                                       ///< world; base owners are world ranks
                                       ///< and locality delegates to the base)
  const size_type u_per_coarse_slab_ = 1;  ///< # of FINE (U) slabs each COARSE
                                           ///< SUMMA slab covers (COARSEN-H);
                                           ///< 1 == identity-H (group by U slab)

  /// \return the COARSE slab covering U slab \p u_slab (== u_slab for
  /// identity-H, where u_per_coarse_slab_ == 1)
  size_type coarse_slab_of(const size_type u_slab) const {
    return u_slab / u_per_coarse_slab_;
  }

  /// \return whether this map distributes the slab axis over a process plane
  bool hgrouped() const { return proc_h_stride_ != 0ul; }

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

  /// Construct an h-grouped (3-d grid) slabbed process map

  /// \param world The world where the tiles will be mapped
  /// \param base The base process map, defined over one slab; its owners
  ///        must be GROUP-LOCAL ranks in [0, proc_h_stride)
  /// \param nslabs The number of slabs
  /// \param proc_h The number of slab groups (slab h -> group h % proc_h)
  /// \param proc_h_stride The number of (contiguous) world ranks per group
  /// \param u_per_coarse_slab The number of FINE (U) slabs each COARSE SUMMA
  ///        slab covers (COARSEN-H); the group of U slab \c h is
  ///        `(h / u_per_coarse_slab) % proc_h`. Default 1 (identity-H) groups
  ///        by the U slab directly, byte-for-byte the prior behavior.
  SlabbedPmap(World& world, std::shared_ptr<const Pmap> base,
              const size_type nslabs, const size_type proc_h,
              const size_type proc_h_stride,
              const size_type u_per_coarse_slab = 1ul)
      : Pmap(world, base->size() * nslabs),
        base_(std::move(base)),
        slab_size_(base_->size()),
        nslabs_(nslabs),
        proc_h_(proc_h),
        proc_h_stride_(proc_h_stride),
        u_per_coarse_slab_(u_per_coarse_slab) {
    TA_ASSERT(base_);
    TA_ASSERT(nslabs_ > 0ul);
    TA_ASSERT(proc_h_ > 0ul);
    TA_ASSERT(proc_h_stride_ > 0ul);
    TA_ASSERT(proc_h_ * proc_h_stride_ <= size_type(world.size()));
    TA_ASSERT(u_per_coarse_slab_ > 0ul);
    TA_ASSERT(nslabs_ % u_per_coarse_slab_ == 0ul);

    // this rank's group and group-local rank; ranks beyond the grouped
    // prefix of the world own nothing
    const size_type rank = world.rank();
    if (rank < proc_h_ * proc_h_stride_) {
      const size_type my_group = rank / proc_h_stride_;
      const size_type my_group_rank = rank % proc_h_stride_;
      // count of my group's U slabs: U slab h in [0, nslabs) whose COVERING
      // coarse slab (h / u_per_coarse_slab) is congruent to my_group mod
      // proc_h. The n_coarse coarse slabs split round-robin over proc_h groups
      // and each coarse slab carries u_per_coarse_slab U slabs.
      const size_type n_coarse = nslabs_ / u_per_coarse_slab_;
      const size_type my_coarse_slabs =
          (n_coarse / proc_h_) + (my_group < (n_coarse % proc_h_) ? 1u : 0u);
      const size_type my_slabs = my_coarse_slabs * u_per_coarse_slab_;
      // count of slab indices the base assigns to my group-local rank
      size_type base_local = 0ul;
      for (size_type j = 0ul; j < slab_size_; ++j)
        if (base_->owner(j) == my_group_rank) ++base_local;
      this->local_size_ = my_slabs * base_local;
    } else {
      this->local_size_ = 0ul;
    }
  }

  virtual ~SlabbedPmap() {}

  /// \return the per-slab base map
  const std::shared_ptr<const Pmap>& base() const { return base_; }
  /// \return the number of indices per slab
  size_type slab_size() const { return slab_size_; }
  /// \return the number of slabs
  size_type nslabs() const { return nslabs_; }
  /// \return the number of slab (h) groups
  size_type proc_h() const { return proc_h_; }

  /// Maps \c tile to the process that owns it

  /// \param tile The tile to be queried
  /// \return Process that logically owns \c tile
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    if (!hgrouped()) return base_->owner(tile % slab_size_);
    // The U slab of this tile is (tile / slab_size_); its h-plane is that of
    // the COARSE slab covering it (coarse_slab_of), so co-located U tiles of
    // one coarse slab land on the rank that runs that coarse slab's SUMMA.
    const size_type group = coarse_slab_of(tile / slab_size_) % proc_h_;
    return group * proc_h_stride_ + base_->owner(tile % slab_size_);
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false
  virtual bool is_local(const size_type tile) const {
    if (!hgrouped()) return base_->is_local(tile % slab_size_);
    return owner(tile) == size_type(rank_);
  }

  virtual bool known_local_size() const {
    return hgrouped() ? true : base_->known_local_size();
  }

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
