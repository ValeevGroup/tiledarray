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

#include <TiledArray/error.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/madness.h>
#include <algorithm>

namespace TiledArray {
  namespace detail {

    /// A blocked process map

    /// Map N elements among P processes into blocks that are approximately N/P
    /// elements in size. A minimum block size may also be specified.
    class BlockedPmap : public Pmap {
    private:

      // Import Pmap protected variables
      using Pmap::local_;
      using Pmap::rank_;
      using Pmap::procs_;
      using Pmap::size_;

    public:
      typedef Pmap::size_type size_type; ///< Key type

      /// Construct Blocked map

      /// \param world A reference to the world
      /// \param size The number of elements to be mapped
      BlockedPmap(madness::World& world, std::size_t size) :
          Pmap(world, size),
          n_(size_ / procs_),
          r_(size_ % procs_),
          n1_(n_ + 1),
          rn1_(r_ * n1_)
      {
        // Find the first and last tiles of the local block
        size_type first = rank_ * n_ + std::min<std::size_t>(rank_, r_);
        const ProcessID rank1 = rank_ + 1;
        const size_type last = rank1 * n_ + std::min<std::size_t>(rank1, r_); // Compute the local block size

        local_.reserve(last - first);


        // Construct a map of all local processes
        for(; first < last; ++first) {
          TA_ASSERT(BlockedPmap::owner(first) == rank_);
          local_.push_back(first);
        }
      }

      virtual ~BlockedPmap() { }

      /// Maps \c tile to the processor that owns it

      /// \param tile The tile to be queried
      /// \return Processor that logically owns \c tile
      virtual ProcessID owner(const size_type tile) const {
        TA_ASSERT(tile < size_);
        return (tile < rn1_ ?
            tile / n1_ : // = tile / (n + 1)
            ((tile - rn1_) / n_) + r_); // = (tile - r_ * (n_ + 1)) / n_ + r_
      }

    private:

      size_type n_; ///< block size
      size_type r_; ///< tile remainder
      size_type n1_; ///< Cashed value: (n + 1)
      size_type rn1_; ///< Cached value: r * (n + 1)
    }; // class MapByRow

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_BLOCKED_PMAP_H__INCLUDED
