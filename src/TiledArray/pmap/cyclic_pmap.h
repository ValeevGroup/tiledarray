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
 *  cyclic_pmap.h
 *  May 1, 2012
 *
 */

#ifndef TILEDARRAY_PMAP_CYCLIC_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_CYCLIC_PMAP_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/madness.h>
#include <cmath>

namespace TiledArray {
  namespace detail {

    /// Map processes using cyclic decomposition

    /// This map does a cyclic decomposition of an m-by-n matrix. It distributes
    /// processors into an x-by-y matrix such that the ratios of m/n and x/y are
    /// approximately equal.
    class CyclicPmap : public Pmap {
    private:

      // Import Pmap protected variables
      using Pmap::local_;
      using Pmap::rank_;
      using Pmap::procs_;
      using Pmap::size_;

    public:
      typedef Pmap::size_type size_type;

      CyclicPmap(madness::World& world, std::size_t m, std::size_t n) :
          Pmap(world, m * n),
          m_(m),
          n_(n),
          x_(),
          y_()
      {
        TA_ASSERT(m > 0ul);
        TA_ASSERT(n > 0ul);

        // Get a rough estimate of the process dimensions
        // The ratios of x_ / y_  and m_ / n_ should be approximately equal.
        // Constraints: 1 <= x_ <= procs_ && 1 <= y_
        // The process map should be no bigger than m * n
        y_ = procs_ / std::max<std::size_t>(std::min<std::size_t>(
            std::sqrt(procs_ * m_ / n_), procs_), 1ul);
        x_ = procs_ / y_;

        // Maximum size is m and n
        x_ = std::min(x_, m_);
        y_ = std::min(y_, n_);

        TA_ASSERT(x_ * y_ <= procs_);

        // Initialize local
        local_.reserve((m_ / x_) * (n_ / y_));
        const std::size_t end = m_ * n_;
        const std::size_t m_step = n_ * x_;
        const std::size_t row_end_offset = n_ - (rank_ % y_);
        for(std::size_t m = (rank_ / y_) * n_ + (rank_ % y_); m < end; m += m_step) {
          const std::size_t row_end = m + row_end_offset;
          for(std::size_t i = m; i < row_end; i += y_) {
            TA_ASSERT(CyclicPmap::owner(i) == rank_);
            local_.push_back(i);
          }
        }
      }

      CyclicPmap(madness::World& world, std::size_t m, std::size_t n, std::size_t x, std::size_t y) :
          Pmap(world, m * n),
          m_(m),
          n_(n),
          x_(std::min<std::size_t>(x,m)),
          y_(std::min<std::size_t>(y,n))
      {
        TA_ASSERT(x_ * y_ <= procs_);

        // Initialize local
        local_.reserve((m_ / x_) * (n_ / y_));
        const std::size_t end = m_ * n_;
        const std::size_t m_step = n_ * x_;
        const std::size_t row_end_offset = n_ - (rank_ % y_);
        for(std::size_t m = (rank_ / y_) * n_ + (rank_ % y_); m < end; m += m_step) {
          const std::size_t row_end = m + row_end_offset;
          for(std::size_t i = m; i < row_end; i += y_) {
            TA_ASSERT(CyclicPmap::owner(i) == rank_);
            local_.push_back(i);
          }
        }
      }

      virtual ~CyclicPmap() { }

      /// Maps \c tile to the processor that owns it

      /// \param tile The tile to be queried
      /// \return Processor that logically owns \c tile
      virtual ProcessID owner(const size_type tile) const {
        TA_ASSERT(tile < (m_ * n_));
//        // Get matrix coordinate
//        const std::size_t m = key / n_;
//        const std::size_t n = key % n_;
//        // Get processor coordinate
//        const std::size_t x = m % x_;
//        const std::size_t y = n % y_;
//        // Get processor ordinal
//        const std::size_t o = x * y_ + y;
        const std::size_t o = ((tile / n_) % x_) * y_ + ((tile % n_) % y_);

        TA_ASSERT(o < procs_);

        return o;
      }

    private:

      std::size_t m_; ///< Number of rows to be mapped
      std::size_t n_; ///< Number of columns to be mapped
      std::size_t x_; ///< Number of process rows
      std::size_t y_; ///< Number of process columns
//      madness::hashT seed_; ///< Hashing seed for process randomization
    }; // class CyclicPmap

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_CYCLIC_PMAP_H__INCLUDED
