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

    /// Map processes using a 2D cyclic decomposition

    /// This map cyclicly distributes a two-dimensional grid of tiles among a
    /// two-dimensional grid of processes.
    class CyclicPmap : public Pmap {
    protected:

      // Import Pmap protected variables
      using Pmap::rank_; ///< The rank of this process
      using Pmap::procs_; ///< The number of processes
      using Pmap::size_; ///< The number of tiles mapped among all processes
      using Pmap::local_; ///< A list of local tiles

    private:

      const size_type rows_; ///< Number of tile rows to be mapped
      const size_type cols_; ///< Number of tile columns to be mapped
      const size_type proc_cols_; ///< Number of process columns
      const size_type proc_rows_; ///< Number of process rows

      /// Initialize local tile list
      void init_local() {
        if(rank_ < (proc_rows_ * proc_cols_)) {
          local_.reserve((rows_ / proc_rows_) * (cols_ / proc_cols_));
          // Compute rank coordinates
          const size_type rank_row = rank_ / proc_cols_;
          const size_type rank_col = rank_ % proc_cols_;

          // Iterate over local tiles
          for(size_type i = rank_row; i < rows_; i += proc_rows_) {
            const size_type row_end = (i + 1) * cols_;
            for(size_type tile = i * cols_ + rank_col; tile < row_end; tile += proc_cols_) {
              TA_ASSERT(CyclicPmap::owner(tile) == rank_);
              local_.push_back(tile);
            }
          }
        }
      }

      /// Get value subject to \c min and \c max constraints

      /// \param min The minimum value
      /// \param value The test value
      /// \param max The maximum value
      /// \return \c value if it is in the range \c [min,max], otherwise the
      /// nearest extreme.
      static size_type min_max(const size_type min, const size_type value, const size_type max) {
        TA_ASSERT(min <= max);
        return std::max(min, std::min(value, max));
      }

    public:
      typedef Pmap::size_type size_type; ///< Size type

      /// Construct process map

      /// \param world The world where the tiles will be mapped
      /// \param rows The number of tile rows to be mapped
      /// \param cols The number of tile columns to be mapped
      CyclicPmap(madness::World& world, size_type rows, size_type cols) :
          Pmap(world, rows * cols), rows_(rows), cols_(cols),
          proc_cols_(min_max(1ul, std::min<size_type>(std::sqrt(double(procs_) *
              double(cols) / double(rows)), procs_), cols_)),
          proc_rows_(std::min<size_type>(procs_ / proc_cols_, rows_))
      {
        TA_ASSERT(rows_ >= 1ul);
        TA_ASSERT(cols_ >= 1ul);
        TA_ASSERT((proc_rows_ * proc_cols_) <= procs_);
        // This constructor makes a rough estimate of the optimal process
        // dimensions. The goal is for the ratios of proc_rows_/proc_cols_ and rows_/cols_
        // to be approximately equal.
        // Constraints: 1 <= proc_rows_ <= procs_ && 1 <= proc_cols_
        // The process map should be no bigger than m * n
        // Maximum size is m and n

        init_local();
      }

      /// Construct process map

      /// \param world The world where the tiles will be mapped
      /// \param rows The number of tile rows to be mapped
      /// \param cols The number of tile columns to be mapped
      /// \param proc_rows The number of process rows in the map
      /// \param proc_cols The number of process columns in the map
      /// \throw TiledArray::Exception When <tt>proc_rows > rows</tt>
      /// \throw TiledArray::Exception When <tt>proc_cols > cols</tt>
      /// \throw TiledArray::Exception When <tt>proc_rows * proc_cols > world.size()</tt>
      CyclicPmap(madness::World& world, size_type rows, size_type cols, size_type proc_rows, size_type proc_cols) :
          Pmap(world, rows * cols), rows_(rows), cols_(cols),
          proc_cols_(proc_cols), proc_rows_(proc_rows)
      {
        // Check that the processor grid is non-zero
        TA_ASSERT(proc_rows_ <= rows_);
        TA_ASSERT(proc_cols_ <= cols_);
        TA_ASSERT(proc_rows_ * proc_cols_ <= procs_);

        init_local();
      }

      virtual ~CyclicPmap() { }

      /// Maps \c tile to the processor that owns it

      /// \param tile The tile to be queried
      /// \return Processor that logically owns \c tile
      virtual size_type owner(const size_type tile) const {
        TA_ASSERT(tile < size_);
        // Compute tile coordinate in tile grid
        const size_type tile_row = tile / cols_;
        const size_type tile_col = tile % cols_;
        // Compute process coordinate of tile in the process grid
        const size_type proc_row = tile_row % proc_rows_;
        const size_type proc_col = tile_col % proc_cols_;
        // Compute the process that owns tile
        const size_type proc = proc_row * proc_cols_ + proc_col;

        TA_ASSERT(proc < procs_);

        return proc;
      }


      /// Check that the tile is owned by this process

      /// \param tile The tile to be checked
      /// \return \c true if \c tile is owned by this process, otherwise \c false .
      virtual bool is_local(const size_type tile) const {
        return (CyclicPmap::owner(tile) == rank_);
      }

      /// Row count accessor

      /// \return The number of rows
      size_type rows() const { return rows_; }

      /// Column count accessor

      /// \return The number of columns
      size_type cols() const { return cols_; }

      /// Process row count accessor

      /// \return The number of process rows
      size_type proc_rows() const { return proc_rows_; }

      /// Process column count accessor

      /// \return The number of process columns
      size_type proc_cols() const { return proc_cols_; }

    }; // class CyclicPmap

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_CYCLIC_PMAP_H__INCLUDED
