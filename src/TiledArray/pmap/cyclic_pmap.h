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

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {
namespace detail {

enum class CyclicPmapOrder {
  RowMajor,
  ColMajor
};

/// Maps cyclically a sequence of indices onto a 2-d matrix of processes

/// Consider a sequence of indices \f$ \{ k | k \in [0,N) \} \f$,
/// organized into a matrix, \f$ \{ \{i,j\} | i \in [0,N_{\rm
/// row}),j\in[0,N_{\rm col})\} \f$, in row-major form, i.e. index \f$ k \f$
/// corresponds to \f$ \{ k_{\rm row}, k_{\rm col} \} \equiv \{ k / N_{\rm col},
/// k \% N_{\rm col} \} \f$. Similarly, a sequence of processes is organized
/// into a matrix with \f$ P_{\rm row} \f$ rows and \f$ P_{\rm cols} \f$
/// columns, i.e process \f$ p \equiv \{ p_{\rm row}, p_{\rm col} \} \f$. Then
/// index \f$ \{ k_{\rm row}, k_{\rm col} \} \f$ maps to process \f$ \{ p_{\rm
/// row}, p_{\rm col} \} = \{ k_{\rm row} \% N_{\rm row}, k_{\rm col} \% N_{\rm
/// col} \} \f$
///
/// \note This class is used to map <em>tile</em> indices to processes.
template <CyclicPmapOrder Order = CyclicPmapOrder::RowMajor>
class CyclicPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::procs_;  ///< The number of processes
  using Pmap::rank_;   ///< The rank of this process
  using Pmap::size_;   ///< The number of tiles mapped among all processes

 private:
  const size_type rows_;       ///< Number of tile rows to be mapped
  const size_type cols_;       ///< Number of tile columns to be mapped
  const size_type proc_cols_;  ///< Number of process columns
  const size_type proc_rows_;  ///< Number of process rows
  size_type rank_row_ = 0;     ///< This rank's row in the process grid
  size_type rank_col_ = 0;     ///< This rank's column in the process grid
  size_type local_rows_ = 0;   ///< The number of rows that belong to this rank
  size_type local_cols_ =
      0;  ///< The number of columns that belong to this rank

  inline size_type coordinate_to_index(size_type i, size_type j) const noexcept {
    if constexpr (is_row_major_v) return i*cols_ + j;
    else                          return i + j*rows_;
  }
 public:
  static constexpr bool is_row_major_v = (Order == CyclicPmapOrder::RowMajor);
  typedef Pmap::size_type size_type;  ///< Size type

  /// Construct process map

  /// \param world The world where the tiles will be mapped
  /// \param rows The number of tile rows to be mapped
  /// \param cols The number of tile columns to be mapped
  /// \param proc_rows The number of process rows in the map
  /// \param proc_cols The number of process columns in the map
  /// \throw TiledArray::Exception When <tt>proc_rows > rows</tt>
  /// \throw TiledArray::Exception When <tt>proc_cols > cols</tt>
  /// \throw TiledArray::Exception When <tt>proc_rows * proc_cols >
  /// world.size()</tt>
  CyclicPmap(World& world, size_type rows, size_type cols, size_type proc_rows,
             size_type proc_cols)
      : Pmap(world, rows * cols),
        rows_(rows),
        cols_(cols),
        proc_cols_(proc_cols),
        proc_rows_(proc_rows) {
    // Check that the size is non-zero
    TA_ASSERT(rows_ >= 1ul);
    TA_ASSERT(cols_ >= 1ul);

    // Check limits of process rows and columns
    TA_ASSERT(proc_rows_ >= 1ul);
    TA_ASSERT(proc_cols_ >= 1ul);
    TA_ASSERT((proc_rows_ * proc_cols_) <= procs_);

    // Compute local size_, if have any
    if (rank_ < (proc_rows_ * proc_cols_)) {
      // Compute rank coordinates
      if constexpr (is_row_major_v) {
        rank_row_ = rank_ / proc_cols_;
        rank_col_ = rank_ % proc_cols_;
      } else {
        rank_row_ = rank_ % proc_rows_;
        rank_col_ = rank_ / proc_rows_;
      }

      local_rows_ =
          (rows_ / proc_rows_) + ((rows_ % proc_rows_) > rank_row_ ? 1ul : 0ul);
      local_cols_ =
          (cols_ / proc_cols_) + ((cols_ % proc_cols_) > rank_col_ ? 1ul : 0ul);

      // Allocate memory for the local tile list
      this->local_size_ = local_rows_ * local_cols_;
    }
  }

  virtual ~CyclicPmap() {}

  /// Access number of rows in the tile index matrix
  size_type nrows() const { return rows_; }
  /// Access number of columns in the tile index matrix
  size_type ncols() const { return cols_; }
  /// Access number of rows in the process matrix
  size_type nrows_proc() const { return proc_rows_; }
  /// Access number of columns in the process matrix
  size_type ncols_proc() const { return proc_cols_; }

  /// Maps \c tile to the processor that owns it

  /// \param tile The tile to be queried
  /// \return Processor that logically owns \c tile
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    // Compute tile coordinate in tile grid
    const size_type tile_row = is_row_major_v ? tile / cols_ : tile % rows_;
    const size_type tile_col = is_row_major_v ? tile % cols_ : tile / rows_;
    // Compute process coordinate of tile in the process grid
    const size_type proc_row = tile_row % proc_rows_;
    const size_type proc_col = tile_col % proc_cols_;
    // Compute the process that owns tile
    const size_type proc = is_row_major_v ? 
        proc_row * proc_cols_ + proc_col : proc_row + proc_col * proc_rows_;

    TA_ASSERT(proc < procs_);

    return proc;
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false .
  virtual bool is_local(const size_type tile) const {
    return (CyclicPmap::owner(tile) == rank_);
  }

 private:
#if 0
  virtual void advance(size_type& value, bool increment) const {
    if constexpr (is_row_major_v) {
      if (increment) {
        auto row = value / cols_;
        const auto row_end = (row + 1) * cols_;
        value += proc_cols_;
        if (value >= row_end) {  // if past the end of row ...
          row += proc_rows_;
          if (row < rows_) {                              // still have tiles
            value = coordinate_to_index(row, rank_col_);  // first tile in this row
          } else                                          // done
            value = size_;
        }
      } else {  // decrement
        auto row = value / cols_;
        const auto row_begin = row * cols_;
        if (value < proc_cols_) {  // protect against unsigned wraparound
          return;
        }
        value -= proc_cols_;
        if (value < row_begin) {  // if past the beginning of row ...
          if (row < proc_rows_)   // protect against unsigned wraparound
            return;
          row -= proc_rows_;
          value = coordinate_to_index(row, rank_col_) +
                  (local_cols_ - 1) * proc_cols_;  // last tile in this row
        }
      }
    } else {
      Pmap::advance(value, increment);
    }
  }
#endif

 public:

#if 0
  virtual const_iterator begin() const {
    const auto proc_index = coordinate_to_index(rank_row_, rank_col_);
    return this->local_size_ > 0
               ? Iterator(*this, proc_index, this->size_, proc_index, 
                          false, is_row_major_v)
               : end();  // make end() if empty
  }
  virtual const_iterator end() const {
    const auto proc_index = coordinate_to_index(rank_row_, rank_col_);
    return this->local_size_ > 0
               ? Iterator(*this, proc_index, this->size_,
                          this->size_, false, is_row_major_v)
               : Iterator(*this, 0, this->size_, this->size_, false, is_row_major_v);
  }
#endif

};  // class CyclicPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_CYCLIC_PMAP_H__INCLUDED
