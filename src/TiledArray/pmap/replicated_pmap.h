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
 *  replicated_pmap.h
 *  February 8, 2013
 *
 */

#ifndef TILEDARRAY_PMAP_REPLICATED_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_REPLICATED_PMAP_H__INCLUDED

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {
namespace detail {

/// A Replicated process map

/// Defines a process map where all processes own data.
class ReplicatedPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::procs_;  ///< The number of processes
  using Pmap::rank_;   ///< The rank of this process
  using Pmap::size_;   ///< The number of tiles mapped among all processes

 public:
  typedef Pmap::size_type size_type;  ///< Size type

  /// Construct a replicated map

  /// \param world A reference to the world
  /// \param size The number of elements to be mapped
  ReplicatedPmap(World& world, size_type size) : Pmap(world, size) {
    this->local_size_ = this->size_;
  }

  virtual ~ReplicatedPmap() {}

  /// Maps \c tile to the processor that owns it

  /// \param tile The tile to be queried
  /// \return Processor that logically owns \c tile
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return rank_;
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false .
  virtual bool is_local(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return true;
  }

  /// Replicated array status

  /// \return \c true if the array is replicated, and false otherwise
  virtual bool is_replicated() const { return true; }

  virtual const_iterator begin() const {
    return Iterator(*this, 0, this->size_, 0, false);
  }
  virtual const_iterator end() const {
    return Iterator(*this, 0, this->size_, this->size_, false);
  }

};  // class ReplicatedPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_REPLICATED_PMAP_H__INCLUDED
