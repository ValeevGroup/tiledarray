/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2021  Virginia Tech
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
 *  user_pmap.h
 *  October 11, 2021
 *
 */

#ifndef TILEDARRAY_PMAP_USER_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_USER_PMAP_H__INCLUDED

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {
namespace detail {

/// Process map specified by a user callable
class UserPmap : public Pmap {
 protected:
  // Import Pmap protected variables
  using Pmap::procs_;  ///< The number of processes
  using Pmap::rank_;   ///< The rank of this process
  using Pmap::size_;   ///< The number of tiles mapped among all processes

 public:
  typedef Pmap::size_type size_type;  ///< Size type

  /// Constructs map that does not know the number of local elements

  /// \tparam Index2Rank a callable type with `size_type(size_t)` signature
  /// \param world A reference to the world
  /// \param size The number of elements to be mapped
  /// \param i2r A callable specifying the index->rank map
  template <typename Index2Rank>
  UserPmap(World& world, size_type size, Index2Rank&& i2r)
      : Pmap(world, size), index2rank_(std::forward<Index2Rank>(i2r)) {}

  /// Constructs map that does not know the number of local elements

  /// \tparam Index2Rank a callable type with `size_type(size_t)` signature
  /// \param world A reference to the world
  /// \param size The number of elements to be mapped
  /// \param local_size The number of elements mapped to this rank
  /// \param i2r A callable specifying the index->rank map
  template <typename Index2Rank>
  UserPmap(World& world, size_type size, size_type local_size, Index2Rank&& i2r)
      : Pmap(world, size, local_size),
        known_local_size_(true),
        index2rank_(std::forward<Index2Rank>(i2r)) {}

  virtual ~UserPmap() {}

  /// Maps \c tile to the processor that owns it

  /// \param tile The tile to be queried
  /// \return Processor that logically owns \c tile
  virtual size_type owner(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return index2rank_(tile);
  }

  /// Check that the tile is owned by this process

  /// \param tile The tile to be checked
  /// \return \c true if \c tile is owned by this process, otherwise \c false .
  virtual bool is_local(const size_type tile) const {
    TA_ASSERT(tile < size_);
    return owner(tile) == rank_;
  }

  virtual bool known_local_size() const { return known_local_size_; }

  virtual const_iterator begin() const {
    return Iterator(*this, 0, this->size_, 0, false);
  }
  virtual const_iterator end() const {
    return Iterator(*this, 0, this->size_, this->size_, false);
  }

 private:
  bool known_local_size_ = false;
  std::function<size_type(size_type)> index2rank_;
};  // class UserPmap

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_USER_PMAP_H__INCLUDED
