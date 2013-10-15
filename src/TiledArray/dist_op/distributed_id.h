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
 *  distributed_id.h
 *  Oct 13, 2013
 *
 */

#ifndef TILEDARRAY_DIST_OP_DISTRIBUTED_ID_H__INCLUDED
#define TILEDARRAY_DIST_OP_DISTRIBUTED_ID_H__INCLUDED

#include <TiledArray/madness.h>

namespace TiledArray {
  namespace dist_op {

    /// Distributed ID which is used to synchronize data in distributed operations
    typedef std::pair<madness::uniqueidT, std::size_t> DistributedID;

    /// DistCache key_type comparison operator

    /// \param left The first key to compare
    /// \param right The second key to compare
    /// \return \c true when \c first and \c second of \c left and \c right are
    /// equal, otherwise \c false
    inline bool operator==(const DistributedID& left, const DistributedID& right) {
      return (left.first == right.first) && (left.second == right.second);
    }

  }  // namespace dist_op
} // namespace TiledArray

namespace std {

  /// Hash a DistributedID

  /// \param id The distributed id to be hashed
  /// \return The hash value of \c id
  inline madness::hashT hash_value(const TiledArray::dist_op::DistributedID& id) {
      madness::hashT seed = madness::hash_value(id.first);
      madness::detail::combine_hash(seed, madness::hash_value(id.second));

      return seed;
  }

} // namespace std

#endif // TILEDARRAY_DIST_OP_DISTRIBUTED_ID_H__INCLUDED
