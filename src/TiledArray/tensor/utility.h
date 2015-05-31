/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  utility.h
 *  May 31, 2015
 *
 */

#ifndef TILEDARRAY_UTILITY_H__INCLUDED
#define TILEDARRAY_UTILITY_H__INCLUDED

namespace TiledArray {
  namespace detail {

    // The following function are used as helper function for composing generic
    // tensor algebra algorithms.


    /// Check for congruent range objects

    /// \tparam Left The left-hand tensor type
    /// \tparam Right The right-hand tensor type
    /// \param left The left-hand tensor
    /// \param right The right-hand tensor
    /// \return \c true if the lower and upper bounds of the the left- and
    /// right-hand tensor ranges are equal, otherwise \c false
    template <typename Left, typename Right>
    inline bool is_range_congruent(const Left& left, const Right& right) {
      return left.range() == right.range();
    }

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_UTILITY_H__INCLUDED
