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

#include <TiledArray/error.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/madness.h>
#include <algorithm>

namespace TiledArray {
  namespace detail {

    /// A Replicated process map

    /// Defines a process map where all processes own data.
    class ReplicatedPmap : public Pmap {
    protected:

      // Import Pmap protected variables
      using Pmap::local_;
      using Pmap::rank_;
      using Pmap::procs_;
      using Pmap::size_;

    public:
      typedef Pmap::size_type size_type; ///< Size type

      /// Construct Blocked map

      /// \param world A reference to the world
      /// \param size The number of elements to be mapped
      ReplicatedPmap(madness::World& world, std::size_t size) :
          Pmap(world, size)
      {
        // Construct a map of all local processes
        local_.reserve(size_);
        for(std::size_t i = 0; i < size_; ++i) {
          TA_ASSERT(ReplicatedPmap::owner(i));
          local_.push_back(i);
        }
      }

      virtual ~ReplicatedPmap() { }

      /// Maps \c tile to the processor that owns it

      /// \param tile The tile to be queried
      /// \return Processor that logically owns \c tile
      virtual ProcessID owner(const size_type tile) const {
        TA_ASSERT(tile < size_);
        return rank_;
      }

      /// Replicated array status

      /// \return \c true if the array is replicated, and false otherwise
      virtual bool is_replicated() const { return true; }

    }; // class MapByRow

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_REPLICATED_PMAP_H__INCLUDED
