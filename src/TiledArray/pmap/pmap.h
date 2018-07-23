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
 *  pmap.h
 *  April 24, 2012
 *
 */

#ifndef TILEDARRAY_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_H__INCLUDED

#include <TiledArray/external/madness.h>
#include <TiledArray/error.h>

namespace TiledArray {

  /// Process map

  /// This is the base and interface class for other process maps. It provides
  /// access to the local tile iterator and basic process map information.
  /// Derived classes are responsible for distribution of tiles. The general
  /// idea of process map objects is to compute process owners with an O(1)
  /// algorithm to provide fast access to tile owner information and avoid
  /// storage of process map. A cache a list of local tiles is stored so that
  /// algorithms that need to iterate over local tiles can do so without
  /// computing the owner of all tiles. The algorithm to generate the cached
  /// list of local tiles and the memory requirement should scale as
  /// O(tiles/processes), if possible.
  class Pmap {
  public:
    typedef std::size_t size_type; ///< Size type
    typedef std::vector<size_type>::const_iterator const_iterator; ///< Iterator type

  protected:
    const size_type rank_; ///< The rank of this process
    const size_type procs_; ///< The number of processes
    const size_type size_; ///< The number of tiles mapped among all processes
    std::vector<size_type> local_; ///< A list of local tiles

  private:
    // Not allowed
    Pmap(const Pmap&);
    Pmap& operator=(const Pmap&);

  public:

    /// Process map constructor

    /// \param world The world where the tiles will be mapped
    /// \param size The number of processes to be mapped
    Pmap(World& world, const size_type size) :
      rank_(world.rank()), procs_(world.size()), size_(size), local_()
    {
      TA_ASSERT(size_ > 0ul);
    }

    virtual ~Pmap() { }

    /// Maps \c tile to the processor that owns it

    /// \param tile The tile to be queried
    /// \return Processor that logically owns \c tile
    virtual size_type owner(const size_type tile) const = 0;

    /// Check that the tile is owned by this process

    /// \param tile The tile to be checked
    /// \return \c true if \c tile is owned by this process, otherwise \c false .
    virtual bool is_local(const size_type tile) const = 0;

    /// Size accessor

    /// \return The number of elements
    size_type size() const { return size_; }

    /// Process rank accessor

    /// \return The rank of this process
    size_type rank() const { return rank_; }

    /// Process count accessor

    /// \return The number of processes
    size_type procs() const { return procs_; }

    /// Local size accessor

    /// \return The number of local elements
    size_type local_size() const { return local_.size(); }

    /// Check if there are any local elements

    /// \return \c true when there are no local tiles, otherwise \c false .
    bool empty() const { return local_.empty(); }

    /// Replicated array status

    /// \return \c true if the array is replicated, and false otherwise
    virtual bool is_replicated() const { return false; }

    /// Begin local element iterator

    /// \return An iterator that points to the beginning of the local element set
    const_iterator begin() const { return local_.begin(); }

    /// End local element iterator

    /// \return An iterator that points to the beginning of the local element set
    const_iterator end() const { return local_.end(); }

  }; // class Pmap

}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_H__INCLUDED
