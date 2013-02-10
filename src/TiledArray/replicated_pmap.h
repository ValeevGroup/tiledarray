/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_REPLICATED_PMAP_H__INCLUDED
#define TILEDARRAY_REPLICATED_PMAP_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/pmap.h>
#include <TiledArray/madness.h>
#include <algorithm>

namespace TiledArray {
  namespace detail {

    /// A Replicated process map

    /// Defines a process map where all processes own data.
    class ReplicatedPmap : public Pmap<std::size_t> {
    public:
      typedef Pmap<std::size_t>::key_type key_type; ///< Key type
      typedef Pmap<std::size_t>::const_iterator const_iterator;

      /// Construct Blocked map

      /// \param world A reference to the world
      /// \param size The number of elements to be mapped
      ReplicatedPmap(madness::World& world, std::size_t size) :
          size_(size),
          rank_(world.rank()),
          procs_(world.size()),
          local_()
      { }

    private:

      ReplicatedPmap(const ReplicatedPmap& other) :
          size_(other.size_),
          rank_(other.rank_),
          procs_(other.procs_),
          local_()
      { }

    public:

      ~ReplicatedPmap() { }

      virtual void set_seed(madness::hashT) {
        // Construct a map of all local processes
        local_.reserve(size_);
        for(std::size_t i = 0; i < size_; ++i)
          local_.push_back(i);
      }

      /// Create a copy of this pmap

      /// \return A shared pointer to the new object
      virtual std::shared_ptr<Pmap<key_type> > clone() const {
        return std::shared_ptr<Pmap<key_type> >(new ReplicatedPmap(*this));
      }

      /// Maps key to processor

      /// \param key Key for container
      /// \return Processor that logically owns the key
      ProcessID owner(const key_type& key) const {
        TA_ASSERT(key < size_);
        return rank_;
      }

      /// Local size accessor

      /// \return The number of local elements
      std::size_t local_size() const { return local_.size(); }

      /// Local elements

      /// \return \c true when there are no local elements, otherwise \c false .
      bool empty() const { return local_.empty(); }

      /// Begin local element iterator

      /// \return An iterator that points to the beginning of the local element set
      const_iterator begin() const { return local_.begin(); }

      /// End local element iterator

      /// \return An iterator that points to the beginning of the local element set
      const_iterator end() const { return local_.end(); }

      /// Local element vector accessor

      /// \return A const reference to a vector of local elements
      virtual const std::vector<key_type>& local() const { return local_; }

    private:

      const std::size_t size_; ///< The number of elements to be mapped
      const std::size_t rank_;
      const std::size_t procs_; ///< The number of processes in the world
      std::vector<std::size_t> local_; ///< A list of local elements
    }; // class MapByRow

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_REPLICATED_PMAP_H__INCLUDED
