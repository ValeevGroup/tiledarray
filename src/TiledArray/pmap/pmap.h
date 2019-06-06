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
#include <boost/iterator/iterator_facade.hpp>

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

  protected:
    const size_type rank_; ///< The rank of this process
    const size_type procs_; ///< The number of processes
    const size_type size_; ///< The number of tiles mapped among all processes
    std::vector<size_type> local_; ///< A list of local tiles
    size_type local_size_; ///< The number of tiles mapped to this process (if local_ is not empty, this equals local_.size()); if local_size_known()==false this is not used

  private:
    Pmap(const Pmap&) = delete;
    Pmap& operator=(const Pmap&) = delete;

  public:

    /// Process map constructor

    /// \param world The world where the tiles will be mapped
    /// \param size The number of tiles to be mapped
    Pmap(World& world, const size_type size) :
      rank_(world.rank()), procs_(world.size()), size_(size), local_(), local_size_(0) {}

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

    /// Queries whether local size is known

    /// \return true if the number of local elements is known
    /// \note Override if it is too expensive to precompute
    virtual bool known_local_size() const { return true; }

    /// Local size accessor

    /// \return The number of local elements
    /// \warning if \c size()>0 asserts that \c known_local_size()==true
    size_type local_size() const {
      if (size_ > 0)
        TA_ASSERT(known_local_size()==true);
      return local_size_;
    }

    /// Check if there are any local elements

    /// \return \c true when there are no local tiles, otherwise \c false .
    /// \warning if \c size()>0 asserts that \c known_local_size()==true
    bool empty() const {
      if (size_ > 0)
        TA_ASSERT(known_local_size()==true);
      return local_size_ == 0;
    }

    /// Replicated array status

    /// \return \c true if the array is replicated, and false otherwise
    virtual bool is_replicated() const { return false; }

    /// \name Iteration
    /// @{

   private:

    virtual void advance(size_type& value, bool increment) const {
      if (increment)
        ++value;
      else
        --value;
    }

   public:

    /// iterates over an integer range, optionally checking locality of each element, or simply use local_::iterator
    class Iterator : public boost::iterator_facade<Iterator, const size_type, boost::bidirectional_traversal_tag> {
     public:
      Iterator(const Pmap& pmap, size_type begin_idx, size_type end_idx, size_type val, bool checking, bool use_pmap_advance = false) :
      pmap_(&pmap), use_it_(false), value_(val), begin_idx_(begin_idx), end_idx_(end_idx), checking_(checking), use_pmap_advance_(use_pmap_advance) {
        if (value_ != end_idx) {  // unless this is end
          if (checking_ && value_ == begin_idx_) { // create valid begin iterator if needed
            while (value_ < end_idx_ && !pmap_->is_local(value_)) {
              ++value_;
            }
          }
          else  // else assert that this is a valid iterator
            TA_ASSERT(pmap_->is_local(value_));
        }
      }
      Iterator(const Pmap& pmap, std::vector<size_type>::const_iterator it) : use_it_(true), it_(it) {
        TA_ASSERT(it_ == pmap_->local_.end() || pmap_->is_local(*it_));
      }

     private:
      friend class boost::iterator_core_access;
      const Pmap* pmap_;

      bool use_it_;

      // have iterator
      std::vector<size_type>::const_iterator it_ = pmap_->local_.end();

      /// have range
      size_type value_ = 0;
      size_type begin_idx_ = 0;
      size_type end_idx_ = 0;
      bool checking_ = true;
      bool use_pmap_advance_ = false;

      void increment() {
        if (use_it_) {
          TA_ASSERT(it_ != pmap_->local_.end());
          ++it_;
        }
        else {
          if (value_ == end_idx_)
            return;
          if (!use_pmap_advance_) {
            ++value_;
            if (checking_) {
              while (value_ < end_idx_ && !pmap_->is_local(value_)) {
                ++value_;
              }
            }
          }
          else {
            pmap_->advance(value_, true);
          }
          if (value_ > end_idx_)  // normalize if past end
            value_ = end_idx_;
          TA_ASSERT(value_ == end_idx_ || pmap_->is_local(value_));
        }
      }
      void decrement() {
        if (use_it_) {
          TA_ASSERT(it_ != pmap_->local_.begin()); // no good will happen if we decrement begin
          TA_ASSERT(it_ != pmap_->local_.end());  // don't decrement the end since it's an invalid iterator
          --it_;
        }
        else {
          TA_ASSERT(value_ != begin_idx_);
          TA_ASSERT(value_ != end_idx_);
          if (!use_pmap_advance_) {
            --value_;
            if (checking_) {
              while (value_ > begin_idx_ && !pmap_->is_local(value_)) {
                --value_;
              }
            }
          }
          else {
            pmap_->advance(value_, false);
          }
          if (value_ < begin_idx_)  // normalize if past begin
            value_ = begin_idx_;
          TA_ASSERT(value_ == begin_idx_ || pmap_->is_local(value_));
        }
      }

      bool equal(Iterator const& other) const {
        TA_ASSERT(this->pmap_ == other.pmap_ && this->use_it_ == other.use_it_);
        return use_it_ ? this->it_ == other.it_ : this->value_ == other.value_;
      }

      const size_type& dereference() const { return use_it_ ? *it_ : value_; }
    };
    friend class Iterator;

    typedef Iterator const_iterator; ///< Iterator type

    /// Begin local element iterator

    /// \return An iterator that points to the beginning of the local element set
    virtual const_iterator begin() const { return !local_.empty() ? Iterator(*this, local_.begin()) : Iterator(*this, 0, size_, 0, true); }

    /// End local element iterator

    /// \return An iterator that points to the beginning of the local element set
    virtual const_iterator end() const { return !local_.empty() ? Iterator(*this, local_.end()) : Iterator(*this, 0, size_, size_, true); }

    /// Begin local element iterator

    /// \return An iterator that points to the beginning of the local element set
    const const_iterator cbegin() const { return begin(); }

    /// End local element iterator

    /// \return An iterator that points to the beginning of the local element set
    const const_iterator cend() const { return end(); }

    /// @}

  }; // class Pmap

}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_H__INCLUDED
