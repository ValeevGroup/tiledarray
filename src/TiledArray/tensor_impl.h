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

#ifndef TILEDARRAY_TENSOR_IMPL_H__INCLUDED
#define TILEDARRAY_TENSOR_IMPL_H__INCLUDED

#include <TiledArray/distributed_storage.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/bitset.h>

namespace TiledArray {
  namespace detail {

    /// Tensor implementation and base for other tensor implementation objects

    /// This implementation object holds the data for tensor object, which
    /// includes tiled range, shape, and tiles. The tiles are held in a
    /// distributed container, stored according to a given process map.
    /// \tparam Tile The tile or value_type of this tensor
    /// \note The process map must be set before data elements can be set.
    /// \note It is the users responsibility to ensure the process maps on all
    /// nodes are identical.
    template <typename Tile>
    class TensorImpl : private NO_DEFAULTS{
    public:
      typedef TiledRange trange_type; ///< Tiled range type
      typedef typename trange_type::range_type range_type; ///< Tile range type
      typedef Tile value_type; ///< Tile or data type
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; ///< The data container type
      typedef typename storage_type::size_type size_type; ///< Size type
      typedef typename storage_type::const_iterator const_iterator; ///< Constant iterator type
      typedef typename storage_type::iterator iterator; ///< Iterator type
      typedef typename storage_type::future const_reference; ///< Constant reference type
      typedef typename storage_type::future reference; ///< Reference type
      typedef Pmap<size_type> pmap_interface; ///< Process map interface type

    private:

      trange_type trange_; ///< Tiled range type
      Bitset<> shape_; ///< Tensor shape (zero size == dense)
      storage_type data_; ///< Tile container

    public:

      /// Constructor

      /// The size of shape must be equal to the volume of the tiled range tiles.
      /// Also, the volume of trange must remain constant. This restriction allows
      /// the tiled range to be permuted, but not resized.
      /// \param arg The argument
      /// \param op The element transform operation
      /// \throw TiledArray::Exception When the size of shape is not equal to
      /// zero
      TensorImpl(madness::World& world, const trange_type& trange, const Bitset<>& shape) :
        trange_(trange), shape_(shape), data_(world, trange_.tiles().volume())
      {
        TA_ASSERT((shape_.size() == trange_.tiles().volume()) || (shape_.size() == 0ul));
      }

      /// Virtual destructor
      virtual ~TensorImpl() { }

      /// Tensor process map accessor

      /// \return A shared pointer to the process map of this tensor
      /// \throw nothing
      const std::shared_ptr<pmap_interface>& pmap() const { return data_.get_pmap(); }

      /// Initialize pmap

      /// \param pmap The process map
      /// \throw TiledArray::Exception When the process map has already been set
      /// \throw TiledArray::Exception When \c pmap is \c NULL
      void pmap(const std::shared_ptr<pmap_interface>& pmap) { data_.init(pmap); }

      /// Evaluate tensor to destination

      /// \tparam Dest The destination tensor type
      /// \param dest The destination to evaluate this tensor to
      /// \throw TiledArray::Exception when the trange of \c dest is not equal
      /// to the trange of this tensor.
      template <typename Dest>
      void eval_to(Dest& dest) {
        TA_ASSERT(trange() == dest.trange());

        // Add result tiles to dest
        typename pmap_interface::const_iterator end = data_.get_pmap()->end();
        typename pmap_interface::const_iterator it = data_.get_pmap()->begin();
        if(is_dense()) {
          for(; it != end; ++it)
            dest.set(*it, move(*it));
        } else {
          for(; it != end; ++it)
            if(! is_zero(*it))
              dest.set(*it, move(*it));
        }
      }

      /// Tensor tile size array accessor

      /// \return The size array of the tensor tiles
      /// \throw nothing
      const range_type& range() const { return trange_.tiles(); }

      /// Tensor tile volume accessor

      /// \return The number of tiles in the tensor
      /// \throw nothing
      size_type size() const { return trange_.tiles().volume(); }

      /// Local element count

      /// This function is primarily available for debugging  purposes. The
      /// returned value is volatile and may change at any time; you should not
      /// rely on it in your algorithms.
      /// \return The current number of local tiles stored in the tensor.
      size_type local_size() const { return data_.size(); }

      /// Query a tile owner

      /// \tparam Index The index type
      /// \param i The tile index to query
      /// \return The process ID of the node that owns tile \c i
      /// \throw TiledArray::Exception When \c i is outside the tiled range tile
      /// range
      /// \throw TiledArray::Exception When the process map has not been set
      template <typename Index>
      ProcessID owner(const Index& i) const {
        TA_ASSERT(trange_.tiles().includes(i));
        return data_.owner(trange_.tiles().ord(i));
      }

      /// Query for a locally owned tile

      /// \tparam Index The index type
      /// \param i The tile index to query
      /// \return \c true if the tile is owned by this node, otherwise \c false
      /// \throw TiledArray::Exception When the process map has not been set
      template <typename Index>
      bool is_local(const Index& i) const {
        TA_ASSERT(trange_.tiles().includes(i));
        return data_.is_local(trange_.tiles().ord(i));
      }

      /// Query for a zero tile

      /// \tparam Index The index type
      /// \param i The tile index to query
      /// \return \c true if the tile is zero, otherwise \c false
      /// \throw TiledArray::Exception When \c i is outside the tiled range tile
      /// range
      template <typename Index>
      bool is_zero(const Index& i) const {
        TA_ASSERT(trange_.tiles().includes(i));
        if(is_dense())
          return false;
        return ! (shape_[trange_.tiles().ord(i)]);
      }

      /// Query the density of the tensor

      /// \return \c true if the tensor is dense, otherwise false
      /// \throw nothing
      bool is_dense() const { return shape_.size() == 0ul; }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      /// \throw TiledArray::Exception When this tensor is dense
      const TiledArray::detail::Bitset<>& shape() const {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      /// \throw TiledArray::Exception When this tensor is dense
      TiledArray::detail::Bitset<>& shape() {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Set the shape

      /// \param s The new shape
      /// \throw TiledArray::Exception When the size of \c s is not equal to the
      /// size of this tensor or zero.
      void shape(TiledArray::detail::Bitset<> s) {
        TA_ASSERT((s.size() == trange_.tiles().volume()) || (s.size() == 0ul));
        s.swap(shape_);
      }

      /// Set shape values

      /// Modify the shape value for tile \c i to \c value
      /// \tparam Index The index type
      /// \param i Tile index
      /// \param value The value of the tile
      /// \throw TiledArray::Exception When this tensor is dense
      template <typename Index>
      void shape(const Index& i, bool value = true) {
        TA_ASSERT(trange_.tiles().includes(i));
        TA_ASSERT(! is_dense());
        shape_.set(trange_.tiles().ord(i), value);
      }

      /// Tiled range accessor

      /// \return The tiled range of the tensor
      const trange_type& trange() const { return trange_; }

      /// Set tiled range

      /// \param tr Tiled range to set
      void trange(const trange_type& tr) {
        TA_ASSERT(tr.tiles().volume() == trange_.tiles().volume());
        trange_ = tr;
      }

      /// Tile accessor

      /// \tparam Index The index type
      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When tile \c i is zero
      template <typename Index>
      const_reference operator[](const Index& i) const {
        TA_ASSERT(! is_zero(i));
        return data_[trange_.tiles().ord(i)];
      }

      /// Tile accessor

      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When tile \c i is zero
      template <typename Index>
      reference operator[](const Index& i) {
        TA_ASSERT(! is_zero(i));
        return data_[trange_.tiles().ord(i)];
      }

      /// Tile move

      /// Tile is removed after it is set.
      /// \tparam Index The index type
      /// \param i The tile index
      /// \return Tile \c i
      /// \throw TiledArray::Exception When tile \c i is zero
      template <typename Index>
      const_reference move(const Index& i) {
        TA_ASSERT(! is_zero(i));
        return data_.move(trange_.tiles().ord(i));
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      const_iterator begin() const { return data_.begin(); }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      iterator begin() { return data_.begin(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const { return data_.end(); }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      iterator end() { return data_.end(); }

      /// World accessor

      /// \return A reference to the world that contains this tensor
      madness::World& get_world() const { return data_.get_world(); }

      /// Set tile

      /// Set the tile at \c i with \c value . \c Value type may be \c value_type ,
      /// \c madness::Future<value_type> , or
      /// \c madness::detail::MoveWrapper<value_type> .
      /// \tparam Index The index type
      /// \tparam Value The value type
      /// \param i The index of the tile to be set
      /// \param value The object tat contains the tile value
      template <typename Index, typename Value>
      void set(const Index& i, const Value& value) {
        TA_ASSERT(! is_zero(i));
        data_.set(trange_.tiles().ord(i), value);
      }

      /// Clear the tile data

      /// Remove all local tiles from the tensor.
      /// \note: Any tiles will remain in memory until the last reference
      /// is destroyed. This function only removes them from the container.
      void clear() { data_.clear(); }

    }; // class TensorImplBase

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_TENSOR_IMPL_H__INCLUDED
