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
 */

#ifndef TILEDARRAY_TENSOR_IMPL_H__INCLUDED
#define TILEDARRAY_TENSOR_IMPL_H__INCLUDED

#include <TiledArray/distributed_storage.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/bitset.h>

namespace TiledArray {
  namespace detail {

    // Forward declaration
    template <typename> class TensorImpl;
    template <typename> class TensorIterator;
    template <typename> class ConstTensorIterator;

    /// Tensor tile reference
    template <typename Tile>
    class TileReference {
    private:

      template <typename>
      friend class ConstTensorIterator;

      TensorImpl<Tile>* tensor_; ///< The tensor that owns the referenced tile
      std::size_t index_; ///< The index of the tensor

      // Not allowed
      TileReference<Tile>& operator=(const TileReference<Tile>&);
    public:

      TileReference(TensorImpl<Tile>* tensor, const std::size_t index) :
        tensor_(tensor), index_(index)
      { }

      TileReference(const TileReference<Tile>& other) :
        tensor_(other.tensor_), index_(other.index_)
      { }

      template <typename Value>
      TileReference<Tile>& operator=(const Value& value) {
        tensor_->set(index_, value);
        return *this;
      }

      typename TensorImpl<Tile>::future get() const { return (*tensor_)[index_]; }

      operator typename TensorImpl<Tile>::future() const { return get(); }
    }; // class reference


    /// Distributed tensor iterator

    /// This iterator will reference local tiles for a TensorImpl object. It can
    /// be used to get or set futures to a tile, or access the coordinate and
    /// ordinal index of the tile.
    /// \tparam Tile The tile time of the TensorImpl object
    template <typename Tile>
    class TensorIterator {
    private:
      // Give access to other iterator types.
      template <typename>
      friend class ConstTensorIterator;

      TensorImpl<Tile>* tensor_;
      typename TensorImpl<Tile>::pmap_interface::const_iterator it_;

    public:
      typedef ptrdiff_t difference_type;                        ///< Difference type
      typedef typename TensorImpl<Tile>::future value_type; ///< Iterator dereference value type
      typedef PointerProxy<value_type> pointer;                 ///< Pointer type to iterator value
      typedef TileReference<Tile> reference;                             ///< Reference type to iterator value
      typedef std::forward_iterator_tag iterator_category;        ///< Iterator category type
      typedef TensorIterator<Tile> TensorIterator_;       ///< This object type
      typedef typename TensorImpl<Tile>::range_type::index index_type;
      typedef typename TensorImpl<Tile>::size_type ordinal_type;

      /// Default constructor
      TensorIterator() : tensor_(NULL), it_() { }

      /// Constructor
      TensorIterator(TensorImpl<Tile>* tensor,
          typename TensorImpl<Tile>::pmap_interface::const_iterator it) :
        tensor_(tensor), it_(it)
      { }

      /// Copy constructor

      /// \param other The transform iterator to copy
      TensorIterator(const TensorIterator_& other) :
        tensor_(other.tensor_), it_(other.it_)
      { }

      /// Copy const iterator constructor

      /// \param other The transform iterator to copy
      TensorIterator(const ConstTensorIterator<Tile>& other) :
        tensor_(other.tensor_), it_(other.it_)
      { }

      /// Copy operator

      /// \param other The transform iterator to copy
      /// \return A reference to this object
      TensorIterator_& operator=(const TensorIterator_& other) {
        tensor_ = other.tensor_;
        it_ = other.it_;

        return *this;
      }

      /// Copy operator

      /// \param other The transform iterator to copy
      /// \return A reference to this object
      TensorIterator_& operator=(const ConstTensorIterator<Tile>& other) {
        tensor_ = other.tensor_;
        it_ = other.it_;

        return *this;
      }

      /// Prefix increment operator

      /// \return A reference to this object after it has been incremented.
      TensorIterator_& operator++() {
        TA_ASSERT(tensor_);
        ++it_;
        return *this;
      }

      /// Post-fix increment operator

      /// \return A copy of this object before it is incremented.
      TensorIterator_ operator++(int) {
        TA_ASSERT(tensor_);
        TensorIterator_ tmp(*this);
        ++it_;
        return tmp;
      }

      /// Equality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are equal to each other, otherwise
      /// \c false.
      bool operator==(const TensorIterator_& other) const {
        return (tensor_ == other.tensor_) && (it_ == other.it_);
      }

      /// Inequality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are not equal to each other,
      /// otherwise \c false.
      bool operator!=(const TensorIterator_& other) const {
        return (tensor_ != other.tensor_) || (it_ != other.it_);
      }

      /// Equality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are equal to each other, otherwise
      /// \c false.
      bool operator==(const ConstTensorIterator<Tile>& other) const {
        return (tensor_ == other.tensor_) && (it_ == other.it_);
      }

      /// Inequality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are not equal to each other,
      /// otherwise \c false.
      bool operator!=(const ConstTensorIterator<Tile>& other) const {
        return (tensor_ != other.tensor_) || (it_ != other.it_);
      }

      /// Dereference operator

      /// \return A referenc to the current tile future.
      reference operator*() const {
        TA_ASSERT(tensor_);
        return reference(tensor_, *it_);
      }

      /// Arrow dereference operator

      /// \return A pointer-proxy to the current tile
      pointer operator->() const {
        TA_ASSERT(tensor_);
        return pointer((*tensor_)[*it_]);
      }

      /// Tile coordinate index accessor

      /// \return The coordinate index of the current tile
      index_type index() const {
        TA_ASSERT(tensor_);
        return tensor_->range().idx(*it_);
      }

      /// Tile ordinal index accessor

      /// \return The ordinal index of the current tile
      ordinal_type ordinal() const {
        TA_ASSERT(tensor_);
        return *it_;
      }

    }; // class TensorIterator

    /// Distributed tensor iterator

    /// This iterator will reference local tiles for a TensorImpl object. It can
    /// be used to get or set futures to a tile, or access the coordinate and
    /// ordinal index of the tile.
    /// \tparam Tile The tile time of the TensorImpl object
    template <typename Tile>
    class ConstTensorIterator {
    private:
      // Give access to other iterator types.
      template <typename>
      friend class TensorIterator;

      TensorImpl<Tile>* tensor_;
      typename TensorImpl<Tile>::pmap_interface::const_iterator it_;

    public:
      typedef ptrdiff_t difference_type; ///< Difference type
      typedef typename TensorImpl<Tile>::future value_type; ///< Iterator dereference value type
      typedef PointerProxy<value_type> pointer; ///< Pointer type to iterator value
      typedef value_type reference; ///< Reference type to iterator value
      typedef std::forward_iterator_tag iterator_category; ///< Iterator category type
      typedef ConstTensorIterator<Tile> ConstTensorIterator_; ///< This object type
      typedef typename TensorImpl<Tile>::range_type::index index_type; ///< Coordinate index type
      typedef typename TensorImpl<Tile>::size_type ordinal_type; ///< Ordinal index type

      /// Default constructor
      ConstTensorIterator() : tensor_(NULL), it_() { }

      /// Constructor
      ConstTensorIterator(TensorImpl<Tile>& tensor,
          typename TensorImpl<Tile>::pmap_interface::const_iterator it) :
        tensor_(&tensor), it_(it)
      { }

      /// Copy constructor

      /// \param other The transform iterator to copy
      ConstTensorIterator(const ConstTensorIterator_& other) :
        tensor_(other.tensor_), it_(other.it_)
      { }

      /// Copy const iterator constructor

      /// \param other The transform iterator to copy
      ConstTensorIterator(const TensorIterator<Tile>& other) :
        tensor_(other.tensor_), it_(other.it_)
      { }

      /// Copy operator

      /// \param other The transform iterator to copy
      /// \return A reference to this object
      ConstTensorIterator_& operator=(const ConstTensorIterator_& other) {
        tensor_ = other.tensor_;
        it_ = other.it_;

        return *this;
      }

      /// Copy operator

      /// \param other The transform iterator to copy
      /// \return A reference to this object
      ConstTensorIterator_& operator=(const TensorIterator<Tile>& other) {
        tensor_ = other.tensor_;
        it_ = other.it_;

        return *this;
      }

      /// Prefix increment operator

      /// \return A reference to this object after it has been incremented.
      ConstTensorIterator_& operator++() {
        TA_ASSERT(tensor_);
        ++it_;
        return *this;
      }

      /// Post-fix increment operator

      /// \return A copy of this object before it is incremented.
      ConstTensorIterator_ operator++(int) {
        TA_ASSERT(tensor_);
        ConstTensorIterator_ tmp(*this);
        ++it_;
        return tmp;
      }

      /// Equality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are equal to each other, otherwise
      /// \c false.
      bool operator==(const ConstTensorIterator_& other) const {
        return (tensor_ == other.tensor_) && (it_ == other.it_);
      }

      /// Inequality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are not equal to each other,
      /// otherwise \c false.
      bool operator!=(const ConstTensorIterator_& other) const {
        return (tensor_ != other.tensor_) || (it_ != other.it_);
      }

      /// Equality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are equal to each other, otherwise
      /// \c false.
      bool operator==(const TensorIterator<Tile>& other) const {
        return (tensor_ == other.tensor_) && (it_ == other.it_);
      }

      /// Inequality operator

      /// \param other The iterator to compare to this iterator.
      /// \return \c true when the iterators are not equal to each other,
      /// otherwise \c false.
      bool operator!=(const TensorIterator<Tile>& other) const {
        return (tensor_ != other.tensor_) || (it_ != other.it_);
      }

      /// Dereference operator

      /// \return A referenc to the current tile future.
      reference operator*() const {
        TA_ASSERT(tensor_);
        return (*tensor_)[*it_];
      }

      /// Arrow dereference operator

      /// \return A pointer-proxy to the current tile
      pointer operator->() const {
        TA_ASSERT(tensor_);
        return pointer((*tensor_)[*it_]);
      }

      /// Tile coordinate index accessor

      /// \return The coordinate index of the current tile
      index_type index() const {
        TA_ASSERT(tensor_);
        return tensor_->range().idx(*it_);
      }

      /// Tile ordinal index accessor

      /// \return The ordinal index of the current tile
      ordinal_type ordinal() const {
        TA_ASSERT(tensor_);
        return *it_;
      }

    }; // class ConstTensorIterator

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
      typedef TensorImpl<Tile> TensorImpl_;
      typedef TiledRange trange_type; ///< Tiled range type
      typedef typename trange_type::range_type range_type; ///< Tile range type
      typedef Bitset<> shape_type; ///< Tensor shape type
      typedef Tile value_type; ///< Tile or data type
      typedef typename Tile::eval_type eval_type; ///< The tile evaluation type
      typedef typename TiledArray::detail::scalar_type<typename value_type::value_type>::type
          numeric_type; ///< the numeric type that supports Tile
      typedef TiledArray::detail::DistributedStorage<value_type> storage_type; ///< The data container type
      typedef typename storage_type::size_type size_type; ///< Size type
      typedef typename storage_type::future future; ///< Future tile type
      typedef TileReference<Tile> reference;
      typedef typename storage_type::future const_reference; ///< Constant reference type
      typedef TensorIterator<Tile> iterator; ///< Iterator type
      typedef ConstTensorIterator<Tile> const_iterator; ///< Constatnt iterator type
      typedef Pmap<size_type> pmap_interface; ///< Process map interface type

    private:

      trange_type trange_; ///< Tiled range type
      shape_type shape_; ///< Tensor shape (zero size == dense)
      storage_type data_; ///< Tile container

    public:

      /// Constructor

      /// The size of shape must be equal to the volume of the tiled range tiles.
      /// \param world The world where this tensor will live
      /// \param trange The tiled range for this tensor
      /// \param shape The shape of this tensor
      /// \throw TiledArray::Exception When the size of shape is not equal to
      /// zero
      TensorImpl(madness::World& world, const trange_type& trange, const shape_type& shape) :
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
      const shape_type& shape() const {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Tensor shape accessor

      /// \return A reference to the tensor shape map
      /// \throw TiledArray::Exception When this tensor is dense
      shape_type& shape() {
        TA_ASSERT(! is_dense());
        return shape_;
      }

      /// Set the shape

      /// \param s The new shape
      /// \throw TiledArray::Exception When the size of \c s is not equal to the
      /// size of this tensor or zero.
      void shape(shape_type s) {
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
        return reference(this, trange_.tiles().ord(i));
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
      const_iterator begin() const {
        TA_ASSERT(data_.get_pmap());
        return iterator(this, data_.get_pmap()->begin());
      }

      /// Array begin iterator

      /// \return A const iterator to the first element of the array.
      iterator begin() {
        TA_ASSERT(data_.get_pmap());
        return iterator(this, data_.get_pmap()->begin());
      }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      const_iterator end() const {
        TA_ASSERT(data_.get_pmap());
        return iterator(this, data_.get_pmap()->end());
      }

      /// Array end iterator

      /// \return A const iterator to one past the last element of the array.
      iterator end() {
        TA_ASSERT(data_.get_pmap());
        return iterator(this, data_.get_pmap()->end());
      }

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
