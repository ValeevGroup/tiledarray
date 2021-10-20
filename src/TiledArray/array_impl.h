/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  array_impl.h
 *  Oct 24, 2014
 *
 */

#ifndef TILEDARRAY_ARRAY_IMPL_H__INCLUDED
#define TILEDARRAY_ARRAY_IMPL_H__INCLUDED

#include <TiledArray/distributed_storage.h>
#include <TiledArray/tensor_impl.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
namespace detail {

// Forward declaration
template <typename>
class TensorReference;
template <typename>
class TensorConstReference;
template <typename, typename>
class ArrayIterator;

/// Tensor tile reference

/// \tparam Impl The TensorImpl type
template <typename Impl>
class TileReference {
 private:
  template <typename, typename>
  friend class ArrayIterator;

  template <typename>
  friend class TileConstReference;

  typedef typename Impl::range_type range_type;
  typedef typename Impl::range_type::index index_type;
  typedef typename Impl::ordinal_type ordinal_type;

  Impl* tensor_;        ///< The tensor that owns the referenced tile
  ordinal_type index_;  ///< The ordinal index of the tile

  // Not allowed
  TileReference<Impl>& operator=(const TileReference<Impl>&);

 public:
  TileReference(Impl* tensor, const typename Impl::ordinal_type index)
      : tensor_(tensor), index_(index) {}

  TileReference(const TileReference<Impl>& other)
      : tensor_(other.tensor_), index_(other.index_) {}

  template <typename Value>
  TileReference<Impl>& operator=(const Value& value) {
    tensor_->set(index_, value);
    return *this;
  }

  typename Impl::future future() const {
    TA_ASSERT(tensor_);
    return tensor_->get(index_);
  }

  typename Impl::value_type get() const {
    // NOTE: return by value to avoid lifetime issues.
    TA_ASSERT(tensor_);
    return future().get();
  }

  operator typename Impl::future() const { return this->future(); }

  operator typename Impl::value_type() const { return get(); }

  /// Tile coordinate index accessor

  /// \return The coordinate index of the current tile
  index_type index() const {
    TA_ASSERT(tensor_);
    return tensor_->tiles_range().idx(index_);
  }

  /// Tile ordinal index accessor

  /// \return The ordinal index of the current tile
  ordinal_type ordinal() const { return index_; }

  /// Tile range factory function

  /// Construct a range object for the current tile
  range_type make_range() const {
    TA_ASSERT(tensor_);
    return tensor_->trange().make_tile_range(index_);
  }

};  // class TileReference

/// comparison operator for TileReference objects
template <typename Impl>
bool operator==(const TileReference<Impl>& a, const TileReference<Impl>& b) {
  return a.get() == b.get();
}

/// inequality operator for TileReference objects
template <typename Impl>
bool operator!=(const TileReference<Impl>& a, const TileReference<Impl>& b) {
  return !(a == b);
}

/// redirect operator to std::ostream for TileReference objects
template <typename Impl>
std::ostream& operator<<(std::ostream& os, const TileReference<Impl>& a) {
  os << a.get();
  return os;
}

/// Tensor tile reference

/// \tparam Impl The TensorImpl type
template <typename Impl>
class TileConstReference {
 private:
  template <typename, typename>
  friend class ArrayIterator;

  const Impl* tensor_;  ///< The tensor that owns the referenced tile
  typename Impl::ordinal_type index_;  ///< The ordinal index of the tile

  // Not allowed
  TileConstReference<Impl>& operator=(const TileConstReference<Impl>&);

 public:
  TileConstReference(const Impl* tensor,
                     const typename Impl::ordinal_type index)
      : tensor_(tensor), index_(index) {}

  TileConstReference(const TileConstReference<Impl>& other)
      : tensor_(other.tensor_), index_(other.index_) {}

  TileConstReference(const TileReference<Impl>& other)
      : tensor_(other.tensor_), index_(other.index_) {}

  typename Impl::future future() const {
    TA_ASSERT(tensor_);
    return tensor_->get(index_);
  }

  typename Impl::value_type get() const {
    // NOTE: return by value to avoid lifetime issues.
    TA_ASSERT(tensor_);
    return future().get();
  }

  operator typename Impl::future() const { return tensor_->get(index_); }

  operator typename Impl::value_type() const { return get(); }
};  // class TileConstReference

/// comparison operator for TileConstReference objects
template <typename Impl>
bool operator==(const TileConstReference<Impl>& a,
                const TileConstReference<Impl>& b) {
  return a.get() == b.get();
}

/// inequality operator for TileConstReference objects
template <typename Impl>
bool operator!=(const TileConstReference<Impl>& a,
                const TileConstReference<Impl>& b) {
  return !(a == b);
}

/// redirect operator to std::ostream for TileConstReference objects
template <typename Impl>
std::ostream& operator<<(std::ostream& os, const TileConstReference<Impl>& a) {
  os << a.get();
  return os;
}

}  // namespace detail
}  // namespace TiledArray

namespace madness {
namespace detail {

// The following class specializations are required so MADNESS will do the
// right thing when given a TileReference or TileConstReference object
// as an input for task functions.

template <typename Impl>
struct task_arg<TiledArray::detail::TileReference<Impl>> {
  typedef typename Impl::value_type type;
  typedef typename Impl::future holderT;
};  // struct task_arg<TiledArray::detail::TileReference<Impl> >

template <typename Impl>
struct task_arg<TiledArray::detail::TileConstReference<Impl>> {
  typedef typename Impl::value_type type;
  typedef typename Impl::future holderT;
};  // struct task_arg<TiledArray::detail::TileConstReference<Impl> >

}  // namespace detail
}  // namespace madness

namespace TiledArray {
namespace detail {

/// Distributed tensor iterator

/// This iterator will reference local tiles for a TensorImpl object. It can
/// be used to get or set futures to a tile, or access the coordinate and
/// ordinal index of the tile.
/// \tparam Impl The TensorImpl type
/// \tparam Reference The iterator reference type
template <typename Impl, typename Reference>
class ArrayIterator {
 private:
  // Give access to other iterator types.
  template <typename, typename>
  friend class ArrayIterator;

  Impl* array_;
  typename Impl::pmap_interface::const_iterator it_;

 public:
  typedef ptrdiff_t difference_type;  ///< Difference type
  typedef
      typename Impl::future value_type;  ///< Iterator dereference value type
  typedef PointerProxy<value_type> pointer;  ///< Pointer type to iterator value
  typedef Reference reference;  ///< Reference type to iterator value
  typedef std::forward_iterator_tag
      iterator_category;  ///< Iterator category type
  typedef ArrayIterator<Impl, Reference> ArrayIterator_;  ///< This object type
  typedef typename Impl::range_type::index index_type;
  typedef typename Impl::ordinal_type ordinal_type;
  typedef typename Impl::range_type range_type;
  typedef typename Impl::value_type tile_type;

 private:
  void advance() {
    TA_ASSERT(array_);
    const typename Impl::pmap_interface::const_iterator end =
        array_->pmap()->end();
    do {
      ++it_;
    } while ((it_ != end) && array_->is_zero(*it_));
  }

 public:
  /// Default constructor
  ArrayIterator() : array_(NULL), it_() {}

  /// Constructor
  ArrayIterator(Impl* tensor, typename Impl::pmap_interface::const_iterator it)
      : array_(tensor), it_(it) {}

  /// Copy constructor

  /// \param other The transform iterator to copy
  ArrayIterator(const ArrayIterator_& other)
      : array_(other.array_), it_(other.it_) {}

  /// Copy const iterator constructor

  /// \tparam R Iterator reference type
  /// \param other The transform iterator to copy
  template <
      typename I, typename R,
      typename std::enable_if<!((!std::is_const<Impl>::value) &&
                                std::is_const<I>::value)>::type* = nullptr>
  ArrayIterator(const ArrayIterator<I, R>& other)
      : array_(other.array_), it_(other.it_) {}

  /// Copy operator

  /// \param other The transform iterator to copy
  /// \return A reference to this object
  ArrayIterator_& operator=(const ArrayIterator_& other) {
    array_ = other.array_;
    it_ = other.it_;

    return *this;
  }

  /// Copy operator

  /// \tparam R Iterator reference type
  /// \param other The transform iterator to copy
  /// \return A reference to this object
  template <typename R>
  ArrayIterator_& operator=(const ArrayIterator<Impl, R>& other) {
    array_ = other.array_;
    it_ = other.it_;

    return *this;
  }

  /// Prefix increment operator

  /// \return A reference to this object after it has been incremented.
  ArrayIterator_& operator++() {
    advance();
    return *this;
  }

  /// Post-fix increment operator

  /// \return A copy of this object before it is incremented.
  ArrayIterator_ operator++(int) {
    ArrayIterator_ tmp(*this);
    advance();
    return tmp;
  }

  /// Equality operator

  /// \tparam R Iterator reference type
  /// \param other The iterator to compare to this iterator.
  /// \return \c true when the iterators are equal to each other, otherwise
  /// \c false.
  template <typename I, typename R>
  bool operator==(const ArrayIterator<I, R>& other) const {
    return (array_ == other.array_) && (it_ == other.it_);
  }

  /// Inequality operator

  /// \tparam R Iterator reference type
  /// \param other The iterator to compare to this iterator.
  /// \return \c true when the iterators are not equal to each other,
  /// otherwise \c false.
  template <typename I, typename R>
  bool operator!=(const ArrayIterator<I, R>& other) const {
    return (array_ != other.array_) || (it_ != other.it_);
  }

  /// Dereference operator

  /// \return A reference to the current tile future.
  reference operator*() const {
    TA_ASSERT(array_);
    return reference(array_, *it_);
  }

  /// Arrow dereference operator

  /// \return A pointer-proxy to the current tile
  pointer operator->() const {
    TA_ASSERT(array_);
    return pointer(array_->get(*it_));
  }

  /// Tile coordinate index accessor

  /// \return The coordinate index of the current tile
  index_type index() const {
    TA_ASSERT(array_);
    return array_->tiles_range().idx(*it_);
  }

  /// Tile ordinal index accessor

  /// \return The ordinal index of the current tile
  ordinal_type ordinal() const {
    TA_ASSERT(array_);
    return *it_;
  }

  /// Tile range factory function

  /// Construct a range object for the current tile
  range_type make_range() const {
    TA_ASSERT(array_);
    TA_ASSERT(it_ != array_->pmap()->end());
    return array_->trange().make_tile_range(*it_);
  }

};  // class TensorIterator

/// Tensor implementation and base for other tensor implementation objects

/// This implementation object holds the data for tensor object, which
/// includes tiled range, shape, and tiles. The tiles are held in a
/// distributed container, stored according to a given process map.
/// \tparam Tile The tile or value_type of this tensor
/// \note The process map must be set before data elements can be set.
/// \note It is the users responsibility to ensure the process maps on all
/// nodes are identical.
template <typename Tile, typename Policy>
class ArrayImpl : public TensorImpl<Policy> {
 public:
  typedef ArrayImpl<Tile, Policy> ArrayImpl_;  ///< This object type
  typedef TensorImpl<Policy> TensorImpl_;  ///< The base class of this object
  typedef typename TensorImpl_::index1_type index1_type;    ///< 1-index type
  typedef typename TensorImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename TensorImpl_::policy_type
      policy_type;  ///< Policy type for this object
  typedef typename TensorImpl_::trange_type
      trange_type;  ///< Tiled range type for this object
  typedef typename TensorImpl_::range_type
      range_type;  ///< Elements/tiles range type
  typedef typename TensorImpl_::shape_type shape_type;  ///< Shape type
  typedef typename TensorImpl_::pmap_interface
      pmap_interface;       ///< process map interface type
  typedef Tile value_type;  ///< Tile or data type
  typedef
      typename eval_trait<Tile>::type eval_type;  ///< The tile evaluation type
  typedef typename numeric_type<value_type>::type
      numeric_type;  ///< the numeric type that supports Tile
  typedef DistributedStorage<value_type>
      storage_type;                              ///< The data container type
  typedef typename storage_type::future future;  ///< Future tile type
  typedef TileReference<ArrayImpl_> reference;   ///< Tile reference type
  typedef TileConstReference<ArrayImpl_>
      const_reference;  ///< Tile constant reference type
  typedef ArrayIterator<ArrayImpl_, reference> iterator;  ///< Iterator type
  typedef ArrayIterator<const ArrayImpl_, const_reference>
      const_iterator;  ///< Constant iterator type

 private:
  storage_type data_;  ///< Tile container

 public:
  /// Constructor

  /// The size of shape must be equal to the volume of the tiled range tiles.
  /// \param world The world where this tensor will live
  /// \param trange The tiled range for this tensor
  /// \param shape The shape of this tensor
  /// \param pmap The tile-process map
  /// \throw TiledArray::Exception When the size of shape is not equal to
  /// zero
  ArrayImpl(World& world, const trange_type& trange, const shape_type& shape,
            const std::shared_ptr<pmap_interface>& pmap)
      : TensorImpl_(world, trange, shape, pmap),
        data_(world, trange.tiles_range().volume(), pmap) {}

  /// Virtual destructor
  virtual ~ArrayImpl() {}

  /// Tile future accessor

  /// \tparam Index An integral or integral range type
  /// \param i The tile index or ordinal
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index> ||
                                        detail::is_integral_range_v<Index>>>
  future get(const Index& i) const {
    TA_ASSERT(!TensorImpl_::is_zero(i));
    return data_.get(TensorImpl_::trange().tiles_range().ordinal(i));
  }

  /// Tile future accessor

  /// \tparam Integer An integral type
  /// \param i The tile index, as an \c std::initializer_list<Integer>
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  future get(const std::initializer_list<Integer>& i) const {
    return get<std::initializer_list<Integer>>(i);
  }

  /// Local tile future accessor

  /// \tparam Index An integral or integral range type
  /// \param i The tile index or ordinal
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index> ||
                                        detail::is_integral_range_v<Index>>>
  const future& get_local(const Index& i) const {
    TA_ASSERT(!TensorImpl_::is_zero(i) && TensorImpl_::is_local(i));
    return data_.get_local(TensorImpl_::trange().tiles_range().ordinal(i));
  }

  /// Local tile future accessor

  /// \tparam Integer An integral type
  /// \param i The tile index, as an \c std::initializer_list<Integer>
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  const future& get_local(const std::initializer_list<Integer>& i) const {
    return get_local<std::initializer_list<Integer>>(i);
  }

  /// Local tile future accessor

  /// \tparam Index An integral or integral range type
  /// \param i The tile index or ordinal
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index> ||
                                        detail::is_integral_range_v<Index>>>
  future& get_local(const Index& i) {
    TA_ASSERT(!TensorImpl_::is_zero(i) && TensorImpl_::is_local(i));
    return data_.get_local(TensorImpl_::trange().tiles_range().ordinal(i));
  }

  /// Local tile future accessor

  /// \tparam Integer An integral type
  /// \param i The tile index, as an \c std::initializer_list<Integer>
  /// \return A \c future to tile \c i
  /// \throw TiledArray::Exception When tile \c i is zero or not local
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  future& get_local(const std::initializer_list<Integer>& i) {
    return get_local<std::initializer_list<Integer>>(i);
  }

  /// Set tile

  /// Set the tile at \c i with \c value . \c Value type may be \c value_type ,
  /// \c Future<value_type> , or
  /// \c madness::detail::MoveWrapper<value_type> .
  /// \tparam Index An integral or integral range type
  /// \tparam Value The value type
  /// \param i The index of the tile to be set
  /// \param value The object tat contains the tile value
  template <typename Index, typename Value,
            typename = std::enable_if_t<std::is_integral_v<Index> ||
                                        detail::is_integral_range_v<Index>>>
  void set(const Index& i, Value&& value) {
    TA_ASSERT(!TensorImpl_::is_zero(i));
    const auto ord = TensorImpl_::trange().tiles_range().ordinal(i);
    data_.set(ord, std::forward<Value>(value));
    if (set_notifier_accessor()) {
      set_notifier_accessor()(*this, ord);
    }
  }

  /// Set tile

  /// Set the tile at \c i with \c value . \c Value type may be \c value_type ,
  /// \c Future<value_type> , or
  /// \c madness::detail::MoveWrapper<value_type> .
  /// \tparam Index An integral type
  /// \tparam Value The value type
  /// \param i The index of the tile to be set
  /// \param value The object tat contains the tile value
  template <typename Index, typename Value,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  void set(const std::initializer_list<Index>& i, Value&& value) {
    TA_ASSERT(!TensorImpl_::is_zero(i));
    const auto ord = TensorImpl_::trange().tiles_range().ordinal(i);
    data_.set(ord, std::forward<Value>(value));
    if (set_notifier_accessor()) {
      set_notifier_accessor()(*this, ord);
    }
  }

  /// Array begin iterator

  /// \return A const iterator to the first local element of the array.
  iterator begin() {
    // Get the pmap iterator
    typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();

    // Find the first non-zero iterator
    const typename pmap_interface::const_iterator end =
        TensorImpl_::pmap()->end();
    while ((it != end) && TensorImpl_::is_zero(*it)) ++it;

    // Construct and return the iterator
    return iterator(this, it);
  }

  /// Array begin iterator

  /// \return A const iterator to the first local element of the array.
  const_iterator cbegin() const {
    // Get the pmap iterator
    typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();

    // Find the fist non-zero iterator
    const typename pmap_interface::const_iterator end =
        TensorImpl_::pmap()->end();
    while ((it != end) && TensorImpl_::is_zero(*it)) ++it;

    // Construct and return the iterator
    return const_iterator(this, it);
  }

  /// Array end iterator

  /// \return A const iterator to one past the last local element of the array.
  iterator end() { return iterator(this, TensorImpl_::pmap()->end()); }

  /// Array end iterator

  /// \return A const iterator to one past the last local element of the array.
  const_iterator cend() const {
    return const_iterator(this, TensorImpl_::pmap()->end());
  }

  /// Unique object id accessor

  /// \return A const reference to this object unique id
  const madness::uniqueidT& id() const { return data_.id(); }

  static std::function<void(const ArrayImpl_&, int64_t)>&
  set_notifier_accessor() {
    static std::function<void(const ArrayImpl_&, int64_t)> value;
    return value;
  }

  /// Reports the number of live DelayedSet requests for this object's
  /// DistributedStorage

  /// @return const reference to the atomic counter of live DelayedSet requests
  const madness::AtomicInt& num_live_ds() const {
    return data_.num_live_ds();
  }

};  // class ArrayImpl

#ifndef TILEDARRAY_HEADER_ONLY

extern template class ArrayImpl<Tensor<double>, DensePolicy>;
extern template class ArrayImpl<Tensor<float>, DensePolicy>;
// extern template class ArrayImpl<Tensor<int>,
//                                DensePolicy>;
// extern template class ArrayImpl<Tensor<long>,
//                                DensePolicy>;
extern template class ArrayImpl<Tensor<std::complex<double>>, DensePolicy>;
extern template class ArrayImpl<Tensor<std::complex<float>>, DensePolicy>;

extern template class ArrayImpl<Tensor<double>, SparsePolicy>;
extern template class ArrayImpl<Tensor<float>, SparsePolicy>;
// extern template class ArrayImpl<Tensor<int>,
//                                SparsePolicy>;
// extern template class ArrayImpl<Tensor<long>,
//                                SparsePolicy>;
extern template class ArrayImpl<Tensor<std::complex<double>>, SparsePolicy>;
extern template class ArrayImpl<Tensor<std::complex<float>>, SparsePolicy>;

#endif  // TILEDARRAY_HEADER_ONLY

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
