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
#include <TiledArray/util/function.h>

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
class ArrayImpl : public TensorImpl<Policy>,
                  public std::enable_shared_from_this<ArrayImpl<Tile, Policy>> {
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
  typedef typename Tile::value_type
      element_type;  ///< The value type of a tile. It is the numeric_type for
                     ///< tensor-of-scalars tiles.
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
  static madness::AtomicInt cleanup_counter_;

  /// Array deleter function

  /// This function schedules a task for lazy cleanup. Array objects are
  /// deleted only after the object has been deleted in all processes.
  /// \param pimpl The implementation pointer to be deleted.
  static void lazy_deleter(const ArrayImpl_* const pimpl) {
    if (pimpl) {
      if (madness::initialized()) {
        World& world = pimpl->world();
        const madness::uniqueidT id = pimpl->id();
        cleanup_counter_++;

        // wait for all DelayedSet's to vanish
        world.await([&]() { return (pimpl->num_live_ds() == 0); }, true);

        try {
          world.gop.lazy_sync(id, [pimpl]() {
            delete pimpl;
            ArrayImpl_::cleanup_counter_--;
          });
        } catch (madness::MadnessException& e) {
          fprintf(stderr,
                  "!! ERROR TiledArray: madness::MadnessException thrown in "
                  "DistArray::lazy_deleter().\n"
                  "%s\n"
                  "!! ERROR TiledArray: The exception has been absorbed.\n"
                  "!! ERROR TiledArray: rank=%i\n",
                  e.what(), world.rank());

          cleanup_counter_--;
          delete pimpl;
        } catch (std::exception& e) {
          fprintf(stderr,
                  "!! ERROR TiledArray: std::exception thrown in "
                  "DistArray::lazy_deleter().\n"
                  "%s\n"
                  "!! ERROR TiledArray: The exception has been absorbed.\n"
                  "!! ERROR TiledArray: rank=%i\n",
                  e.what(), world.rank());

          cleanup_counter_--;
          delete pimpl;
        } catch (...) {
          fprintf(stderr,
                  "!! ERROR TiledArray: An unknown exception was thrown in "
                  "DistArray::lazy_deleter().\n"
                  "!! ERROR TiledArray: The exception has been absorbed.\n"
                  "!! ERROR TiledArray: rank=%i\n",
                  world.rank());

          cleanup_counter_--;
          delete pimpl;
        }
      } else {
        delete pimpl;
      }
    }
  }

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
            const std::shared_ptr<const pmap_interface>& pmap)
      : TensorImpl_(world, trange, shape, pmap),
        data_(world, trange.tiles_range().volume(), pmap) {
    // Validate the process map
    TA_ASSERT(pmap->size() == trange.tiles_range().volume() &&
              "TiledArray::DistArray::DistArray() -- The size of the process "
              "map is not "
              "equal to the number of tiles in the TiledRange object.");
    TA_ASSERT(pmap->rank() ==
                  typename pmap_interface::size_type(world.rank()) &&
              "TiledArray::DistArray::DistArray() -- The rank of the process "
              "map is not equal to that "
              "of the world object.");
    TA_ASSERT(pmap->procs() ==
                  typename pmap_interface::size_type(world.size()) &&
              "TiledArray::DistArray::DistArray() -- The number of processes "
              "in the process map is not "
              "equal to that of the world object.");

    // Validate the shape
    TA_ASSERT(
        !shape.empty() &&
        "TiledArray::DistArray::DistArray() -- The shape is not initialized.");
    TA_ASSERT(shape.validate(trange.tiles_range()) &&
              "TiledArray::DistArray::DistArray() -- The range of the shape is "
              "not equal to "
              "the tiles range.");
  }

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
  const std::atomic<std::size_t>& num_live_ds() const {
    return data_.num_live_ds();
  }

  /// Reports the number of live DelayedForward requests for this object's
  /// DistributedStorage

  /// @return const reference to the atomic counter of live DelayedForward
  /// requests
  const std::atomic<std::size_t>& num_live_df() const {
    return data_.num_live_df();
  }

  /// Initialize (local) tiles with a user provided functor

  /// This function is used to initialize the local, non-zero tiles of the array
  /// via a function (or functor). The work is done in parallel, therefore \c op
  /// must be a thread safe function/functor. The signature of the functor
  /// should be:
  /// \code
  /// value_type op(const range_type&)
  /// \endcode
  /// For example, in the following code, the array tiles are initialized with
  /// random numbers from 0 to 1:
  /// \code
  /// array.init_tiles([] (const TiledArray::Range& range) ->
  /// TiledArray::Tensor<double>
  ///     {
  ///        // Initialize the tile with the given range object
  ///        TiledArray::Tensor<double> tile(range);
  ///
  ///        // Initialize the random number generator
  ///        std::default_random_engine generator;
  ///        std::uniform_real_distribution<double> distribution(0.0,1.0);
  ///
  ///        // Fill the tile with random numbers
  ///        for(auto& value : tile)
  ///           value = distribution(generator);
  ///
  ///        return tile;
  ///     });
  /// \endcode
  /// \tparam Op The type of the functor/function
  /// \param[in] op The operation used to generate tiles
  /// \param[in] skip_set If false, will throw if any tiles are already set
  /// \throw TiledArray::Exception if the PIMPL is not set. Strong throw
  ///                              guarantee.
  /// \throw TiledArray::Exception if a tile is already set and skip_set is
  ///                              false. Weak throw guarantee.
  template <HostExecutor Exec = HostExecutor::Default, typename Op>
  void init_tiles(Op&& op, bool skip_set = false) {
    // lifetime management of op depends on whether it is a lvalue ref (i.e. has
    // an external owner) or an rvalue ref
    // - if op is an lvalue ref: pass op to tasks
    // - if op is an rvalue ref pass make_shared_function(op) to tasks
    auto op_shared_handle = make_op_shared_handle(std::forward<Op>(op));

    auto it = this->pmap()->begin();
    const auto end = this->pmap()->end();
    for (; it != end; ++it) {
      const auto& index = *it;
      if (!this->is_zero(index)) {
        if (skip_set) {
          auto& fut = this->get_local(index);
          if (fut.probe()) continue;
        }
        if constexpr (Exec == HostExecutor::MADWorld) {
          Future<value_type> tile = this->world().taskq.add(
              [this_sptr = this->shared_from_this(),
               index = ordinal_type(index), op_shared_handle]() -> value_type {
                return op_shared_handle(
                    this_sptr->trange().make_tile_range(index));
              });
          set(index, std::move(tile));
        } else {
          static_assert(Exec == HostExecutor::Thread);
          set(index, op_shared_handle(this->trange().make_tile_range(index)));
        }
      }
    }
  }

};  // class ArrayImpl

template <typename Tile, typename Policy>
madness::AtomicInt ArrayImpl<Tile, Policy>::cleanup_counter_;

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

template <typename Tile, typename Policy>
void write_tile_block(madness::uniqueidT target_array_id,
                      std::size_t target_tile_ord,
                      const Tile& target_tile_contribution) {
  auto* world_ptr = World::world_from_id(target_array_id.get_world_id());
  auto target_array_ptr_opt =
      world_ptr->ptr_from_id<typename ArrayImpl<Tile, Policy>::storage_type>(
          target_array_id);
  TA_ASSERT(target_array_ptr_opt);
  TA_ASSERT((*target_array_ptr_opt)->is_local(target_tile_ord));
  (*target_array_ptr_opt)
      ->get_local(target_tile_ord)
      .get()
      .block(target_tile_contribution.range()) = target_tile_contribution;
}

template <typename Tile, typename Policy>
std::shared_ptr<ArrayImpl<Tile, Policy>> make_with_new_trange(
    const std::shared_ptr<const ArrayImpl<Tile, Policy>>& source_array_sptr,
    const TiledRange& target_trange,
    typename ArrayImpl<Tile, Policy>::element_type new_value_fill =
        typename ArrayImpl<Tile, Policy>::element_type{}) {
  TA_ASSERT(source_array_sptr);
  auto& source_array = *source_array_sptr;
  auto& world = source_array.world();
  const auto rank = source_array.trange().rank();
  TA_ASSERT(rank == target_trange.rank());

  // compute metadata
  // - list of target tile indices and the corresponding Range1 for each 1-d
  // source tile
  using target_tiles_t = std::vector<std::pair<TA_1INDEX_TYPE, Range1>>;
  using mode_target_tiles_t = std::vector<target_tiles_t>;
  using all_target_tiles_t = std::vector<mode_target_tiles_t>;

  all_target_tiles_t all_target_tiles(target_trange.rank());
  // for each mode ...
  for (auto d = 0; d != target_trange.rank(); ++d) {
    mode_target_tiles_t& mode_target_tiles = all_target_tiles[d];
    auto& target_tr1 = target_trange.dim(d);
    auto& target_element_range = target_tr1.elements_range();
    // ... and each tile in that mode ...
    for (auto&& source_tile : source_array.trange().dim(d)) {
      mode_target_tiles.emplace_back();
      auto& target_tiles = mode_target_tiles.back();
      auto source_tile_lo = source_tile.lobound();
      auto source_tile_up = source_tile.upbound();
      auto source_element_idx = source_tile_lo;
      // ... find all target tiles what overlap with it
      if (target_element_range.overlaps_with(source_tile)) {
        while (source_element_idx < source_tile_up) {
          if (target_element_range.includes(source_element_idx)) {
            auto target_tile_idx =
                target_tr1.element_to_tile(source_element_idx);
            auto target_tile = target_tr1.tile(target_tile_idx);
            auto target_lo =
                std::max(source_element_idx, target_tile.lobound());
            auto target_up = std::min(source_tile_up, target_tile.upbound());
            target_tiles.emplace_back(target_tile_idx,
                                      Range1(target_lo, target_up));
            source_element_idx = target_up;
          } else if (source_element_idx < target_element_range.lobound()) {
            source_element_idx = target_element_range.lobound();
          } else if (source_element_idx >= target_element_range.upbound())
            break;
        }
      }
    }
  }

  // estimate the shape, if sparse
  // use max value for each nonzero tile, then will recompute after tiles are
  // assigned
  using shape_type = typename Policy::shape_type;
  shape_type target_shape;
  const auto& target_tiles_range = target_trange.tiles_range();
  if constexpr (!is_dense_v<Policy>) {
    // each rank computes contributions to the shape norms from its local tiles
    Tensor<float> target_shape_norms(target_tiles_range, 0);
    auto& source_trange = source_array.trange();
    const auto e = source_array.cend();
    for (auto it = source_array.cbegin(); it != e; ++it) {
      auto source_tile_idx = it.index();

      // make range for iterating over all possible target tile idx combinations
      TA::Index target_tile_ord_extent_range(rank);
      for (auto d = 0; d != rank; ++d) {
        target_tile_ord_extent_range[d] =
            all_target_tiles[d][source_tile_idx[d]].size();
      }

      // loop over every target tile combination
      TA::Range target_tile_ord_extent(target_tile_ord_extent_range);
      for (auto& target_tile_ord : target_tile_ord_extent) {
        TA::Index target_tile_idx(rank);
        for (auto d = 0; d != rank; ++d) {
          target_tile_idx[d] =
              all_target_tiles[d][source_tile_idx[d]][target_tile_ord[d]].first;
        }
        target_shape_norms(target_tile_idx) = std::numeric_limits<float>::max();
      }
    }
    world.gop.max(target_shape_norms.data(), target_shape_norms.size());
    target_shape = SparseShape(target_shape_norms, target_trange);
  }

  using Array = ArrayImpl<Tile, Policy>;
  auto target_array_sptr = std::shared_ptr<Array>(
      new Array(
          source_array.world(), target_trange, target_shape,
          Policy::default_pmap(world, target_trange.tiles_range().volume())),
      Array::lazy_deleter);
  auto& target_array = *target_array_sptr;
  target_array.init_tiles([value = new_value_fill](const Range& range) {
    return typename Array::value_type(range, value);
  });
  target_array.world().gop.fence();

  // loop over local tile and sends its contributions to the targets
  {
    auto& source_trange = source_array.trange();
    const auto e = source_array.cend();
    auto& target_tiles_range = target_trange.tiles_range();
    for (auto it = source_array.cbegin(); it != e; ++it) {
      const auto& source_tile = *it;
      auto source_tile_idx = it.index();

      // make range for iterating over all possible target tile idx combinations
      TA::Index target_tile_ord_extent_range(rank);
      for (auto d = 0; d != rank; ++d) {
        target_tile_ord_extent_range[d] =
            all_target_tiles[d][source_tile_idx[d]].size();
      }

      // loop over every target tile combination
      TA::Range target_tile_ord_extent(target_tile_ord_extent_range);
      for (auto& target_tile_ord : target_tile_ord_extent) {
        TA::Index target_tile_idx(rank);
        container::svector<TA::Range1> target_tile_rngs1(rank);
        for (auto d = 0; d != rank; ++d) {
          std::tie(target_tile_idx[d], target_tile_rngs1[d]) =
              all_target_tiles[d][source_tile_idx[d]][target_tile_ord[d]];
        }
        TA_ASSERT(source_tile.future().probe());
        Tile target_tile_contribution(
            source_tile.get().block(target_tile_rngs1));
        auto target_tile_idx_ord = target_tiles_range.ordinal(target_tile_idx);
        auto target_proc = target_array.pmap()->owner(target_tile_idx_ord);
        world.taskq.add(target_proc, &write_tile_block<Tile, Policy>,
                        target_array.id(), target_tile_idx_ord,
                        target_tile_contribution);
      }
    }
  }
  // data is mutated in place, so must wait for all tasks to complete
  target_array.world().gop.fence();
  // WARNING!! need to truncate in DistArray ctor

  return target_array_sptr;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_ARRAY_IMPL_H__INCLUDED
