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

#ifndef TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
#define TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {

/// Describes how to get remote data
enum class RemoteDataGetPolicy {
  /// no caching = each get will trigger data fetch
  nocache,
  /// aggregate gets until data arrives, subsequent gets will trigger new gets
  aggregate,
  /// get once, read forever
  cache
};

namespace detail {

/// Distributed storage container.

/// Each element in this container is owned by a single node, but any node
/// may request a copy of the element in the form of a \c Future .
/// The owner of each element is defined by a process map (pmap), which is
/// passed to the constructor. Elements do not need to be explicitly
/// initialized because they will be added to the container when the element
/// is first accessed, though you may manually initialize an element with
/// the \c insert() function. All elements are stored in \c Future ,
/// which may be set only once.
/// \note This object is derived from \c WorldObject , which means
/// the order of construction of object must be the same on all nodes. This
/// can easily be achieved by only constructing world objects in the main
/// thread. DO NOT construct world objects within tasks where the order of
/// execution is nondeterministic.
template <typename T>
class DistributedStorage : public madness::WorldObject<DistributedStorage<T>> {
 public:
  typedef DistributedStorage<T> DistributedStorage_;  ///< This object type
  typedef madness::WorldObject<DistributedStorage_>
      WorldObject_;  ///< Base object type

  typedef std::size_t size_type;      ///< size type
  typedef size_type key_type;         ///< element key type
  typedef T value_type;               ///< Element type
  typedef Future<value_type> future;  ///< Element container type
  typedef Pmap pmap_interface;        ///< Process map interface type
  typedef madness::ConcurrentHashMap<key_type, future>
      container_type;  ///< Local container type
  typedef typename container_type::accessor
      accessor;  ///< Local element accessor type
  typedef typename container_type::const_accessor
      const_accessor;  ///< Local element const accessor type

 private:
  const size_type max_size_;  ///< The maximum number of elements that can be
                              ///< stored by this container
  std::shared_ptr<const pmap_interface>
      pmap_;  ///< The process map that defines the element distribution
  mutable container_type data_;  ///< The local data container

  // tracing/defensive driving artifacts
  mutable std::atomic<std::size_t>
      num_live_ds_;  ///< Number of live DelayedSet objects
  mutable std::atomic<std::size_t>
      num_live_df_;  ///< Number of live DelayedForward objects
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  mutable std::vector<std::atomic<std::size_t>>
      ngets_served_per_rank_;  ///< Counts # of gets served to remote ranks
  mutable std::vector<std::atomic<std::size_t>>
      ngets_sent_per_rank_;  ///< Counts # of gets sent to remote ranks
  mutable std::vector<std::atomic<std::size_t>>
      ngets_received_per_rank_;  ///< Counts # of gets received from remote
                                 ///< ranks
#endif

  // not allowed
  DistributedStorage(const DistributedStorage_&);
  DistributedStorage_& operator=(const DistributedStorage_&);

  template <typename Value>
  std::enable_if_t<std::is_same_v<std::decay_t<Value>, value_type>, void>
  set_handler(const size_type i, Value&& value) {
    future& f = get_local(i);

    // Check that the future has not been set already.
    TA_ASSERT(!f.probe() && "Tile has already been assigned.");

    f.set(std::forward<Value>(value));
  }

  void get_handler(const size_type i,
                   const typename future::remote_refT& ref) const {
    const future& f = get_local(i);
    future remote_f(ref);
    remote_f.set(f);
  }

  template <typename Value>
  std::enable_if_t<std::is_same_v<std::decay_t<Value>, value_type> ||
                       std::is_same_v<std::decay_t<Value>, future>,
                   void>
  set_remote(const size_type i, Value&& value) {
    WorldObject_::task(
        owner(i), &DistributedStorage_::set_handler<std::decay_t<value_type>&>,
        i, std::forward<Value>(value),
        madness::TaskAttributes::hipri_unordered());
  }

  struct DelayedSet : public madness::CallbackInterface {
   private:
    DistributedStorage_& ds_;  ///< A reference to the owning object
    size_type index_;          ///< The index that will own the future
    future future_;            ///< The future that we are waiting on.

   public:
    DelayedSet(DistributedStorage_& ds, size_type i, const future& f)
        : ds_(ds), index_(i), future_(f) {
      ++ds_.num_live_ds_;
    }

    virtual ~DelayedSet() { --ds_.num_live_ds_; }

    virtual void notify() {
      ds_.set_remote(index_, future_);
      delete this;
    }
  };  // struct DelayedSet
  friend struct DelayedSet;

  /// Tile cache works just like madness::detail::DistCache (and in fact is
  /// based on it) in that it implements a local cache for asynchronous data
  /// pulls. Unlike madness::detail::DistCache:
  /// - this is unidirectional, i.e. there is no need to manually push data into
  /// the cache (a task sending data
  ///   will be posted).
  /// - depending on get policy data will either stay in the cache forever or
  /// will be discarded upon arrival;
  ///   subsequent gets will need to fetch the data again (may make this
  ///   user-controllable in the future)
  mutable container_type remote_data_cache_;

  /// Get the cache value accosted with \c key

  /// This will get the value associated with \c key to \c value. If
  /// the cache element does not exist, a task requesting the data will be sent
  /// to the owner, a future referring to the result will be inserted in the
  /// cache so that the subsequent gets will receive the same data. After data
  /// arrival the future will be removed from the cache, thus subsequent gets
  /// will need to fetch the data again. \param[in] key The target key \return A
  /// future that holds/will hold the cache value
  future get_cached(const key_type& key, bool keep_in_cache = false) const {
    // Retrieve the cached future
    typename container_type::const_accessor acc;
    if (remote_data_cache_.insert(
            acc, key)) {  // no future in cache yet, create a task
      static_assert(std::is_signed_v<ProcessID>);
      const ProcessID rank = this->get_world().rank();
      ProcessID rank_w_persistence = keep_in_cache ? rank : -(rank + 1);
      WorldObject_::task(owner(key), &DistributedStorage_::get_cached_handler,
                         key, rank_w_persistence,
                         madness::TaskAttributes::hipri_unordered());
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
      ngets_sent_per_rank_.at(owner(key))++;
#endif
    }
    return acc->second;
  }

  /// used to forward data that were unassigned at the time of request arrival
  struct DelayedForward : public madness::CallbackInterface {
   public:
    DelayedForward(const DistributedStorage_& ds, key_type key,
                   ProcessID destination_rank, bool keep_in_cache)
        : ds(ds),
          key(key),
          destination_rank(destination_rank),
          keep_in_cache(keep_in_cache) {}

    void notify() override {
      auto& data_fut = ds.get_local(key);
      TA_ASSERT(
          data_fut.probe());  // must be ready, otherwise why is this invoked?
      if (keep_in_cache) {
        ds.task(destination_rank,
                &DistributedStorage_::template set_cached_handler<true>, key,
                data_fut, madness::TaskAttributes::hipri());
      } else {
        ds.task(destination_rank,
                &DistributedStorage_::template set_cached_handler<false>, key,
                data_fut, madness::TaskAttributes::hipri());
      }
      delete this;
    }

   private:
    const DistributedStorage_& ds;
    key_type key;
    ProcessID destination_rank;
    bool keep_in_cache;
  };

  void get_cached_handler(const size_type key,
                          ProcessID destination_rank_w_persistence) const {
    const bool keep_in_cache = destination_rank_w_persistence >= 0;
    const ProcessID destination_rank =
        destination_rank_w_persistence < 0
            ? (-destination_rank_w_persistence - 1)
            : destination_rank_w_persistence;
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    ngets_served_per_rank_.at(destination_rank)++;
#endif
    auto& data_fut = get_local(key);
    if (data_fut.probe()) {
      if (keep_in_cache) {
        WorldObject_::task(
            destination_rank,
            &DistributedStorage_::template set_cached_handler<true>, key,
            data_fut, madness::TaskAttributes::hipri_unordered());
      } else {
        WorldObject_::task(
            destination_rank,
            &DistributedStorage_::template set_cached_handler<false>, key,
            data_fut, madness::TaskAttributes::hipri_unordered());
      }
    } else {  // data not ready yet, defer send to a callback (maybe task??)
      const_cast<future&>(data_fut).register_callback(
          new DelayedForward(*this, key, destination_rank, keep_in_cache));
    }
  }

  template <bool KeepInCache>
  void set_cached_handler(const size_type key, const value_type& datum) const {
    // assign the future first, then remove from the cache
    typename container_type::accessor acc;
    [[maybe_unused]] const bool inserted = remote_data_cache_.insert(acc, key);
    // future must be in cache
    TA_ASSERT(!inserted);
    // assign it
    acc->second.set(datum);
    // remove it from the cache
    if constexpr (!KeepInCache) remote_data_cache_.erase(acc);

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    ngets_received_per_rank_.at(this->owner(key))++;
#endif
  }

 public:
  /// Makes an initialized, empty container with default data distribution (no
  /// communication)

  /// A unique ID is associated with every distributed container within a
  /// world.  In order to avoid synchronization when making a container, we
  /// have to assume that all processes execute this constructor in the same
  /// order (does not apply to the non-initializing, default constructor).
  /// \param world The world where the distributed container lives
  /// \param max_size The maximum capacity of this container
  /// \param pmap The process map for the container (default = null pointer)
  DistributedStorage(World& world, size_type max_size,
                     const std::shared_ptr<const pmap_interface>& pmap)
      : WorldObject_(world),
        max_size_(max_size),
        pmap_(pmap),
        data_((max_size / world.size()) + 11)
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
        ,
        ngets_served_per_rank_(world.size()),
        ngets_sent_per_rank_(world.size()),
        ngets_received_per_rank_(world.size())
#endif
  {
    // Check that the process map is appropriate for this storage object
    TA_ASSERT(pmap_);
    TA_ASSERT(pmap_->size() == max_size);
    TA_ASSERT(pmap_->rank() == pmap_interface::size_type(world.rank()));
    TA_ASSERT(pmap_->procs() == pmap_interface::size_type(world.size()));
    num_live_ds_ = 0;
    num_live_df_ = 0;
#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
    for (auto rank = 0; rank != world.size(); ++rank) {
      ngets_served_per_rank_[rank] = 0;
      ngets_sent_per_rank_[rank] = 0;
      ngets_received_per_rank_[rank] = 0;
    }
#endif
    WorldObject_::process_pending();
  }

  virtual ~DistributedStorage() {
    if (num_live_ds_ != 0) {
      madness::print_error("DistributedStorage (object id=", this->id(),
                           ") destroyed while "
                           "pending tasks that set its data exist. Add a "
                           "fence() to extend the lifetime of "
                           "this object.");
      abort();
    }
    if (num_live_df_ != 0) {
      madness::print_error("DistributedStorage (object id=", this->id(),
                           ") destroyed while "
                           "pending callbacks that forward its data to other "
                           "ranks exist. This may indicate a bug in your "
                           "program or you may need to extend the lifetime of "
                           "this object.");
      abort();
    }
  }

  using WorldObject_::get_world;

  /// Process map accessor

  /// \return A shared pointer to the process map.
  /// \throw nothing
  const std::shared_ptr<const pmap_interface>& pmap() const { return pmap_; }

  /// Element owner

  /// \return The process that owns element \c i
  ProcessID owner(size_type i) const {
    TA_ASSERT(i < max_size_);
    TA_ASSERT(pmap_);
    return pmap_->owner(i);
  }

  /// Local element query

  /// Check if element \c i belongs to this node. The element may or may not
  /// be stored. Use \c find to determine if an element is present.
  /// \param i The element to check.
  /// \return \c true when the element is stored locally, otherwise \c false.
  bool is_local(size_type i) const {
    TA_ASSERT(i < max_size_);
    TA_ASSERT(pmap_);
    return pmap_->is_local(i);
  }

  /// Number of local elements

  /// No communication.
  /// \return The number of local elements stored by the container.
  /// \throw nothing
  size_type size() const { return data_.size(); }

  /// Max size accessor

  /// The maximum size is the total number of elements that can be held by
  /// this container on all nodes, not on each node.
  /// \return The maximum number of elements.
  /// \throw nothing
  size_type max_size() const { return max_size_; }

  /// Get local or remote element

  /// \param i The element to get
  /// \return A future to element \c i
  /// \throw TiledArray::Exception If \c i is greater than or equal to \c
  /// max_size() .
  future get(size_type i,
             RemoteDataGetPolicy policy = RemoteDataGetPolicy::nocache) const {
    TA_ASSERT(i < max_size_);
    if (is_local(i)) {
      return get_local(i);
    } else {
      if (policy == RemoteDataGetPolicy::nocache) {
        // Send a request to the owner of i for the element.
        future result;
        WorldObject_::task(owner(i), &DistributedStorage_::get_handler, i,
                           result.remote_ref(get_world()),
                           madness::TaskAttributes::hipri_unordered());
        return result;
      } else
        return get_cached(i, policy == RemoteDataGetPolicy::cache);
    }
  }

  /// Get local element

  /// \param i The element to get
  /// \return A const reference to element \p i
  /// \throw TiledArray::Exception If \p i is greater than or equal to
  /// max_size() or \p i is not local.
  const future& get_local(const size_type i) const {
    TA_ASSERT(pmap_->is_local(i));

    // Return the local element.
    const_accessor acc;
    [[maybe_unused]] const bool inserted = data_.insert(acc, i);
#ifdef MADNESS_WORLDOBJECT_FUTURE_TRACE
    if (inserted) {
      auto& f_nonconst_ref =
          const_cast<std::remove_const_t<decltype(acc->second)>&>(acc->second);
      this->trace(f_nonconst_ref);
    }
#endif
    return acc->second;
  }

  /// Get local element

  /// \param i The element to get
  /// \return A reference to element \p i
  /// \throw TiledArray::Exception If \p i is greater than or equal to
  /// max_size() or \p i is not local.
  future& get_local(const size_type i) {
    TA_ASSERT(pmap_->is_local(i));

    // Return the local element.
    accessor acc;
    [[maybe_unused]] const bool inserted = data_.insert(acc, i);
#ifdef MADNESS_WORLDOBJECT_FUTURE_TRACE
    if (inserted) {
      auto& f_nonconst_ref =
          const_cast<std::remove_const_t<decltype(acc->second)>&>(acc->second);
      this->trace(f_nonconst_ref);
    }
#endif
    return acc->second;
  }

  /// Set element \c i with \c value

  /// \param i The element to be set
  /// \param value The value of element \c i
  /// \throw TiledArray::Exception If \c i is greater than or equal to \c
  /// max_size() .
  /// \throw madness::MadnessException If \c i has already been
  /// set.
  void set(size_type i, const value_type& value) {
    TA_ASSERT(i < max_size_);
    if (is_local(i))
      set_handler(i, value);
    else
      set_remote(i, value);
  }

  /// Set element \c i with \c value

  /// \param i The element to be set
  /// \param value The value of element \c i
  /// \throw TiledArray::Exception If \c i is greater than or equal to \c
  /// max_size() .
  /// \throw madness::MadnessException If \c i has already been
  /// set.
  void set(size_type i, value_type&& value) {
    TA_ASSERT(i < max_size_);
    if (is_local(i))
      set_handler(i, std::move(value));
    else
      set_remote(i, std::move(value));
  }

  /// Set element \c i with a \c Future \c f

  /// The owner of \c i may be local or remote. If \c i is remote, a task
  /// is spawned on the owning node after the local future has been assigned.
  /// If \c i is not already in the container, it will be inserted.
  /// \param i The element to be set
  /// \param f The future for element \c i
  /// \throw madness::MadnessException If \c i has already been set.
  /// \throw TiledArray::Exception If \c i is greater than or equal to \c
  /// max_size() .
  void set(size_type i, const future& f) {
    TA_ASSERT(i < max_size_);
    if (is_local(i)) {
      const_accessor acc;
      if (!data_.insert(acc, typename container_type::datumT(i, f))) {
        // The element was already in the container, so set it with f.
        future existing_f = acc->second;
        acc.release();

        // Check that the future has not been set already.
        TA_ASSERT(!existing_f.probe() && "Tile has already been assigned.");
        // Set the future
        existing_f.set(f);
      }
#ifdef MADNESS_WORLDOBJECT_FUTURE_TRACE
      else {
        auto& f_nonconst_ref =
            const_cast<std::remove_const_t<decltype(acc->second)>&>(
                acc->second);
        this->trace(f_nonconst_ref);
      }
#endif
    } else {
      if (f.probe()) {
        set_remote(i, f);
      } else {
        DelayedSet* set_callback = new DelayedSet(*this, i, f);
        const_cast<future&>(f).register_callback(set_callback);
      }
    }
  }

  /// Reports the number of live DelayedSet requests

  /// @return const reference to the atomic counter of live DelayedSet requests
  const std::atomic<std::size_t>& num_live_ds() const { return num_live_ds_; }

  /// Reports the number of live DelayedForward requests

  /// @return const reference to the atomic counter of live DelayedForward
  /// requests
  const std::atomic<std::size_t>& num_live_df() const { return num_live_df_; }

#ifdef TILEDARRAY_ENABLE_GLOBAL_COMM_STATS_TRACE
  const std::vector<std::atomic<std::size_t>>& ngets_served_per_rank() const {
    return ngets_served_per_rank_;
  }
  const std::vector<std::atomic<std::size_t>>& ngets_sent_per_rank() const {
    return ngets_sent_per_rank_;
  }
  const std::vector<std::atomic<std::size_t>>& ngets_received_per_rank() const {
    return ngets_received_per_rank_;
  }
#endif
};  // class DistributedStorage

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
