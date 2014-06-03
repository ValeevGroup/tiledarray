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
  namespace detail {

    /// Distributed storage container.

    /// Each element in this container is owned by a single node, but any node
    /// may request a copy of the element in the form of a \c madness::Future .
    /// The owner of each element is defined by a process map (pmap), which is
    /// passed to the constructor. Elements do not need to be explicitly
    /// initialized because they will be added to the container when the element
    /// is first accessed, though you may manually initialize an element with
    /// the \c insert() function. All elements are stored in \c madness::Future ,
    /// which may be set only once.
    /// \note This object is derived from \c madness::WorldObject , which means
    /// the order of construction of object must be the same on all nodes. This
    /// can easily be achieved by only constructing world objects in the main
    /// thread. DO NOT construct world objects within tasks where the order of
    /// execution is nondeterministic.
    template <typename T>
    class DistributedStorage : public madness::WorldObject<DistributedStorage<T> > {
    public:
      typedef DistributedStorage<T> DistributedStorage_; ///< This object type
      typedef madness::WorldObject<DistributedStorage_> WorldObject_; ///< Base object type

      typedef std::size_t size_type; ///< size type
      typedef size_type key_type; ///< element key type
      typedef T value_type; ///< Element type
      typedef madness::Future<value_type> future; ///< Element container type
      typedef Pmap pmap_interface; ///< Process map interface type
      typedef madness::ConcurrentHashMap<key_type, future> container_type; ///< Local container type
      typedef typename container_type::accessor accessor; ///< Local element accessor type
      typedef typename container_type::const_accessor const_accessor; ///< Local element const accessor type

    private:

      const size_type max_size_; ///< The maximum number of elements that can be stored by this container
      std::shared_ptr<pmap_interface> pmap_; ///< The process map that defines the element distribution
      mutable container_type data_; ///< The local data container

      // not allowed
      DistributedStorage(const DistributedStorage_&);
      DistributedStorage_& operator=(const DistributedStorage_&);

      future get_local(const size_type i) const {
        TA_ASSERT(pmap_->is_local(i));

        // Return the local element.
        const_accessor acc;
        data_.insert(acc, i);
        return acc->second;
      }

      future get_cache_local(const size_type i) const {
        TA_ASSERT(pmap_->is_local(i));

        accessor acc;
        const bool do_cleanup = !data_.insert(acc, i);
        const future f = acc->second;
        if(do_cleanup)
          data_.erase(acc);
        return f;
      }

      void set_handler(const size_type i, const value_type& value) {
        future f = get_local(i);

#ifndef NDEBUG
          // Check that the future has not been set already.
          if(f.probe())
            TA_EXCEPTION("Tile has already been assigned.");
#endif // NDEBUG

        f.set(value);
      }

      void get_handler(const size_type i, const typename future::remote_refT& ref) {
        future f = get_local(i);
        future remote_f(ref);
        remote_f.set(f);
      }

      void set_cache_handler(const size_type i, const value_type& value) {
        future f = get_cache_local(i);

#ifndef NDEBUG
        // Check that the future has not been set already.
        if(f.probe())
          TA_EXCEPTION("Tile has already been assigned.");
#endif // NDEBUG

        f.set(value);
      }

      void get_cache_handler(const size_type i, const typename future::remote_refT& ref) {
        future f = get_cache_local(i);
        future remote_f(ref);
        remote_f.set(f);
      }

      void set_remote(const size_type i, const value_type& value) {
        WorldObject_::task(owner(i), & DistributedStorage_::set_handler,
            i, value, madness::TaskAttributes::hipri());
      }

      void set_cache_remote(const size_type i, const value_type& value) {
        WorldObject_::task(owner(i), & DistributedStorage_::set_cache_handler,
            i, value, madness::TaskAttributes::hipri());
      }

      struct DelayedSet : public madness::CallbackInterface {
      private:
        DistributedStorage_& ds_; ///< A reference to the owning object
        size_type index_; ///< The index that will own the future
        future future_; ///< The future that we are waiting on.

      public:

        DelayedSet(DistributedStorage_& ds, size_type i, const future& f) :
            ds_(ds), index_(i), future_(f)
        { }

        virtual ~DelayedSet() { }

        virtual void notify() {
          ds_.set_remote(index_, future_);
          delete this;
        }
      }; // struct DelayedSet

      struct DelayedSetCache : public madness::CallbackInterface {
      private:
        DistributedStorage_& ds_; ///< A reference to the owning object
        size_type index_; ///< The index that will own the future
        future future_; ///< The future that we are waiting on.

      public:

        DelayedSetCache(DistributedStorage_& ds, size_type i, const future& f) :
            ds_(ds), index_(i), future_(f)
        { }

        virtual ~DelayedSetCache() { }

        virtual void notify() {
          ds_.set_cache_remote(index_, future_);
          delete this;
        }
      }; // struct DelayedSetCache


    public:

      /// Makes an initialized, empty container with default data distribution (no communication)

      /// A unique ID is associated with every distributed container within a
      /// world.  In order to avoid synchronization when making a container, we
      /// have to assume that all processes execute this constructor in the same
      /// order (does not apply to the non-initializing, default constructor).
      /// \param world The world where the distributed container lives
      /// \param max_size The maximum capacity of this container
      /// \param pmap The process map for the container (default = null pointer)
      DistributedStorage(madness::World& world, size_type max_size,
          const std::shared_ptr<pmap_interface>& pmap) :
        WorldObject_(world), max_size_(max_size),
        pmap_(pmap),
        data_((max_size / world.size()) + 11)
      {
        // Check that the process map is appropriate for this storage object
        TA_ASSERT(pmap_);
        TA_ASSERT(pmap_->size() == max_size);
        TA_ASSERT(pmap_->rank() == world.rank());
        TA_ASSERT(pmap_->procs() == world.size());
        WorldObject_::process_pending();
      }

      virtual ~DistributedStorage() { }

      using WorldObject_::get_world;

      /// Process map accessor

      /// \return A shared pointer to the process map.
      /// \throw nothing
      const std::shared_ptr<pmap_interface>& get_pmap() const { return pmap_; }

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

      /// Get local or remote node

      /// \param i The element to get
      /// \return A future to element \c i
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      future get(size_type i) const {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          return get_local(i);
        } else {
          // Send a request to the owner of i for the element.
          future result;
          WorldObject_::task(owner(i), & DistributedStorage_::get_handler, i,
              result.remote_ref(get_world()), madness::TaskAttributes::hipri());

          return result;
        }
      }

      /// Set element \c i with \c value

      /// \param i The element to be set
      /// \param value The value of element \c i
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      /// \throw madness::MadnessException If \c i has already been set.
      void set(size_type i, const value_type& value) {
        TA_ASSERT(i < max_size_);
        if(is_local(i))
          set_handler(i, value);
        else
          set_remote(i, value);
      }

      /// Set element \c i with a \c madness::Future \c f

      /// The owner of \c i may be local or remote. If \c i is remote, a task
      /// is spawned on the owning node after the local future has been assigned.
      /// If \c i is not already in the container, it will be inserted.
      /// \param i The element to be set
      /// \param f The future for element \c i
      /// \throw madness::MadnessException If \c i has already been set.
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      void set(size_type i, const future& f) {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          const_accessor acc;
          if(! data_.insert(acc, typename container_type::datumT(i, f))) {
            // The element was already in the container, so set it with f.
            future existing_f = acc->second;
            acc.release();

            // Check that the future has not been set already.
#ifndef NDEBUG
            if(existing_f.probe())
              TA_EXCEPTION("Tile has already been assigned.");
#endif // NDEBUG
            // Set the future
            existing_f.set(f);
          }
        } else {
          if(f.probe()) {
            set_remote(i, f);
          } else {
            DelayedSet* set_callback = new DelayedSet(*this, i, f);
            const_cast<future&>(f).register_callback(set_callback);
          }
        }
      }

      /// Element accessor

      /// This operator returns a future to the local or remote element \c i .
      /// If the element is not present, either locally or remotely, it is
      /// inserted into the container on the owner's node.
      /// \param i The element to get
      /// \return A future to element \c i
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      future get_cache(size_type i) const {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          return get_cache_local(i);
        } else {
          // Send a request to the owner of i for the element.
          future result;
          WorldObject_::task(owner(i), & DistributedStorage_::get_cache_handler, i,
              result.remote_ref(get_world()), madness::TaskAttributes::hipri());

          return result;
        }
      }


      /// Set element \c i with \c value

      /// The owner of \c i may be local or remote. If \c i is remote, a task
      /// is spawned on the owning node to set it. If \c i is not already in
      /// the container, it will be inserted.
      /// \param i The element to be set
      /// \param value The value of element \c i
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      /// \throw madness::MadnessException If \c i has already been set.
      void set_cache(size_type i, const value_type& value) {
        TA_ASSERT(i < max_size_);
        if(is_local(i))
          set_cache_handler(i, value);
        else
          set_cache_remote(i, value);
      }

      /// Set element \c i with a \c madness::Future \c f

      /// The owner of \c i may be local or remote. If \c i is remote, a task
      /// is spawned on the owning node after the local future has been assigned.
      /// If \c i is not already in the container, it will be inserted.
      /// \param i The element to be set
      /// \param f The future for element \c i
      /// \throw madness::MadnessException If \c i has already been set.
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      void set_cache(size_type i, const future& f) {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          const_accessor acc;
          if(! data_.insert(acc, typename container_type::datumT(i, f))) {
            // The element was already in the container, so set it with f.
            future existing_f = acc->second;
            data_.erase(acc);

            // Check that the future has not been set already.
#ifndef NDEBUG
            if(existing_f.probe())
              TA_EXCEPTION("Tile has already been assigned.");
#endif // NDEBUG

            // Set the future
            existing_f.set(f);
          }

        } else {
          if(f.probe()) {
            set_cache_remote(i, f);
          } else {
            DelayedSetCache* set_callback = new DelayedSetCache(*this, i, f);
            const_cast<future&>(f).register_callback(set_callback);
          }
        }
      }

    }; // class DistributedStorage

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
