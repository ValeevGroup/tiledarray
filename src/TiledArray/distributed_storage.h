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

#include <TiledArray/error.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/madness.h>

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
      // not allowed
      DistributedStorage(const DistributedStorage_&);
      DistributedStorage_& operator=(const DistributedStorage_&);

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
          const std::shared_ptr<pmap_interface>& pmap = std::shared_ptr<pmap_interface>()) :
        WorldObject_(world), max_size_(max_size),
        pmap_(pmap),
        data_((max_size / world.size()) + 11)
      {
        if(pmap_) {
          // Check that the process map is appropriate for this storage object
          TA_ASSERT(pmap_->size() == max_size);
          TA_ASSERT(pmap_->rank() == world.rank());
          TA_ASSERT(pmap_->procs() == world.size());
//          pmap_->set_seed(WorldObject_::id().get_obj_id());
          WorldObject_::process_pending();
        }
      }

      virtual ~DistributedStorage() { }

      /// Initialize the container

      /// Process any messages that arrived before this object was constructed
      /// locally.
      /// \note No incoming messages are processed until this routine is invoked.
      /// It can be invoked in the constructor by passing \c true to the
      /// \c do_pending argument.
      void init(const std::shared_ptr<pmap_interface>& pmap) {
        TA_ASSERT(!pmap_);
        // Check that the process map is appropriate for this storage object
        TA_ASSERT(pmap);
        TA_ASSERT(pmap->size() == max_size_);
        TA_ASSERT(pmap->rank() == WorldObject_::get_world().rank());
        TA_ASSERT(pmap->procs() == WorldObject_::get_world().size());
        pmap_ = pmap;
//        pmap_->set_seed(WorldObject_::id().get_obj_id());
        WorldObject_::process_pending();
      }

      /// World accessor

      /// \return A reference to the world this object is associated with.
      /// \throw nothing
      madness::World& get_world() const { return WorldObject_::get_world(); }

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
      bool is_local(size_type i) const { return owner(i) == get_world().rank(); }

      /// Clear local data

      /// Remove all local data.
      /// \throw nothing
      void clear() { data_.clear(); }

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

      /// Insert element

      /// Add element \c i to the container. If it is not local, a message is
      /// sent to the owner to insert it.
      /// \param i The element to be inserted.
      /// \return true if the tile was inserted locally, otherwise false.
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      /// \note If you want to insert an element with a value, use the \c set()
      /// function.
      bool insert(size_type i) {
        TA_ASSERT(i < max_size_);
        if(is_local(i))
          return data_.insert(typename container_type::datumT(i, future())).second;

        WorldObject_::send(owner(i), & DistributedStorage_::remote_insert, i);
        return false;
      }

      /// Insert an empty future into all locally owned elements
      void insert_local() {
        typename pmap_interface::const_iterator end = pmap_->end();
        for(typename pmap_interface::const_iterator it = pmap_->begin(); it != end; ++it) {
          data_.insert(*it);
        }
      }

      /// Insert an empty future into all locally owned elements where \c pred \c true

      /// \tparam Pred A predicate type
      /// \param pred A predicate that returns true or false for a given element
      /// index
      template <typename Pred>
      void insert_local(const Pred& pred) {
        typename pmap_interface::const_iterator end = pmap_->end();
        for(typename pmap_interface::const_iterator it = pmap_->begin(); it != end; ++it) {
          if(pred(*it))
            data_.insert(*it);
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
      void set(size_type i, const value_type& value) {
        TA_ASSERT(i < max_size_);
        set_value(i, value);
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
            check_future(f);
#endif // NDEBUG

            // Set the future
            existing_f.set(f);
          }
        } else {
          if(f.probe()) {
            // f is ready, so it can be immidiately sent to the owner.
            set_value(i, f.get());
          } else {
            // f is not ready, so create a callback to send it to the owner when
            // it is set.
            DelayedSet* set_callback = new DelayedSet(*this, i, f);
            const_cast<future&>(f).register_callback(set_callback);
          }
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
      void set(size_type i, const madness::detail::MoveWrapper<value_type>& value) {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          set_local_value(i, value);
        } else {
          WorldObject_::send(owner(i), & DistributedStorage_::set_value, i,
              madness::unwrap_move(value));
        }
      }

      /// Element accessor

      /// This operator returns a future to the local or remote element \c i .
      /// If the element is not present, either local or remote, it is inserted
      /// into the container on the owner's node.
      /// \return A future to the element.
      /// \throw TiledArray::Exception If \c i is greater than or equal to \c max_size() .
      future operator[](size_type i) const {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          // Return the local element.
          const_accessor acc;
          data_.insert(acc, i);
          return acc->second;
        }

        // Send a request to the owner of i for the element.
        future result;
        WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()), false, madness::TaskAttributes::hipri());

        return result;
      }

      /// The caller takes ownership of element \c i

      /// Ownership can only take place after the element has been set. If the
      /// element has not been set when this function is called, then the
      /// transfer of ownership will be delayed (and handled automatically) until
      /// the element is set.
      /// \note It is the caller's responsibility to ensure that move is only
      /// call once. Otherwise, the program will hang.
      future move(size_type i) {
        TA_ASSERT(i < max_size_);

        if(is_local(i)) {
          // Get element i
          const_accessor acc;
          data_.insert(acc, i);
          future result = acc->second;

          // Remove the element from the local container.
          if(result.probe()) {
            data_.erase(acc);
          } else {
            acc.release();
            result.register_callback(new DelayedMove(*this, i));
          }

          return result;
        }

        // Send a request to the owner of i for the element.
        future result;
        WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()), true, madness::TaskAttributes::hipri());

        return result;
      }

    private:

      /// Check that the future has not been previously assigned

      /// \param f The future to be checked
      /// \throw TiledArray::Exception When \c f has been previously set.
      static void check_future(const future& f) {
        if(f.probe())
          TA_EXCEPTION("Tile has already been assigned.");
      }

      /// Callback object to move a tile from this container once it has been set

      /// This callback object is used to remove elements from the container
      /// once it has been set.
      struct DelayedMove : public madness::CallbackInterface {
      private:
        DistributedStorage_& ds_; ///< A reference to the owning object
        size_type index_; ///< The index that will own the future

      public:

        /// Constructor

        /// \param ds The distributed container that owns element i
        /// \param i The element to be moved
        DelayedMove(const DistributedStorage_& ds, size_type i) :
            ds_(const_cast<DistributedStorage_&>(ds)), index_(i)
        { }

        virtual ~DelayedMove() { }

        /// Notify this object when the future is set.

        /// This will set the value of the future on the remote node and delete
        /// this callback object.
        virtual void notify() {
          accessor acc;
          ds_.data_.find(acc, index_);
          ds_.data_.erase(acc);
          delete this;
        }
      }; // struct DelayedSet

      /// Set value callback object

      /// This object is used to set a future that is owned by another node.
      /// Since futures must be set before they can be sent to another node
      /// this additional logic is needed. A task could be used to accomplish
      /// the same goal, but this is more efficient since it is done immediately
      /// when the future is set rather than going through the task queue.
      struct DelayedSet : public madness::CallbackInterface {
      private:
        DistributedStorage_& ds_; ///< A reference to the owning object
        size_type index_; ///< The index that will own the future
        future future_; ///< The future that we are waiting on.

      public:

        DelayedSet(DistributedStorage_& ds, size_type i, const future& fut) :
            ds_(ds), index_(i), future_(fut)
        { }

        virtual ~DelayedSet() { }

        /// Notify this object when the future is set.

        /// This will set the value of the future on the remote node and delete
        /// this callback object.
        virtual void notify() {
          ds_.set_value(index_, future_);
          delete this;
        }
      }; // struct DelayedSet

      /// Set the value of an element
      void set_value(size_type i, const value_type& value) {
        if(is_local(i)) {
          set_local_value(i, value);
        } else {
          WorldObject_::send(owner(i), & DistributedStorage_::set_value, i, value);
        }
      }

      /// Set the value of a local element

      /// Assume that the tile locality has already been checked
      /// \tparam Value The element type
      /// \param i The index of the element to be set
      /// \param value The value that will be assigned to element \c i
      template <typename Value>
      void set_local_value(size_type i, const Value& value) {
        // Get the future for element i
        const_accessor acc;
        data_.insert(acc, i);
        future f = acc->second;
        acc.release();

        // Check that the future has not been set already.
#ifndef NDEBUG
        check_future(f);
#endif // NDEBUG

        // Assign the element
        f.set(value);
      }

      /// Remote insert without a return message

      /// This is a task function that is used to spawn remote insert tasks
      /// \param i The element to be inserted
      void remote_insert(size_type i) {
        TA_ASSERT(is_local(i));
        data_.insert(typename container_type::datumT(i, future()));
      }

      /// A delayed return callback object

      /// This object is used to register callbacks for local elements that will
      /// be sent to another node after it has been set.
      /// \note This object will delete itself after notify has been called.
      struct DelayedReturn : public madness::CallbackInterface {
      private:
        DistributedStorage_& ds_; ///< The distributed storage container that owns the element
        size_type index_; ///< The element index
        typename future::remote_refT ref_; ///< The element remote reference on the requesting node
        future future_; ///< The element future that will be sent
        bool move_; ///< \c true if the element will be removed after sending it

        // Not allowed
        DelayedReturn(const DelayedReturn&);
        DelayedReturn& operator=(const DelayedReturn&);

      public:

        /// Constructor

        /// \param ds The distributed storage container that owns \c fut
        /// \param index The index of the element that will be sent
        /// \param ref The remote reference to the future that will hold the tile on the requesting node
        /// \param fut The local future for the element to be sent
        /// \param move \c true if the element will be removed from \c ds after sending it
        DelayedReturn(const DistributedStorage_& ds, size_type index,
            const typename future::remote_refT& ref, const future& fut, bool move) :
          ds_(const_cast<DistributedStorage_&>(ds)), index_(index),
          ref_(ref), future_(fut), move_(move)
        { }

        virtual void notify() {
          future remote_future(ref_);
          remote_future.set(future_);
          if(move_) {
            accessor acc;
            ds_.data_.find(acc, index_);
            ds_.data_.erase(acc);
          }
          delete this;
        }
      }; // struct DelayedReturn

      /// Handles find request

      /// This function is used to find and return element \c i . If \c mover
      /// is true, the tile will be removed from the local container after
      /// sending it to the node that requested the tile. The element will only
      /// be forwarded after it has been set.
      /// \param i The element to be found.
      /// \param ref The remote reference for the element on the requesting node
      /// \param mover This is \c true if find handler was called by \c move()
      void find_handler(size_type i, const typename future::remote_refT& ref, bool mover) const {
        TA_ASSERT(is_local(i));
        // Find the local element
        const_accessor acc;
        data_.insert(acc, i);
        future f = acc->second;

        if(f.probe()) {
          // The element has been set, so it can be returned immediately.
          future remote_future(ref);
          remote_future.set(f);
          if(mover)
            data_.erase(acc);
        } else {
          // The element has not been set so create a delayed return callback
          acc.release();
          f.register_callback(new DelayedReturn(*this, i, ref, f, mover));
        }
      }

      const size_type max_size_; ///< The maximum number of elements that can be stored by this container
      std::shared_ptr<pmap_interface> pmap_; ///< The process map that defines the element distribution
      mutable container_type data_; ///< The local data container
    };

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
