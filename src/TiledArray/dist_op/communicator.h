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
 *  communicator.h
 *  Oct 11, 2013
 *
 */

#ifndef TILEDARRAY_DIST_EVAL_DIST_OP_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_DIST_OP_H__INCLUDED

#include <TiledArray/madness.h>
#include <TiledArray/error.h>
#include <TiledArray/dist_op/dist_cache.h>
#include <TiledArray/dist_op/group.h>
#include <TiledArray/reduce_task.h>

namespace TiledArray {

  namespace dist_op {

    /// Key object that included the process information

    /// \tparam Key The base key type
    /// \tparam Tag A type to differentiate key types
    template <typename Key, typename Tag = void>
    class ProcessKey {
    private:
      Key key_; ///< The base key type
      ProcessID proc_; ///< The process that generated the key

    public:

      /// Default constructor
      ProcessKey() : key_(), proc_(-1) { }

      /// Constructor

      /// \param key The base key
      /// \param proc The process that generated the key
      ProcessKey(const Key& key, const ProcessID proc) :
        key_(key), proc_(proc)
      { }

      /// Copy constructor

      /// \param other The key to be copied
      ProcessKey(const ProcessKey<Key, Tag>& other) :
        key_(other.key_), proc_(other.proc_)
      { }

      /// Copy assignment operator

      /// \param other The key to be copied
      /// \return A reference to this object
      ProcessKey<Key, Tag>& operator=(const ProcessKey<Key, Tag>& other) {
        key_ = other.key_;
        proc_ = other.proc_;
        return *this;
      }

      /// Base key accessor

      /// \return The base key
      const Key& key() const { return key_; }

      /// Process id accessor

      /// \return The process id
      ProcessID proc() const { return proc_; }

      /// Equality comparison

      /// \param other The key to be compared to this
      /// \return \c true when other key and other process are equal to that of
      /// this key, otherwise \c false.
      bool operator==(const ProcessKey<Key, Tag>& other) const {
        return ((key_ == other.key_) && (proc_ == other.proc_));
      }

      /// Inequality comparison

      /// \param other The key to be compared to this
      /// \return \c true when other key or other process are not equal to that
      /// of this key, otherwise \c false.
      bool operator!=(const ProcessKey<Key, Tag>& other) const {
        return ((key_ != other.key_) || (proc_ != other.proc_));
      }

      /// Serialize this key

      /// \tparam Archive The archive type
      /// \param ar The archive object that will serialize this object
      template <typename Archive>
      void serialize(const Archive& ar) {
        ar & key_ & proc_;
      }

      /// Hashing function

      /// \param key The key to be hashed
      /// \return The hashed key value
      friend madness::hashT hash_value(const ProcessKey<Key, Tag>& key) {
        madness::Hash<Key> hasher;
        madness::hashT seed = hasher(key.key_);
        madness::detail::combine_hash(seed, key.proc_);
        return seed;
      }

    }; // class ProcessKey

    /// Key object that uses a tag to differentiate keys

    /// \tparam Key The base key type
    /// \tparam Tag A type to differentiate key types
    template <typename Key, typename Tag>
    class TaggedKey {
    private:
      Key key_; ///< The base key type

    public:

      /// Default constructor
      TaggedKey() : key_() { }

      /// Constructor

      /// \param key The base key
      /// \param proc The process that generated the key
      TaggedKey(const Key& key) : key_(key) { }

      /// Copy constructor

      /// \param other The key to be copied
      TaggedKey(const TaggedKey<Key, Tag>& other) : key_(other.key_) { }

      /// Copy assignment operator

      /// \param other The key to be copied
      /// \return A reference to this object
      TaggedKey<Key, Tag>& operator=(const TaggedKey<Key, Tag>& other) {
        key_ = other.key_;
        return *this;
      }

      /// Base key accessor

      /// \return The base key
      const Key& key() const { return key_; }

      /// Equality comparison

      /// \param other The key to be compared to this
      /// \return \c true when other key and other process are equal to that of
      /// this key, otherwise \c false.
      bool operator==(const TaggedKey<Key, Tag>& other) const {
        return (key_ == other.key_);
      }

      /// Inequality comparison

      /// \param other The key to be compared to this
      /// \return \c true when other key or other process are not equal to that
      /// of this key, otherwise \c false.
      bool operator!=(const TaggedKey<Key, Tag>& other) const {
        return (key_ != other.key_);
      }

      /// Serialize this key

      /// \tparam Archive The archive type
      /// \param ar The archive object that will serialize this object
      template <typename Archive>
      void serialize(const Archive& ar) { ar & key_; }

      /// Hashing function

      /// \param key The key to be hashed
      /// \return The hashed key value
      friend madness::hashT hash_value(const TaggedKey<Key, Tag>& key) {
        madness::Hash<Key> hasher;
        return hasher(key.key_);
      }

    }; // class TagKey

  } // namespace detail

  class Communicator {
  private:

    mutable madness::World* world_; ///< The world of this communicator

    // Message tags
    struct PointToPointTag { };
    struct LazySyncTag { };
    struct GroupLazySyncTag { };
    struct BcastTag { };
    struct GroupBcastTag { };
    struct ReduceTag { };
    struct GroupReduceTag { };
    struct AllReduceTag { };
    struct GroupAllReduceTag { };


    /// Delayed send callback object

    /// This callback object is used to send local data to a remove process
    /// once it has been set.
    /// \tparam T The type of data to be sent
    template <typename Key, typename T>
    class DelayedSend : public madness::CallbackInterface {
    private:
      madness::World* world_; ///< The communication world
      const ProcessID dest_; ///< The destination process id
      const Key key_; ///< The distributed id associated with \c value_
      madness::Future<T> value_; ///< The data to be sent

      // Not allowed
      DelayedSend(const DelayedSend<Key, T>&);
      DelayedSend<Key, T>& operator=(const DelayedSend<Key, T>&);

    public:

      /// Constructor

      /// \param ds The distributed container that owns element i
      /// \param i The element to be moved
      DelayedSend(madness::World* world, const ProcessID dest,
          const Key& key, const madness::Future<T>& value) :
        world_(world), dest_(dest), key_(key), value_(value)
      { }

      virtual ~DelayedSend() { }

      /// Notify this object that the future has been set.

      /// This will set the value of the future on the remote node and delete
      /// this callback object.
      virtual void notify() {
        TA_ASSERT(value_.probe());
        Communicator::send_internal(world_, dest_, key_, value_.get());
        delete this;
      }
    }; // class DelayedSend

    /// Compute parent and child processes of this process in a binary tree

    /// \param[out] parent The parent process of this process in the binary tree
    /// \param[out] child0 The left child process of this process in the binary tree
    /// \param[out] child1 The right child process of this process in the binary tree
    /// \param[in] root The head process of the binary tree
    static void make_tree(ProcessID& parent, ProcessID& child0, ProcessID& child1,
        const ProcessID root, const madness::World* const world)
    {
      world->mpi.binary_tree_info(root, parent, child0, child1);
    }

    /// Compute parent and child processes of this process in a binary tree

    /// \param[out] parent The parent process of this process in the binary tree
    /// \param[out] child0 The left child process of this process in the binary tree
    /// \param[out] child1 The right child process of this process in the binary tree
    /// \param[in] group_root The head process in the group of the binary tree
    /// \param[in] group The group where the binary tree will be constructed
    static void make_tree(ProcessID& parent, ProcessID& child0, ProcessID& child1,
        const ProcessID group_root, const dist_op::Group& group)
    {
      group.make_tree(group_root, parent, child0, child1);
    }


    template <typename Key, typename T>
    static void bcast_handler(const madness::AmArg& arg) {

      Key key;
      T value;
      ProcessID root;

      arg & key & value & root;

      // Add task to queue
      arg.get_world()->taskq.add(Communicator::template bcast_task<Key, T>,
          arg.get_world(), key, value, root, madness::TaskAttributes::hipri());
    }

    template <typename Key, typename T>
    static void group_bcast_handler(const madness::AmArg& arg) {
      // Deserialize message arguments
      Key key;
      T value;
      ProcessID group_root;
      dist_op::DistributedID group_key;

      arg & key & value & group_root & group_key;

      // Get the local group
      const madness::Future<dist_op::Group> group = dist_op::Group::get_group(group_key);

      // Add task to queue
      arg.get_world()->taskq.add(Communicator::template group_bcast_task<Key, T>,
          arg.get_world(), key, value, group_root, group,
          madness::TaskAttributes::hipri());
    }

    template <typename Key, typename T>
    static void bcast_children(madness::World* world, const Key& key,
        const T& value, const ProcessID root)
    {
      // Get the parent and child processes in the binary tree that will be used
      // to broadcast the data.
      ProcessID parent = -1, child0 = -1, child1 = -1;
      make_tree(parent, child0, child1, root, world);

      const bool send0 = child0 != -1;
      const bool send1 = child1 != -1;
      madness::AmArg* args = (send0 || send1 ?
          madness::new_am_arg(key, value, root) :
          NULL);

      if(send0)
        world->am.send(child0, & Communicator::template bcast_handler<Key, T>,
            args, madness::RMI::ATTR_ORDERED, !send1);
      if(send1)
        world->am.send(child1, & Communicator::template bcast_handler<Key, T>,
            args, madness::RMI::ATTR_ORDERED, true);
    }

    template <typename Key, typename T>
    static void group_bcast_children(madness::World* world, const Key& key,
        const T& value, const ProcessID group_root, const dist_op::Group& group)
    {
      // Get the parent and child processes in the binary tree that will be used
      // to broadcast the data.
      ProcessID parent = -1, child0 = -1, child1 = -1;
      make_tree(parent, child0, child1, group_root, group);

      // Create active message arguments
      const bool send0 = child0 != -1;
      const bool send1 = child1 != -1;
      madness::AmArg* args = (send0 || send1 ?
          madness::new_am_arg(key, value, group_root, group.id()) :
          NULL);

      // Bcast to children
      if(send0)
        world->am.send(child0, & Communicator::template group_bcast_handler<Key, T>,
            args, madness::RMI::ATTR_ORDERED, !send1);
      if(send1)
        world->am.send(child1, & Communicator::template group_bcast_handler<Key, T>,
            args, madness::RMI::ATTR_ORDERED, true);
    }

    template <typename Key, typename T>
    static void bcast_task(madness::World* world, const Key& key,
        const T& value, ProcessID root)
    {
      dist_op::DistCache<Key>::set_cache_data(key, value);
      bcast_children(world, key, value, root);
    }

    template <typename Key, typename T>
    static void group_bcast_task(madness::World* world, const Key& key,
        const T& value, ProcessID group_root, dist_op::Group& group)
    {
      dist_op::DistCache<Key>::set_cache_data(key, value);
      group_bcast_children(world, key, value, group_root, group);
    }

    /// Receive data from remote node

    /// \tparam T The data type stored in cache
    /// \param did The distributed ID
    /// \return A future to the data
    template <typename T, typename Key>
    static madness::Future<T> recv_internal(const Key& key) {
      return dist_op::DistCache<Key>::template get_cache_data<T>(key);
    }

    /// Send value to \c dest

    /// \tparam T The value type
    /// \param world The world that will be used to send the value
    /// \param dest The node where the data will be sent
    /// \param did The distributed id that is associatied with the data
    /// \param value The data to be sent
    template <typename Key, typename T>
    static typename madness::disable_if<madness::is_future<T> >::type
    send_internal(madness::World* const world, const ProcessID dest,
        const Key& key, const T& value)
    {
      typedef TiledArray::dist_op::DistCache<Key> dist_cache;

      if(world->rank() == dest) {
        // When dest is this process, skip the task and set the future immediately.
        dist_cache::set_cache_data(key, value);
      } else {
        // Spawn a remote task to set the value
        world->taskq.add(dest, dist_cache::template set_cache_data<T>, key,
            value, madness::TaskAttributes::hipri());
      }
    }

    /// Send \c value to \c dest

    /// \tparam T The value type
    /// \param world The world that will be used to send the value
    /// \param dest The node where the data will be sent
    /// \param did The distributed id that is associated with the data
    /// \param value The data to be sent
    template <typename Key, typename T>
    static void send_internal(madness::World* world, ProcessID dest,
        const Key& key, const madness::Future<T>& value)
    {
      typedef TiledArray::dist_op::DistCache<Key> dist_cache;

      if(world->rank() == dest) {
        dist_cache::set_cache_data(key, value);
      } else {
        // The destination is not this node, so send it to the destination.
        if(value.probe()) {
          // Spawn a remote task to set the value
          world->taskq.add(dest, dist_cache::template set_cache_data<T>, key,
              value.get(), madness::TaskAttributes::hipri());
        } else {
          // The future is not ready, so create a callback object that will
          // send value to the destination node when it is ready.
          DelayedSend<Key, T>* delayed_send_callback =
              new DelayedSend<Key, T>(world, dest, key, value);
          const_cast<madness::Future<T>&>(value).register_callback(delayed_send_callback);

        }
      }
    }


    template <typename Key>
    static void lazy_sync_parent(madness::World* world, const ProcessID parent,
        const Key& key, const ProcessID, const ProcessID)
    {
      send_internal(world, parent, key, key.proc());
    }

    template <typename Key, typename Op>
    static void lazy_sync_children(madness::World* world, const ProcessID child0,
        const ProcessID child1, const Key& key, const Op& op, const ProcessID)
    {
      // Signal children to execute the operation.
      if(child0 != -1)
        send_internal(world, child0, key, 1);
      if(child1 != -1)
        send_internal(world, child1, key, 1);

      // Execute the operation on this process.
      op();
    }

    /// Lazy sync

    /// Lazy sync functions are asynchronous barriers with a nullary functor
    /// that is called after all processes have called lazy sync with the same
    /// key.
    /// \param key The sync key
    /// \param op The sync operation to be executed on this process
    /// \note It is the user's responsibility to ensure that the key for each
    /// lazy sync operation is unique. You may reuse keys after the associated
    /// sync operations have been completed.
    template <typename Tag, typename Comm, typename Key, typename Op>
    void lazy_sync_internal(Comm& comm, const Key& key, const Op& op) const {
      typedef dist_op::ProcessKey<Key, Tag> key_type;
      ProcessID parent = -1, child0 = -1, child1 = -1;
      make_tree(parent, child0, child1, 0, comm);

      // Get signals from parent and children.
      madness::Future<ProcessID> child0_signal = (child0 != -1 ?
          recv_internal<ProcessID>(key_type(key, child0)) :
          madness::Future<ProcessID>(-1));
      madness::Future<ProcessID> child1_signal = (child1 != -1 ?
          recv_internal<ProcessID>(key_type(key, child1)) :
          madness::Future<ProcessID>(-1));
      madness::Future<ProcessID> parent_signal = (parent != -1 ?
          recv_internal<ProcessID>(key_type(key, parent)) :
          madness::Future<ProcessID>(-1));

      // Construct the task that notifies children to run the operation
      key_type my_key(key, world_->rank());
      world_->taskq.add(Communicator::template lazy_sync_children<key_type, Op>,
          world_, child0_signal, child1_signal, my_key, op, parent_signal,
          madness::TaskAttributes::hipri());

      // Send signal to parent
      if(parent != -1) {
        if(child0_signal.probe() && child1_signal.probe())
          lazy_sync_parent(world_, parent, my_key, child0_signal, child1_signal);
        else
          world_->taskq.add(Communicator::template lazy_sync_parent<key_type>,
              world_, parent, my_key, child0_signal, child1_signal,
              madness::TaskAttributes::hipri());
      }
    }


    /// Distributed reduce

    /// \tparam Key The key type
    /// \tparam T The data type to be reduced
    /// \tparam Op The reduction operation type
    /// \param key The key associated with this reduction
    /// \param value The local value to be reduced
    /// \param op The reduction operation to be applied to local and remote data
    /// \param root The process that will receive the result of the reduction
    /// \return A future to the reduce value on the root process, otherwise an
    /// uninitialized future that may be ignored.
    template <typename Tag, typename Comm, typename Key, typename T, typename Op>
    madness::Future<typename madness::detail::result_of<Op>::type>
    reduce_internal(Comm& comm, const Key& key, const T& value, const Op& op, const ProcessID root) {
      // Create tagged key
      typedef dist_op::ProcessKey<Key, Tag> key_type;
      typedef typename madness::detail::result_of<Op>::type result_type;

      // Reduce local data
      TiledArray::detail::ReduceTask<Op> reduce_task(*world_, op);
      reduce_task.add(value);

      // Get the parent and child processes in the binary tree that will be used
      // to reduce the data.
      ProcessID parent = -1, child0 = -1, child1 = -1;
      make_tree(parent, child0, child1, root, comm);

      // Reduce child data
      if(child0 != -1)
        reduce_task.add(recv_internal<result_type>(key_type(key, child0)));
      if(child1 != -1)
        reduce_task.add(recv_internal<result_type>(key_type(key, child1)));

      // Send reduced value to parent or, if this is the root process, set the
      // result future.
      if(parent != -1)
        send_internal(world_, parent, key_type(key, world_->rank()), reduce_task.submit());
      else
        return reduce_task.submit();

      return madness::Future<result_type>::default_initializer();
    }

    /// Broadcast

    /// Broadcast data from the \c root process to all processes in \c world.
    /// The input/output data is held by \c value.
    /// \param[in] key The key associated with this broadcast
    /// \param[in,out] value On the \c root process, this is used as the input
    /// data that will be broadcast to all other processes in the group.
    /// On other processes it is used as the output to the broadcast
    /// \param root The process that owns the data to be broadcast
    /// \throw TiledArray::Exception When \c root is less than 0 or
    /// greater than or equal to the world size.
    /// \throw TiledArray::Exception When \c value has been set, except on the
    /// \c root process.
    template <typename Tag, typename Key, typename T>
    void bcast_internal(const Key& key, madness::Future<T>& value, const ProcessID root) const {
      TA_ASSERT((root >= 0) && (root < world_->size()));
      TA_ASSERT((world_->rank() == root) || (! value.probe()));

      // Add operation tag to key
      typedef dist_op::TaggedKey<Key, Tag> key_type;
      const key_type k(key);

      if(world_->size() > 1) { // Do nothing for the trivial case
        if(world_->rank() == root) {
          // This is the process that owns the data to be broadcast

          // Spawn remote tasks that will set the local cache for this broadcast
          // on other nodes.
          if(value.probe())
            // The value is ready so send it now
            bcast_children(world_, k, value.get(), root);
          else
            // The value is not ready so spawn a task to send the data when it
            // is ready.
            world_->taskq.add(Communicator::template bcast_children<key_type, T>,
                world_, k, value, root, madness::TaskAttributes::hipri());
        } else {
          TA_ASSERT(! value.probe());

          // Get the broadcast value from local cache
          dist_op::DistCache<key_type>::get_cache_data(k, value);
        }
      }
    }

    /// Group broadcast

    /// Broadcast data from the \c group_root process to all processes in
    /// \c group. The input/output data is held by \c value.
    /// \param[in] key The key associated with this broadcast
    /// \param[in,out] value On the \c group_root process, this is used as the
    /// input data that will be broadcast to all other processes in the group.
    /// On other processes it is used as the output to the broadcast
    /// \param group_root The process in \c group that owns the data to be
    /// broadcast
    /// \param group The process group where value will be broadcast
    /// \throw TiledArray::Exception When the world id of \c group is not
    /// equal to that of the world used to construct this communicator.
    /// \throw TiledArray::Exception When \c group_root is less than 0 or
    /// greater than or equal to \c group size.
    /// \throw TiledArray::Exception When \c data has been set except on the
    /// \c root process.
    /// \throw TiledArray::Exception When this process is not in the group.
    template <typename Tag, typename Key, typename T>
    void bcast_internal(const Key& key, madness::Future<T>& value,
        const ProcessID group_root, const dist_op::Group& group) const
    {
      // Typedefs
      typedef dist_op::TaggedKey<Key, Tag> key_type;
      const key_type k(key);

      if(group.size() > 1) { // Do nothing for the trivial case
        if(group.rank() == group_root) {
          // This is the process that owns the data to be broadcast
          if(value.probe())
            group_bcast_children(world_, k, value.get(), group_root, group);
          else
            world_->taskq.add(& Communicator::template group_bcast_children<key_type, T>,
                world_, k, value, group_root, group, madness::TaskAttributes::hipri());
        } else {
          // This is not the root process, so retrieve the broadcast data
          dist_op::DistCache<key_type>::get_cache_data(k, value);
        }
      }
    }

    void varify_group(const dist_op::Group& group) const {
      TA_ASSERT(! group.empty());
      TA_ASSERT(group.is_registered());
      TA_ASSERT(group.get_world().id() == world_->id());
      TA_ASSERT(group.rank(world_->rank()) != -1);
    }

  public:

    /// Constructor

    /// \param world The world that will be used to send/receive messages
    Communicator(madness::World& world) : world_(&world) { }

    /// Copy constructor

    /// \param other The object to be copied
    Communicator(const Communicator& other) : world_(other.world_) { }

    /// Copy assignment operator

    /// \param other The object to be copied
    /// \return A reference to this object
    Communicator& operator=(const Communicator& other) {
      world_ = other.world_;
      return *this;
    }

    /// Communicator ID accessor

    /// \return The universe wide unique ID of this communicator (same as the
    /// world id).
    unsigned int id() const { return world_->id(); }

    /// Receive data from remote node

    /// \tparam T The data type stored in cache
    /// \param did The distributed ID
    /// \return A future to the data
    template <typename T, typename Key>
    static madness::Future<T> recv(const ProcessID source, const Key& key) {
      return recv_internal<T>(dist_op::ProcessKey<Key, PointToPointTag>(key, source));
    }

    /// Send value to \c dest

    /// \tparam T The value type
    /// \param world The world that will be used to send the value
    /// \param dest The node where the data will be sent
    /// \param did The distributed id that is associatied with the data
    /// \param value The data to be sent
    template <typename Key, typename T>
    void send(const ProcessID dest, const Key& key, const T& value) const {
      send_internal(world_, dest, dist_op::ProcessKey<Key, PointToPointTag>(key,
          world_->rank()), value);
    }

    /// Lazy sync

    /// Lazy sync functions are asynchronous barriers with a nullary functor
    /// that is called after all processes have called lazy sync with the same
    /// key.
    /// \param key The sync key
    /// \param op The sync operation to be executed on this process
    /// \note It is the user's responsibility to ensure that the key for each
    /// lazy sync operation is unique. You may reuse keys after the associated
    /// sync operations have been completed.
    template <typename Key, typename Op>
    void lazy_sync(const Key& key, const Op& op) const {
      lazy_sync_internal<LazySyncTag>(world_, key, op);
    }


    /// Group lazy sync

    /// Lazy sync functions are asynchronous barriers with a nullary functor
    /// that is called after all processes have called lazy sync with the same
    /// key.
    /// \param key The sync key
    /// \param op The sync operation to be executed on this process
    /// \throw TiledArray::Exception When the world id of the group and the
    /// world id of this communicator are not equal.
    /// \throw TiledArray::Exception When this process is not in the group.
    /// \note It is the user's responsibility to ensure that the key for each
    /// lazy sync operation is unique. You may reuse keys after the associated
    /// sync operations have been completed.
    template <typename Key, typename Op>
    void lazy_sync(const Key& key, const Op& op, const dist_op::Group& group) const {
      varify_group(group);
      lazy_sync_internal<GroupLazySyncTag>(group, key, op);
    }

    /// Broadcast

    /// Broadcast data from the \c root process to all processes in \c world.
    /// The input/output data is held by \c value.
    /// \param[in] key The key associated with this broadcast
    /// \param[in,out] value On the \c root process, this is used as the input
    /// data that will be broadcast to all other processes in the group.
    /// On other processes it is used as the output to the broadcast
    /// \param root The process that owns the data to be broadcast
    /// \throw TiledArray::Exception When \c root is less than 0 or
    /// greater than or equal to the world size.
    /// \throw TiledArray::Exception When \c value has been set, except on the
    /// \c root process.
    template <typename Key, typename T>
    void bcast(const Key& key, madness::Future<T>& value, const ProcessID root) const {
      TA_ASSERT((root >= 0) && (root < world_->size()));
      TA_ASSERT((world_->rank() == root) || (! value.probe()));
      bcast_internal<BcastTag>(key, value, root);
    }

    /// Group broadcast

    /// Broadcast data from the \c group_root process to all processes in
    /// \c group. The input/output data is held by \c value.
    /// \param[in] key The key associated with this broadcast
    /// \param[in,out] value On the \c group_root process, this is used as the
    /// input data that will be broadcast to all other processes in the group.
    /// On other processes it is used as the output to the broadcast
    /// \param group_root The process in \c group that owns the data to be
    /// broadcast
    /// \param group The process group where value will be broadcast
    /// \throw TiledArray::Exception When the world id of \c group is not
    /// equal to that of the world used to construct this communicator.
    /// \throw TiledArray::Exception When \c group_root is less than 0 or
    /// greater than or equal to \c group size.
    /// \throw TiledArray::Exception When \c data has been set except on the
    /// \c root process.
    /// \throw TiledArray::Exception When this process is not in the group.
    template <typename Key, typename T>
    void bcast(const Key& key, madness::Future<T>& value,
        const ProcessID group_root, const dist_op::Group& group) const
    {
      varify_group(group);
      TA_ASSERT((group_root >= 0) && (group_root < group.size()));
      TA_ASSERT((group.rank() == group_root) || (! value.probe()));
      bcast_internal<GroupBcastTag>(key, value, group_root, group);
    }

    /// Distributed reduce

    /// \tparam Key The key type
    /// \tparam T The data type to be reduced
    /// \tparam Op The reduction operation type
    /// \param key The key associated with this reduction
    /// \param value The local value to be reduced
    /// \param op The reduction operation to be applied to local and remote data
    /// \param root The process that will receive the result of the reduction
    /// \return A future to the reduce value on the root process, otherwise an
    /// uninitialized future that may be ignored.
    template <typename Key, typename T, typename Op>
    madness::Future<typename madness::detail::result_of<Op>::type>
    reduce(const Key& key, const T& value, const Op& op, const ProcessID root) {
      TA_ASSERT((root >= 0) && (root < world_->size()));
      return reduce_internal<ReduceTag>(world_, key, value, op, root);
    }

    /// Distributed group reduce

    /// \tparam Key The key type
    /// \tparam T The data type to be reduced
    /// \tparam Op The reduction operation type
    /// \param key The key associated with this reduction
    /// \param value The local value to be reduced
    /// \param op The reduction operation to be applied to local and remote data
    /// \param group_root The group process that will receive the result of the reduction
    /// \param group The group that will preform the reduction
    /// \return A future to the reduce value on the root process, otherwise an
    /// uninitialized future that may be ignored.
    template <typename Key, typename T, typename Op>
    madness::Future<typename madness::detail::result_of<Op>::type>
    reduce(const Key& key, const T& value, const Op& op,
        const ProcessID group_root, const dist_op::Group& group)
    {
      varify_group(group);
      TA_ASSERT((group_root >= 0) && (group_root < group.size()));
      return reduce_internal<ReduceTag>(group, key, value, op, group_root);
    }

    /// Distributed all reduce

    /// \tparam Key The key type
    /// \tparam T The data type to be reduced
    /// \tparam Op The reduction operation type
    /// \param key The key associated with this reduction
    /// \param value The local value to be reduced
    /// \param op The reduction operation to be applied to local and remote data
    /// \param root The process that will receive the result of the reduction
    /// \return A future to the reduce value on the root process, otherwise an
    /// uninitialized future that may be ignored.
    template <typename Key, typename T, typename Op>
    madness::Future<typename madness::detail::result_of<Op>::type>
    reduce(const Key& key, const T& value, const Op& op) {
      // Pick a pseudo random root process
      madness::Hash<Key> hasher;
      const ProcessID root = hasher(key) % world_->size();

      // Reduce to 0 process
      madness::Future<typename madness::detail::result_of<Op>::type> reduce_result =
          reduce_internal<AllReduceTag>(world_, key, value, op, root);

      if(world_->rank() != root)
        reduce_result = madness::Future<typename madness::detail::result_of<Op>::type>();

      // Broadcast the result of the reduction to all processes
      bcast_internal<AllReduceTag>(key, reduce_result, root);

      return reduce_result;
    }

    /// Distributed group all reduce

    /// \tparam Key The key type
    /// \tparam T The data type to be reduced
    /// \tparam Op The reduction operation type
    /// \param key The key associated with this reduction
    /// \param value The local value to be reduced
    /// \param op The reduction operation to be applied to local and remote data
    /// \param group_root The group process that will receive the result of the reduction
    /// \param group The group that will preform the reduction
    /// \return A future to the reduce value on the root process, otherwise an
    /// uninitialized future that may be ignored.
    template <typename Key, typename T, typename Op>
    madness::Future<typename madness::detail::result_of<Op>::type>
    reduce(const Key& key, const T& value, const Op& op, const dist_op::Group& group) {
      varify_group(group);

      // Pick a pseudo random root process
      madness::Hash<Key> hasher;
      const ProcessID group_root = hasher(key) % group.size();

      // Reduce the data
      madness::Future<typename madness::detail::result_of<Op>::type> reduce_result =
          reduce_internal<GroupAllReduceTag>(group, key, value, op, group_root);


      if(group.rank() != group_root)
        reduce_result = madness::Future<typename madness::detail::result_of<Op>::type>();

      // Broadcast the result of the reduction to all processes in the group
      bcast_internal<GroupAllReduceTag>(key, reduce_result, 0, group);

      return reduce_result;
    }

  }; // class Communicator

} // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_DIST_OP_H__INCLUDED
