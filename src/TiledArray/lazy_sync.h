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

#ifndef TILEDARRAY_LAZY_SYNC_H__INCLUDED
#define TILEDARRAY_LAZY_SYNC_H__INCLUDED

#include <TiledArray/madness.h>

namespace TiledArray {
  namespace detail {

    /// Lazy synchronization base

    /// This object handles general synchronization between nodes.
    /// \tparam keyT The synchronization key type
    template <typename keyT>
    class LazySyncBase {
    protected:
      typedef LazySyncBase<keyT> LazySyncBase_; ///< This object type
      typedef madness::ConcurrentHashMap<keyT, LazySyncBase_*> mapT; ///< LazySync object container

      static mapT map_; ///< Map to sync objects so they are accessible to
      ///< other nodes via active messages.

      madness::World& world_; ///< The world where the sync object lives
      keyT key_; ///< The sync key
      ProcessID parent_; ///< The parent node of this node ( = -1 if this is the root node)
      ProcessID child0_; ///< The first child node of this node (=-1 if there is no first child node)
      ProcessID child1_; ///< The second child node of this node (=-1 if there is no second child node)
      madness::detail::WorldPtr<LazySyncBase> parent_wptr_; ///< Pointer to the sync object on the parent node
      madness::detail::WorldPtr<LazySyncBase> child0_wptr_; ///< Pointer to the sync object on the first child node
      madness::detail::WorldPtr<LazySyncBase> child1_wptr_; ///< Pointer to the sync object on the second child node


      /// Construct a sync object

      /// This will store the world reference and sync key as well as
      /// determine the parent and children of this node.
      /// \param world The world where this sync object lives
      /// \param key The sync key
      LazySyncBase(madness::World& world, const keyT& key) :
        world_(world), key_(key), parent_(-1), child0_(-1), child1_(-1),
        parent_wptr_(), child0_wptr_(), child1_wptr_()
      {
        world.mpi.binary_tree_info(0, parent_, child0_, child1_);
      }

    private:
      // not allowed
      LazySyncBase(const LazySyncBase_&);
      LazySyncBase_& operator=(const LazySyncBase_&);

    public:

      /// Insert a \c LazySync object into the map

      /// A \c LazySync object is inserted into a map. If the object
      /// already exists, that object is returned.
      /// \tparam lsT The \c LazySync type
      /// \param[out] acc The object accessor.
      /// \param[in] world The world where the sync object lives
      /// \param[in] key The sync object key
      /// \return A pointer to the sync object
      /// \note The sync object will be write locked until the accessor is
      /// destroyed or released.
      template <typename lsT>
      static lsT* insert(typename mapT::accessor& acc, madness::World& world, const keyT& key) {
        if(map_.insert(acc, key))
          acc->second = new lsT(world, key);
        return static_cast<lsT*>(acc->second);
      }

      /// Remove sync object from the sync object map

      /// This function first acquires a write lock on this object to avoid
      /// possible race conditions.
      /// \param key The sync key of the sync object to be removed
      static void erase(const keyT& key) {
        typename mapT::accessor acc;
        map_.find(acc, key);
        map_.erase(acc);
      }

      /// Key accessor

      /// \return The key the key to this object
      const keyT& key() const { return key_; }

      /// World accessor

      /// \return A reference to the world where this sync object lives
      madness::World& get_world() const { return world_; }

      /// Root node query

      /// \return \c true when this is the root node.
      bool is_root() const { return parent_ == -1; }

      /// Children ready query

      /// This function checks that both children have been assigned or
      /// are non-existent
      /// \return \c true when all children have been initialized.
      bool children_ready() const {
        return (child0_wptr_ || (child0_ == -1)) // Check that child0 is ready (or non-existent)
            && (child1_wptr_ || (child1_ == -1)); // Check that child1 is ready (or non-existent)
      }

      /// Set the parent world pointer

      /// \param wptr Set the parent world pointer
      /// \throw TiledArray::Exception When the world pointer is null
      /// \throw TiledArray::Exception If the world pointer does not
      /// belong to the parent node
      void set_parent(const madness::detail::WorldPtr<LazySyncBase_>& wptr) {
        TA_ASSERT(wptr);
        TA_ASSERT(wptr.owner() == parent_);
        parent_wptr_ = wptr;
      }

      /// Set one of the child world pointers

      /// \param wptr A world pointer to one of the children
      /// \throw TiledArray::Exception When the world pointer is null
      /// \throw TiledArray::Exception If the world pointer does not
      /// belong to one of the child nodes
      void set_child(const madness::detail::WorldPtr<LazySyncBase_>& wptr) {
        TA_ASSERT(wptr);
        TA_ASSERT((wptr.owner() == child0_) || (wptr.owner() == child1_));
        if(wptr.owner() == child0_)
          child0_wptr_ = wptr;
        else
          child1_wptr_ = wptr;
      }

      /// World pointer factory function

      /// \return A world pointer to this object
      madness::detail::WorldPtr<LazySyncBase_> get_wptr() const {
        return madness::detail::WorldPtr<LazySyncBase_>(world_, const_cast<LazySyncBase_*>(this));
      }

      /// Execute the sync function on child nodes

      /// Send a task to the child nodes that will execute the sync
      /// function on that node and send tasks to its child nodes.
      template <typename fnT>
      void sync_children(fnT fn) const {
        TA_ASSERT(children_ready());
        if(child0_wptr_.has_owner())
          world_.taskq.add(child0_, fn, child0_wptr_, madness::TaskAttributes::hipri());
        if(child1_wptr_.has_owner())
          world_.taskq.add(child1_, fn, child1_wptr_, madness::TaskAttributes::hipri());
      }

      /// Initialize the sync objects on child the nodes

      /// Send an active message to the child nodes. The active message
      /// arguments are the sync key and a world pointer to this sync object.
      /// The message function needs to construct the sync object if it
      /// is not present and set the parent world pointer of the sync
      /// object.
      /// \tparam fnT The child initialization function type
      /// \param fn The function used to initialize the sync object on child nodes
      template <typename fnT>
      void init_children(fnT fn) const {
        if(child0_ != -1)
          world_.am.send(child0_, fn, madness::new_am_arg(key_, get_wptr()));
        if(child1_ != -1)
          world_.am.send(child1_, fn, madness::new_am_arg(key_, get_wptr()));
      }

      /// Initialize the sync object on the parent the node

      /// Send an active message to the parent node. The active message
      /// arguments are the sync key and a world pointer to this sync object.
      /// The message function needs to construct the sync object if it
      /// is not present and set the child world pointer of the sync
      /// object.
      /// \tparam fnT The parent initialization function type
      /// \param fn The function used to initialize the sync object on parent node
      template <typename fnT>
      void init_parent(fnT fn) const {
        TA_ASSERT(parent_ != -1);
        if(parent_wptr_)
          get_world().am.send(parent_, fn, new_am_arg(key_, get_wptr()));
      }

    }; // class LazySyncBase


    template <typename keyT>
    typename LazySyncBase<keyT>::mapT LazySyncBase<keyT>::map_;

    /// Asynchronous barrier operation

    /// After all objects have been constructed on all nodes, the object
    /// submits a task that runs an arbitrary sync function on all nodes.
    /// The sync operation object must have a default constructor,
    /// an assignment operator, and must be runnable (e.g. op() must be valid).
    /// \tparam keyT The sync key type
    /// \tparam opT The sync operation type
    template <typename keyT, typename opT>
    class LazySync : public LazySyncBase<keyT> {
    private:
      typedef LazySync<keyT, opT> LazySync_; ///< This object type
      typedef LazySyncBase<keyT> LazySyncBase_; ///< Base class type
      typedef typename LazySyncBase_::mapT::accessor accessor; ///< The sync obect accessor type

      opT op_; ///< The sync operation

      /// Wrapper function for the base class insert function

      /// \param acc The sync object accessor type
      /// \param world The world where the sync object lives
      /// \param key The sync key
      static LazySync_* insert(accessor& acc, madness::World& world, const keyT& key) {
        return LazySyncBase_::template insert<LazySync_>(acc, world, key);
      }

      /// Active message function that initializes a child sync object

      /// This task function creates a new task function if it does not
      /// already exist on the child node, and sets the parent world
      /// pointer. It also calls \c init_parent() , which continues the
      /// synchronization process. This function matches the requirements
      /// of the base class \c init_children() function.
      /// \param arg The active message arguments.
      static void init_child_handler(const madness::AmArg& arg) {
        keyT key;
        madness::detail::WorldPtr<LazySyncBase_> parent_wptr;
        arg & key & parent_wptr;

        accessor acc; // Scoped lock object
        LazySync_* p = insert(acc, * (arg.get_world()), key);

        // Set the parent pointer and initialize the parent node if all
        // the children are ready.
        p->set_parent(parent_wptr);
        p->init_parent();
      }

      /// Active message function that initializes a parent sync object

      /// This task function creates a new task function if it does not
      /// already exist on the parent node, and sets the child world
      /// pointer. It also calls \c init_parent() , which continues the
      /// synchronization process. This function matches the requirements
      /// of the base class \c init_parent() function.
      /// \param arg The active message arguments.
      static void init_parent_handler(const madness::AmArg& arg) {
        // Get the active message data
        keyT key;
        madness::detail::WorldPtr<LazySyncBase_> wptr;
        arg & key & wptr;

        accessor acc; // Scoped lock object
        LazySync_* p = insert(acc, * (arg.get_world()), key);

        // set the child and initialize the parent node if all the
        // children are ready
        p->set_child(wptr);
        p->init_parent();
      }

      /// Task function that calls the sync operation for this node and the childern

      /// This task function calls the sync function for this node and
      /// sends a task to the child nodes to do the same. It also does
      /// additional cleanup work.
      /// \param p The sync object world pointer for this node
      /// \throw TiledArray::Exception If \c p is not local
      static void sync_task(const madness::detail::WorldPtr<LazySyncBase_>& p) {
        TA_ASSERT(p.is_local());

        // Erase first to avoid possible race conditions
        LazySyncBase_::erase(p->key());

        // Get a pointer to this type
        LazySync_* ls = static_cast<LazySync_*>(p.get());

        // Execute sync operation on this node and its child nodes
        ls->sync_children(& sync_task);
        ls->op_();

        // Do cleanup
        delete ls;
      }

      /// Initialize the child world pointer of the parent object

      /// Send an active message to the parent sync object that sets the
      /// child pointer and continues the synchronization process.
      void init_parent() const {
        if(LazySyncBase_::children_ready()) { // True after all children have initialized this object
          if(LazySyncBase_::is_root()) {
            // This is the root node. When this condition is true,
            // all nodes have been fully initialized. It is now safe
            // to call the synchronization function on all nodes.
            LazySyncBase_::get_world().taskq.add(sync_task,
                LazySyncBase_::get_wptr(), madness::TaskAttributes::hipri());
          } else {
            // Initialize this node's parent with a world pointer to this object.
            LazySyncBase_::init_parent(init_parent_handler);
          }
        }
      }

      /// Initialize the child world pointer of the parent object

      /// Send an active message to the parent sync object that sets the
      /// child pointer and continues the synchronization process.
      void init_children() const {
        LazySyncBase_::init_children(init_child_handler);

        // Immediately try to initialize the parent because they could
        // already be initialized.
        init_parent();
      }

      // not allowed
      LazySync(const LazySync_&);
      LazySync_& operator=(const LazySync_&);

    public:

      /// Construct a lazy sync object

      /// \param world The world where the sync object lives
      /// \param key The sync key
      LazySync(madness::World& world, const keyT& key) :
        LazySyncBase_(world, key)
      { }

      /// Construct a lazy sync object

      /// Construct a lazy sync object that executes \c once all nodes
      /// have passed \c key sync point. \c op must define a default
      /// constructor, assignment operator, and op() must be a valid
      /// operation.
      /// \param world The world where the sync object lives
      /// \param key The sync key
      /// \param op The sync operation to be executed on this node
      static void make(madness::World& world, const keyT& key, opT op) {
        if(world.size() == 1) {
          op(); // No need to do more than run the sync object when
          // there is only one node
        } else {
          accessor acc;
          LazySync_* p = insert(acc, world, key);

          p->op_ = op;
          p->init_children();
        }
      }

    }; // class LazySync

  }  // namespace detail

  /// Lazy sync object factory function

  /// Construct a lazy sync object that executes \c once all nodes have passed
  /// \c key sync point. \c op must define a default constructor, assignment
  /// operator, and op() must be a valid operation.
  /// \param world The world where the sync object lives
  /// \param key The sync key
  /// \param op The sync operation to be executed on this node
  template <typename keyT, typename opT>
  void lazy_sync(madness::World& world, const keyT& key, const opT& op) {
    detail::LazySync<keyT, opT>::make(world, key, op);
  }


}  // namespace TiledArray

#endif // TILEDARRAY_LAZY_SYNC_H__INCLUDED
