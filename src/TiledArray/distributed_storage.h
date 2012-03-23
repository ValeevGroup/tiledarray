#ifndef TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
#define TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED

// This needs to be defined before world/worldreduce.h and world/worlddc.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/error.h>
#include <TiledArray/indexed_iterator.h>
#include <TiledArray/reduce_task.h>
#include <world/world.h>

namespace TiledArray {
  namespace detail {

    /// Lazy synchronization base

    /// This object handles general synchronization between nodes.
    /// \tparam Key The synchronization key type
    template <typename Key>
    class ReduceGroupBase {
    protected:
      typedef ReduceGroupBase<Key> ReduceGroupBase_; ///< This object type
      typedef madness::ConcurrentHashMap<Key, ReduceGroupBase_*> map_type; ///< LazySync object container

      static map_type map_; ///< Map to sync objects so they are accessible to
      ///< other nodes via active messages.

      madness::World& world_; ///< The world where the sync object lives
      Key key_; ///< The sync key
      std::vector<ProcessID> group_; ///< The group to world process id map
      ProcessID parent_; ///< The parent node of this node ( = -1 if this is the root node)
      ProcessID child0_; ///< The first child node of this node (=-1 if there is no first child node)
      ProcessID child1_; ///< The second child node of this node (=-1 if there is no second child node)

    private:
      void binary_tree_info(ProcessID root, ProcessID& parent, ProcessID& child0, ProcessID& child1) {
        const ProcessID size = group_.size();
        // Renumber processes so root has me=0
        const ProcessID me = (world_to_group(world_.rank()) + size - root) % size;
        parent = (((me - 1) >> 1) + root) % size; // Parent in binary tree
        child0 = (me << 1) + 1 + root; // Left child
        child1 = (me << 1) + 2 + root; // Right child
        if (child0 >= size && child0 < (size + root)) child0 -= size;
        if (child1 >= size && child1 < (size + root)) child1 -= size;

        if (me == 0) parent = -1;
        if (child0 >= size) child0 = -1;
        if (child1 >= size) child1 = -1;
      }


      /// Convert group process id to a world process id.

      /// \param[in,out] proc The process id to be converted.
      ProcessID group_to_world(ProcessID proc) {
        TA_ASSERT(proc >= -1);
        TA_ASSERT(proc < group_.size());

        const ProcessID world_proc = (proc != -1 ? group_[proc] : -1);
        TA_ASSERT(world_proc < world_.size());
        return world_proc;
      }

      ProcessID world_to_group(ProcessID proc) {
        TA_ASSERT(proc < world_.rank());
        const ProcessID group_proc = (proc != -1 ?
            std::distance(group_.begin(), std::find(group_.begin(), group_.end(), proc)) :
            -1);
        TA_ASSERT(group_proc < group_.size());
        return group_proc;
      }

    protected:

      /// Construct a reduce group object

      /// This will store the world reference and sync key as well as
      /// determine the parent and children of this node.
      /// \param world The world where this sync object lives
      /// \param key The sync key
      ReduceGroupBase(madness::World& world, const Key& key, const std::vector<ProcessID>& group, ProcessID root) :
        world_(world), key_(key), group_(group), parent_(-1), child0_(-1), child1_(-1)
      {
        // Get the parent and children in the binary tree
        binary_tree_info(world_to_group(root), parent_, child0_, child1_);

        map_group(parent_);
        map_group(child0_);
        map_group(child1_);
      }

    private:
      // not allowed
      ReduceGroupBase(const ReduceGroupBase_&);
      ReduceGroupBase_& operator=(const ReduceGroupBase_&);

    public:

      /// Insert a \c ReduceGroup object into the map

      /// A \c LazySync object is inserted into a map. If the object
      /// already exists, that object is returned.
      /// \tparam lsT The \c LazySync type
      /// \param[out] acc The object accessor.
      /// \param[in] world The world where the sync object lives
      /// \param[in] key The sync object key
      /// \return A pointer to the sync object
      /// \note The sync object will be write locked until the accessor is
      /// destroyed or released.
      template <typename rgT>
      static rgT* insert(typename map_type::accessor& acc, madness::World& world,
          const Key& key, const std::vector<ProcessID>& group, ProcessID root,
          const typename rgT::op_type& op)
      {
        if(map_.insert(acc, key))
          acc->second = new rgT(world, key, group, root, op);
        return static_cast<rgT*>(acc->second);
      }

      /// Remove sync object from the sync object map

      /// This function first acquires a write lock on this object to avoid
      /// possible race conditions.
      /// \param key The sync key of the sync object to be removed
      static void erase(const Key& key) {
        typename map_type::accessor acc;
        map_.find(acc, key);
        map_.erase(acc);
      }

      /// Key accessor

      /// \return The key the key to this object
      const Key& key() const { return key_; }

      /// World accessor

      /// \return A reference to the world where this sync object lives
      madness::World& get_world() const { return world_; }

      /// Root node query

      /// \return \c true when this is the root node.
      bool is_root() const { return parent_ == -1; }

      template <typename Fn>
      void send_parent(Fn fn, madness::AmArg* args) const {
        TA_ASSERT(parent_ != -1);
        get_world().am.send(parent_, fn, args);
      }

    }; // class ReduceGroupBase


    template <typename Key>
    madness::ConcurrentHashMap<Key, ReduceGroupBase<Key>*> ReduceGroupBase<Key>::map_;

    template <typename Key, typename Op>
    class ReduceGroup : public ReduceGroupBase<Key> {
    private:
      typedef ReduceGroupBase<Key> ReduceGroupBase_;

      ReduceTask<Op> local_reduce_;

      static madness::Void send_parent_handler(const madness::AmArg& args) {

      }

    public:

      /// Construct a reduce group object

      /// This will store the world reference and sync key as well as
      /// determine the parent and children of this node.
      /// \param world The world where this sync object lives
      /// \param key The sync key
      ReduceGroup(madness::World& world, const Key& key, const std::vector<ProcessID>& group, ProcessID root, const Op& op) :
        ReduceGroupBase_(world, key, group, root), local_reduce_(world, op)
      { }

    }; // class ReduceGroup

    /// Distributed storage container.

    /// Each element in this container is owned by a single node, but any node
    /// may request a copy of the element in the form of a \c madness::Future .
    /// The owner of each element is defined by a process map (pmap), which is
    /// passed to the constructor.
    /// Elements do not need to be explicitly initialized because they will be
    /// added to the container when the element is first accessed, though you
    /// may manually initialize the an element with the \c insert() function.
    /// All elements are stored in \c madness::Future , which may be set only
    /// once.
    /// \note This object is derived from \c madness::WorldObject , which means
    /// the order of construction of object must be the same on all nodes. This
    /// can easily be achieved by only constructing world objects in the main
    /// thread. DO NOT construct world objects within tasks where the order of
    /// execution is nondeterministic.
    template <typename T>
    class DistributedStorage : public madness::WorldObject<DistributedStorage<T> > {
    public:
      typedef DistributedStorage<T> DistributedStorage_;
      typedef madness::WorldObject<DistributedStorage_> WorldObject_;

      typedef std::size_t size_type;
      typedef size_type key_type;
      typedef T value_type;
      typedef madness::Future<value_type> future;
      typedef madness::WorldDCPmapInterface<key_type> pmap_interface;

      typedef madness::ConcurrentHashMap<key_type, future> container_type;
      typedef detail::IndexedIterator<typename container_type::iterator> iterator; ///< Local tile iterator
      typedef detail::IndexedIterator<typename container_type::const_iterator> const_iterator; ///< Local tile const iterator
      typedef typename container_type::accessor accessor;
      typedef typename container_type::const_accessor const_accessor;

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
      /// \param max_size The maximum capacity of this container.
      /// \param pmap The process map for the container.
      /// \param do_pending Process pending messages for the container when this
      /// is true. If it is false, it is up to the user to explicitly call
      /// \c process_pending(). [default = true]
      DistributedStorage(madness::World& world, size_type max_size,
          const std::shared_ptr<pmap_interface>& pmap, bool do_pending = true) :
        WorldObject_(world), max_size_(max_size), pmap_(pmap),
        data_((max_size / world.size()) + 11)
      {
        TA_ASSERT(pmap_);
        if(do_pending)
          process_pending();
      }

      virtual ~DistributedStorage() { }

      /// Process pending messages

      /// Process any messages that arrived before this object was constructed
      /// locally.
      /// \note No incoming messages are processed until this routine is invoked.
      /// It can be invoked in the constructor by passing \c true to the
      /// \c do_pending argument.
      void process_pending() { WorldObject_::process_pending(); }

      /// World accessor

      /// \return A reference to the world this object is associated with.
      /// \throw nothing
      madness::World& get_world() const { return WorldObject_::get_world(); }

      /// Process map accessor

      /// \return A shared pointer to the process map.
      /// \throw nothing
      const std::shared_ptr<pmap_interface>& get_pmap() const { return pmap_; }

      /// Begin iterator factory

      /// \return An iterator that points to the first local data element
      /// \note Iterates over local data only and involve no communication.
      iterator begin() { return iterator(data_.begin()); }

      /// Begin const iterator factory

      /// \return A const iterator that points to the first local data element
      /// \note Iterates over local data only and involve no communication.
      const_iterator begin() const { return const_iterator(data_.begin()); }

      /// End iterator factory

      /// \return An iterator that points to the end of the local data elements
      /// \note Iterates over local data only and involve no communication.
      iterator end() { return iterator(data_.end()); }

      /// End const iterator factory

      /// \return A const iterator that points to the end of the local data elements
      /// \note Iterates over local data only and involve no communication.
      const_iterator end() const { return const_iterator(data_.end()); }

      /// Element owner

      /// \return The process that owns element \c i
      ProcessID owner(size_type i) const {
        TA_ASSERT(i < max_size_);
        return pmap_->owner(i);
      }

      /// Local element quary

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
        if(is_local(i)) {
          const_accessor acc;
          return data_.insert(acc, i);
        }

        WorldObject_::send(owner(i), & DistributedStorage_::remote_insert, i);
        return false;
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
          accessor acc;
          if(! data_.insert(acc, typename container_type::datumT(i, f))) {
            // The element was already in the container.
            future existing_f = acc->second;
            acc.release();
            existing_f.set(f);
          }
        } else {
          if(f.probe()) {
            set_value(i, f.get());
          } else {
            DelayedSet* set_callback = new DelayedSet(*this, i, f);
            const_cast<future&>(f).register_callback(set_callback);
          }
        }
      }

      template <typename InIter, typename Op>
      void reduce(size_type i, std::vector<ProcessID> procs, Op op) {

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
          const_accessor acc;
          data_.insert(acc, i);
          return acc->second;
        }

        future result;
#ifdef MADNESS_USE_BSEND_ACKS
        WorldObject_::send(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()));
#else
        WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()), madness::TaskAttributes::hipri());
#endif // MADNESS_USE_BSEND_ACKS

        return result;
      }

      /// Apply a function to a list of data elements

      /// This function will initiate a task that will run \c op with each
      /// element listed in \c indices on the owner's node. Op is distributed
      /// with in a binary tree. \c op must be callable and take \c size_type
      /// and as its arguments \c value_type . It must also be serializable.
      /// Example:
      /// \code
      /// class DistributedOp {
      /// public:
      ///   DistributedOp() {
      ///     // ...
      ///   }
      ///
      ///   DistributedOp& operator=(const DistributedOp& other) {
      ///     // ...
      ///   }
      ///
      ///   void operator()(size_type index, value_type value) const {
      ///     // ...
      ///   }
      ///
      ///   template <typename Archive>
      ///   void serialize(const Archive&) {
      ///     // ...
      ///   }
      /// }; // struct DistributedOp
      /// \endcode
      /// \tparam Op The operation type
      /// \param indices A list of indices that \c op will be applied to.
      /// \param op The operation that will be run on the data of \c indices
      template <typename Op>
      void apply(const std::vector<size_type>& indices, const Op& op) {
        // Construct a map of nodes to indices
        node_map_type node_map;
        for(std::vector<size_type>::const_iterator it = indices.begin(); it !=indices.end(); ++it)
          node_map.insert(std::pair<const ProcessID, size_type>(owner(*it), *it));

        // Compile a list of nodes that own the target indices
        std::vector<ProcessID> nodes;
        for(node_map_type::const_iterator it = node_map.begin(); it != node_map.end(); it = node_map.upper_bound(it->first))
          nodes.push_back(it->first);

        // Submit the task(s) to apply op to data of indices
        const ProcessID rank = WorldObject_::get_world().rank();
        WorldObject_::task(rank, & DistributedStorage_::template submit_tasks<Op>,
            nodes, rank, node_map, op, madness::TaskAttributes::hipri());
      }

    private:

      typedef std::multimap<ProcessID, size_type> node_map_type;

      /// Task function that runs a distributed task

      /// \tparam Op The operation object to be run on the data element
      /// \param i The element index
      /// \param value The value of element \c i
      /// \param op The operation to be run
      /// \return madness::None
      template <typename Op>
      madness::Void apply_task(size_type i, value_type value, const Op& op) const {
        op(i, value);

        return madness::None;
      }

      /// Get binary tree info

      /// \param me This process node
      /// \param size Then number of nodes
      /// \param root The node that the tree starts from
      /// \param[out] child0 The left child node of \c me
      /// \param[out] child1 The right child node of \c me
      void binary_tree_info(ProcessID me, ProcessID size, ProcessID root, ProcessID& child0, ProcessID& child1) {
        // Renumber processes so root has me=0
        me = (me + size - root) % size;

        // Left child
        child0 = (me << 1) + 1 + root;
        if (child0 >= size && child0 < (size + root))
          child0 -= size;
        if (child0 >= size)
          child0 = -1;

        // Right child
        child1 = (me << 1) + 2 + root;
        if (child1 >= size && child1 < (size + root))
          child1 -= size;
        if (child1 >= size)
          child1 = -1;
      }

      template <typename Op>
      madness::Void submit_tasks(const std::vector<ProcessID>& nodes, ProcessID root,
          const std::multimap<ProcessID, size_type>& node_map, const Op& op)
      {
        // Get rank
        const ProcessID rank = WorldObject_::get_world().rank();

        // Get the child nodes that will run the following tasks
        ProcessID child0 = -1;
        ProcessID child1 = -1;
        binary_tree_info(rank, nodes.size(), root, child0, child1);

        // Submit task on child nodes
        if(child0 != -1)
          WorldObject_::task(nodes[child0], & DistributedStorage_::template submit_tasks<Op>,
              nodes, root, node_map, op, madness::TaskAttributes::hipri());
        if(child1 != -1)
          WorldObject_::task(nodes[child1], & DistributedStorage_::template submit_tasks<Op>,
              nodes, root, node_map, op, madness::TaskAttributes::hipri());

        // Submit a local task for each local element.
        const node_map_type::const_iterator end = node_map.upper_bound(rank);
        for(node_map_type::const_iterator it = node_map.lower_bound(rank); it != end; ++it)
          WorldObject_::task(rank, & DistributedStorage_::template apply_task<Op>,
              it->second, operator[](it->second), op);

        return madness::None;
      }

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
      madness::Void set_value(size_type i, const value_type& value) {
        if(is_local(i)) {
          const_accessor acc;
          data_.insert(acc, i);
          future f = acc->second;
          acc.release();
          f.set(value);
        } else {
          WorldObject_::send(owner(i), & DistributedStorage_::set_value, i, value);
        }

        return madness::None;
      }

      /// Remote insert without a return message.
      madness::Void remote_insert(size_type i) {
        TA_ASSERT(is_local(i));
        const_accessor acc;
        data_.insert(acc, i);

        return madness::None;
      }

      static void find_return(const typename future::remote_refT& ref, const value_type& value) {
        future result(ref);
        result.set(value);
      }

      struct DelayedReturn : public madness::CallbackInterface {
      private:
        typename future::remote_refT ref_;
        future future_;

      public:

        DelayedReturn(const typename future::remote_refT& ref, const future& fut) :
            ref_(ref), future_(fut)
        { }

        ~DelayedReturn() { }

        virtual void notify() {
          DistributedStorage_::find_return(ref_, future_);
          delete this;
        }
      }; // struct DelayedReturn

      /// Handles find request
      madness::Void find_handler(size_type i, const typename future::remote_refT& ref) const {
        TA_ASSERT(is_local(i));
        const_accessor acc;
        data_.insert(acc, i);
        future f = acc->second;
        acc.release();
        if(f.probe()) {
          find_return(ref, f);
        } else {
          DelayedReturn* return_callback = new DelayedReturn(ref, f);
          f.register_callback(return_callback);
        }

        return madness::None;
      }

      const size_type max_size_;
      std::shared_ptr<pmap_interface> pmap_;
      mutable container_type data_;
    };


  }  // namespace detail
}  // namespace TiledArray

namespace madness {
  namespace archive {

    // Forward declarations
    template <typename, typename>
    struct ArchiveLoadImpl;
    template <typename, typename>
    struct ArchiveStoreImpl;

    /// Archive load for a std::multimap
    template <typename Archive, typename Key, typename T>
    struct ArchiveLoadImpl<Archive, std::multimap<Key, T> > {
      static inline void load(const Archive& ar, std::multimap<Key, T>& mmap) {
        // Clear the map of any existing values
        mmap.clear();

        std::size_t size = 0ul, n = 0ul, i = 0ul;
        Key key;

        // Get the number of elements in the map
        ar & size;

        // iterate over the data elements
        while(i < size) {
          // Get the key and the number of elements for that key
          ar & key & n;

          // Get the values for the key
          typename std::multimap<Key, T>::value_type value(key, T());
          for(std::size_t j = 0; j < n; ++i, ++j) {
            ar & value.second;
            mmap.insert(value);
          }
        }


      }
    }; // struct ArchiveLoadImpl<Archive, std::shared_ptr<TiledArray::detail::VectorTask> >

    /// Archive store for a std::multimap
    template <typename Archive, typename Key, typename T>
    struct ArchiveStoreImpl<Archive, std::multimap<Key, T> > {
      static inline void store(const Archive& ar, const std::multimap<Key, T>& mmap) {

        // Store the number of elements in the map
        ar & mmap.size();
        typename std::multimap<Key, T>::const_iterator it = mmap.begin();
        while(it != mmap.end()) {
          // Get an iterator to the end of the first key
          typename std::multimap<Key, T>::const_iterator end = mmap.upper_bound(it->first);

          // Store the key and the number of elements associated with the key
          ar & it->first & std::distance(it, end);

          // Store each data element associated with the key
          for(; it != end; ++it)
            ar & it->second;
        }
      }
    }; // struct ArchiveStoreImpl<Archive, std::shared_ptr<TiledArray::detail::VectorTask> >

  } // namespace archive
} // namespace madness

#endif // TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
