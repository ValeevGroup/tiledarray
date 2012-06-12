#ifndef TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
#define TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED

// This needs to be defined before world/worldreduce.h and world/worlddc.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/error.h>
#include <TiledArray/indexed_iterator.h>
#include <TiledArray/pmap.h>
#include <world/world.h>

namespace TiledArray {
  namespace detail {

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
      typedef Pmap<key_type> pmap_interface;

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
      DistributedStorage(madness::World& world, size_type max_size, const std::shared_ptr<pmap_interface>& pmap = std::shared_ptr<pmap_interface>()) :
        WorldObject_(world), max_size_(max_size),
        pmap_(pmap),
        data_((max_size / world.size()) + 11)
      {
        if(pmap_) {
          pmap_->set_seed(WorldObject_::id().get_obj_id());
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
        TA_ASSERT(pmap);
        TA_ASSERT(!pmap_);
        pmap_ = pmap;
        pmap_->set_seed(WorldObject_::id().get_obj_id());
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
        TA_ASSERT(pmap_);
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

      /// Insert an empty future into all locally owned elements
      void insert_local() {
        typename pmap_interface::const_iterator end = pmap_->end();
        for(typename pmap_interface::const_iterator it = pmap_->begin(); it != end; ++it) {
          const_accessor acc;
          data_.insert(acc, *it);
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
          if(pred(*it)) {
            const_accessor acc;
            data_.insert(acc, *it);
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
        WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()), false, madness::TaskAttributes::hipri());

        return result;
      }

      /// The caller takes ownership of element \c i

      /// Ownership can only take place after the element has been set. If the
      /// element has not been set when this function is called, then the
      /// transfer of ownership will be delayed (and handled automatically) until
      /// the element is set.
      /// \note it is the caller's responsibility to ensure that move is only
      /// call once. Otherwise, the program will hang.
      future move(size_type i) {
        TA_ASSERT(i < max_size_);

        future result;

        if(is_local(i)) {
          // Get element i
          const_accessor acc;
          data_.insert(acc, i);
          result = acc->second;

          // Remove the element from this
          if(result.probe())
            data_.erase(acc);
          else
            result.register_callback(new DelayedMove(*this, i));
        } else {
          WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
              result.remote_ref(get_world()), true, madness::TaskAttributes::hipri());
        }

        return result;
      }

    private:

      struct DelayedMove : public madness::CallbackInterface {
      private:
        DistributedStorage_& ds_; ///< A reference to the owning object
        size_type index_; ///< The index that will own the future

      public:

        DelayedMove(const DistributedStorage_& ds, size_type i) :
            ds_(const_cast<DistributedStorage_&>(ds)), index_(i)
        { }

        virtual ~DelayedMove() { }

        /// Notify this object when the future is set.

        /// This will set the value of the future on the remote node and delete
        /// this callback object.
        virtual void notify() {
          ds_.data_.erase(index_);
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
        DistributedStorage_& ds_;
        size_type index_;
        typename future::remote_refT ref_;
        future future_;
        bool move_;

      public:

        DelayedReturn(const DistributedStorage_& ds, size_type index, const typename future::remote_refT& ref, const future& fut, bool move) :
            ds_(const_cast<DistributedStorage_&>(ds)), index_(index), ref_(ref), future_(fut), move_(move)
        { }

        ~DelayedReturn() { }

        virtual void notify() {
          DistributedStorage_::find_return(ref_, future_);
          if(move_)
            ds_.data_.erase(index_);
          delete this;
        }
      }; // struct DelayedReturn

      /// Handles find request
      madness::Void find_handler(size_type i, const typename future::remote_refT& ref, bool mover) const {
        TA_ASSERT(is_local(i));
        const_accessor acc;
        data_.insert(acc, i);
        future f = acc->second;
        if(f.probe()) {
          if(mover)
            data_.erase(acc);
          find_return(ref, f);
        } else {
          acc.release();
          DelayedReturn* return_callback = new DelayedReturn(*this, i, ref, f, mover);
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

#endif // TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
