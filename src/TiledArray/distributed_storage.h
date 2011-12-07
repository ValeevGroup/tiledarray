#ifndef TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
#define TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED

// This needs to be defined before world/worldreduce.h and world/worlddc.h
#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <TiledArray/error.h>
#include <TiledArray/indexed_iterator.h>
//#include <world/world.h>
#include <world/worlddc.h>
#include <world/worldreduce.h>

namespace TiledArray {
  namespace detail {

    template <typename T>
    class DistributedStorage : public madness::WorldReduce<DistributedStorage<T> > {
    public:
      typedef DistributedStorage<T> DistributedStorage_;
      typedef madness::WorldReduce<DistributedStorage_> WorldReduce_;
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
      /// \c process_pending().
      DistributedStorage(madness::World& world, size_type max_size,
          const std::shared_ptr<pmap_interface>& pmap, bool do_pending = true) :
        WorldReduce_(world), max_size_(max_size), pmap_(pmap),
        data_(max_size / world.size() + 1)
      {
        if(do_pending)
          process_pending();
      }

      virtual ~DistributedStorage() { }

      /// Process pending messages

      /// Process any messages that arrived before this object was constructed
      /// locally.
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
      /// \note Iterates over local data only
      iterator begin() { return iterator(data_.begin()); }

      /// Begin const iterator factory

      /// \return A const iterator that points to the first local data element
      /// \note Iterates over local data only
      const_iterator begin() const { return const_iterator(data_.begin()); }

      /// End iterator factory

      /// \return An iterator that points to the end of the local data elements
      /// \note Iterates over local data only
      iterator end() { return iterator(data_.end()); }

      /// End const iterator factory

      /// \return A const iterator that points to the end of the local data elements
      /// \note Iterates over local data only
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
      void clear() { data_.clear(); }

      /// Number of local elements

      /// No communication.
      /// \return The number of local elements stored by the container.
      size_type size() const { return data_.size(); }

      /// Max size accessor

      /// The maximum size is the total number of elements that can be held by
      /// this container on all nodes, not on each node.
      /// \return The maximum number of elements.
      size_type max_size() const { return max_size_; }

      /// Insert element

      /// Add element \c i to the container. If it is not local, a message is
      /// sent to the owner to insert it.
      /// \param i The element to be inserted.
      /// \return true if the tile was inserted locally, otherwise false.
      bool insert(size_type i) {
        if(is_local(i)) {
          const_accessor acc;
          return data_.insert(acc, i);
        }

        WorldObject_::task(owner(i), & DistributedStorage_::remote_insert, i,
            madness::TaskAttributes::hipri());
        return false;
      }
//
//      /// Insert LOCAL element and acquire a read lock
//
//      /// Insert element at \c i . If the element already exists, the \c acc is
//      /// set to point at the existing element.
//      /// \param[out] acc Read-lock element accessor
//      /// \param i The element to access
//      /// \return \c true if a new element was inserted, otherwise false.
//      bool insert(const_accessor& acc, size_type i) {
//        TA_ASSERT(is_local(i));
//        return data_.insert(acc, i);
//      }
//
//      /// Insert LOCAL element and acquire a write lock
//
//      /// Insert element at \c i . If the element already exists, the \c acc is
//      /// set to point at the existing element.
//      /// \param[out] acc Write-lock element accessor
//      /// \param i The element to access
//      /// \return \c true if a new element was inserted, otherwise false.
//      bool insert(accessor& acc, size_type i) {
//        TA_ASSERT(is_local(i));
//        return data_.insert(acc, i);
//      }
//
//      /// Insert LOCAL element and acquire a read lock
//
//      /// Insert element at \c i . If the element already exists, the \c acc is
//      /// set to point at the existing element.
//      /// \param[out] acc Read-lock element accessor
//      /// \param i The element to access
//      /// \param f The future that will or is holding the element datum.
//      /// \return \c true if a new element was inserted, otherwise false.
//      bool insert(accessor& acc, size_type i, const future& f) {
//        TA_ASSERT(is_local(i));
//        return data_.insert(acc, typename container_type::datumT(i, f));
//      }
//
//      /// Insert LOCAL element and acquire a write lock
//
//      /// Insert element at \c i . If the element already exists, the \c acc is
//      /// set to point at the existing element.
//      /// \param[out] acc Write-lock element accessor
//      /// \param i The element to access
//      /// \param f The future that will or is holding the element datum.
//      /// \return \c true if a new element was inserted, otherwise false.
//      bool insert(const_accessor& acc, size_type i, const future& f) {
//        TA_ASSERT(is_local(i));
//        return data_.insert(acc, typename container_type::datumT(i, f));
//      }
//
      void set(size_type i, const value_type& value) {
        TA_ASSERT(i < max_size_);
        set_value(i, value);
      }

      void set(size_type i, const future& f) {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          accessor acc;
          if(! data_.insert(acc, typename container_type::datumT(i, f)))
            acc->second.set(f);
        } else {
          WorldObject_::task(get_world().rank(), & DistributedStorage_::set_value,
              i, f, madness::TaskAttributes::hipri());
        }
      }

      bool find(const_accessor& acc, size_type i) const {
        TA_ASSERT(i < max_size_);
        return data_.find(acc, i);
      }

      bool find(accessor& acc, size_type i) {
        TA_ASSERT(i < max_size_);
        return data_.find(acc, i);
      }

      future operator[](size_type i) const {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          const_accessor acc;
          if(find(acc, i))
            return acc->second;
          else
            return future(value_type());
        }

        future result;
        WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()));

        return result;
      }

      future operator[](size_type i) {
        TA_ASSERT(i < max_size_);
        if(is_local(i)) {
          const_accessor acc;
          data_.insert(acc, i);
          return acc->second;
        }

        future result;
        WorldObject_::task(owner(i), & DistributedStorage_::find_handler, i,
            result.remote_ref(get_world()));

        return result;
      }

    private:

      madness::Void set_value(size_type i, const value_type& value) {
        if(is_local(i)) {
          accessor acc;
          data_.insert(acc, i);
          acc->second.set(value);
        } else {
          WorldObject_::task(owner(i), & DistributedStorage_::set_value, i, value, madness::TaskAttributes::hipri());
        }

        return madness::None;
      }

      madness::Void remote_insert(size_type i) {
        TA_ASSERT(is_local(i));
        const_accessor acc;
        data_.insert(acc, i);

        return madness::None;
      }

      static madness::Void find_return(const typename future::remote_refT& ref, const value_type& value) {
        future result(ref);
        result.set(value);

        return madness::None;
      }

      /// Handles find request
      madness::Void find_handler(size_type i, const typename future::remote_refT& ref) const {
        TA_ASSERT(is_local(i));
        const_accessor acc;
        if(data_.find(acc, i)) {
          if(acc->second.probe())
            find_return(ref, acc->second);
          else
            get_world().taskq.add(& DistributedStorage_::find_return, ref, acc->second,
                madness::TaskAttributes::hipri());
        } else {
          find_return(ref, value_type());
        }

        return madness::None;
      }

      const size_type max_size_;
      std::shared_ptr<pmap_interface> pmap_;
      container_type data_;
    };


  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DISTRIBUTED_STORAGE_H__INCLUDED
