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
 *  Justus
 *  Department of Chemistry, Virginia Tech
 *
 *  cache.h
 *  Oct 13, 2013
 *
 */

#ifndef TILEDARRAY_CACHE_H__INCLUDED
#define TILEDARRAY_CACHE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/dist_op/distributed_id.h>

namespace TiledArray {
  namespace dist_op {
    namespace detail {

      /// Distributed caching utility

      /// This object implements a caching mechanism, where one process can send
      /// data to another without synchronization. Specific data is matched by
      /// by using a multikey-value system. Keys may or may not be associated
      /// with \c WorldObjects (via unique object identifiers).
      class DistCache {
      private:

        // Forward declarations
        class Cache;
        template <typename> class CacheData;

        /// The container that contains cache data
        typedef madness::ConcurrentHashMap<DistributedID, Cache*> cache_container;
        /// Cache container datum type
        typedef cache_container::datumT datum_type;

        static cache_container caches_; ///< Cache container

        mutable madness::World* world_; ///< World where cache data is sent/received

        /// Cache interface class

        /// This base class is used to access derived class data
        class Cache {
        public:

          /// Virtual destructor
          virtual ~Cache() { }

          /// Cache data accessor

          /// \tparam T The cached data type
          /// \return A const reference to the cached future
          template <typename T>
          const madness::Future<T>& get() const {
            TA_ASSERT(this->get_type_info() == typeid(CacheData<T>));
            return static_cast<const CacheData<T>*>(this)->data();
          }

        private:

          /// Typeid accessor of the derived class

          /// \return The std::type_info of the derived class
          virtual const std::type_info& get_type_info() const = 0;

        }; // class Cache

        /// Cache data container

        /// \tparam T The data type stored in the cache
        template <typename T>
        class CacheData : public Cache {
        private:
          madness::Future<T> data_; ///< Local cached data

        public:

          /// Default constructor
          CacheData() : data_() { }

          /// Constructor with future initialization
          CacheData(const madness::Future<T>& data) : data_(data) { }

          /// Constructor with data initialization
          CacheData(const T& data) : data_(data) { }

          /// Virtual destructor
          virtual ~CacheData() { }

          /// Data accessor

          /// \return A const reference to the data
          const madness::Future<T>& data() const { return data_; }

        private:

          /// Typeid accessor of the derived class

          /// \return The std::type_info of the derived class
          virtual const std::type_info& get_type_info() const { return typeid(CacheData<T>); }

        }; // class CacheData


        /// Set the cache data accosted with \c did

        /// \tparam T The object type that will be used to set the cache (may be
        /// a \c madness::Future type).
        /// \param did The distributed id associated with \c value
        /// \param value The data that will be used to set the cache
        template <typename T>
        static void set_cache_data(const DistributedID& did, const T& value) {
          typedef typename madness::remove_future<T>::type value_type;

          // Retrieve the cached future
          cache_container::accessor acc;
          if(caches_.insert(acc, datum_type(did, NULL))) {

            // A new element was inserted, so create a new cache object.
            acc->second = new CacheData<value_type>(value);
            acc.release();

          } else {

            // The element already existed, so retrieve the data
            Cache* cache = acc->second;
            caches_.erase(acc);

            // Set the cache value
            madness::Future<value_type> f = cache->get<value_type>();
            TA_ASSERT(! f.probe());
            f.set(value);

            // Cleanup cache
            delete cache;
          }
        }

        /// Delayed send callback object

        /// This callback object is used to send local data to a remove process
        /// once it has been set.
        /// \tparam T The type of data to be sent
        template <typename T>
        class DelayedSend : public madness::CallbackInterface {
        private:
          madness::World& world_; ///< The communication world
          const ProcessID dest_; ///< The destination process id
          const DistributedID did_; ///< The distributed id associated with \c value_
          madness::Future<T> value_; ///< The data to be sent

          // Not allowed
          DelayedSend(const DelayedSend<T>&);
          DelayedSend<T>& operator=(const DelayedSend<T>&);

        public:

          /// Constructor

          /// \param ds The distributed container that owns element i
          /// \param i The element to be moved
          DelayedSend(madness::World& world, const ProcessID dest,
              const DistributedID& did, const madness::Future<T>& value) :
            world_(world), dest_(dest), did_(did), value_(value)
          { }

          virtual ~DelayedSend() { }

          /// Notify this object that the future has been set.

          /// This will set the value of the future on the remote node and delete
          /// this callback object.
          virtual void notify() {
            TA_ASSERT(value_.probe());
            DistCache::send(world_, dest_, did_, value_.get());
            delete this;
          }
        }; // class DelayedSend

      public:

        /// Constructor

        /// \param world The world that will be used to send/receive messages
        DistCache(madness::World& world) : world_(&world) { }

        /// Copy constructor

        /// \param other The object to be copied
        DistCache(const DistCache& other) : world_(other.world_) { }

        /// Copy assignment operator

        /// \param other The object to be copied
        /// \return A reference to this object
        DistCache& operator=(const DistCache& other) {
          world_ = other.world_;
          return *this;
        }

        /// Receive data from remote node

        /// \tparam T The data type stored in cache
        /// \param did The distributed ID
        /// \return A future to the data
        template <typename T>
        static madness::Future<T> recv(const DistributedID& did) {
          // Retrieve the cached future
          cache_container::accessor acc;
          if(caches_.insert(acc, datum_type(did, NULL))) {

            // A new element was inserted, so create a new cache object.
            acc->second = new CacheData<T>();
            madness::Future<T> result(acc->second->get<T>());
            acc.release();

            return result;
          } else {

            // The element already existed, so retrieve the data
            Cache* cache = acc->second;
            caches_.erase(acc);

            // Get the result
            madness::Future<T> result(cache->get<T>());
            delete cache;

            return result;
          }
        }

        /// Receive data from remote node

        /// \tparam T The data type stored in cache
        /// \param key The key that is associated with the data
        /// \return A future to the data
        template <typename T>
        static madness::Future<T> recv(const std::size_t key) {
          return recv<T>(DistributedID(madness::uniqueidT(), key));
        }

        /// Receive data from remote node

        /// \tparam T The data type stored in cache
        /// \param id The unique object id associated with the data
        /// \param key The key that is associated with the data
        /// \return A future to the data
        template <typename T>
        static madness::Future<T> recv(const madness::uniqueidT& id, const std::size_t key) {
          return recv<T>(DistributedID(id, key));
        }

        /// Send value to \c dest

        /// \tparam T The value type
        /// \param world The world that will be used to send the value
        /// \param dest The node where the data will be sent
        /// \param did The distributed id that is associatied with the data
        /// \param value The data to be sent
        template <typename T>
        static typename madness::disable_if<madness::is_future<T> >::type
        send(madness::World& world, const ProcessID dest, const DistributedID& did,
            const T& value)
        {
          if(world.rank() == dest) {
            // When dest is this process, skip the task and set the future immediately.
            set_cache_data(did, value);
          } else {
            // Spawn a remote task to set the value
            world.taskq.add(dest, DistCache::template set_cache_data<T>, did, value,
                madness::TaskAttributes::hipri());
          }
        }

        /// Send \c value to \c dest

        /// \tparam T The value type
        /// \param world The world that will be used to send the value
        /// \param dest The node where the data will be sent
        /// \param did The distributed id that is associated with the data
        /// \param value The data to be sent
        template <typename T>
        static void send(madness::World& world, ProcessID dest,
            const DistributedID& did, const madness::Future<T>& value)
        {
          if(world.rank() == dest) {
            set_cache_data(did, value);
          } else {
            // The destination is not this node, so send it to the destination.
            if(value.probe()) {
              // Spawn a remote task to set the value
              world.taskq.add(dest, DistCache::template set_cache_data<T>, did,
                  value.get(), madness::TaskAttributes::hipri());
            } else {
              // The future is not ready, so create a callback object that will
              // send value to the destination node when it is ready.
              DelayedSend<T>* delayed_send_callback =
                  new DelayedSend<T>(world, dest, did, value);
              const_cast<madness::Future<T>&>(value).register_callback(delayed_send_callback);

            }
          }
        }

        /// Send \c value to \c dest

        /// \tparam T The value type
        /// \param world The world that will be used to send the value
        /// \param dest The node where the data will be sent
        /// \param key The key that is associated with the data
        /// \param value The data to be sent
        template <typename T>
        static void send(madness::World& world, ProcessID dest,
            const std::size_t key, const T& value)
        {
          send(world, dest, DistributedID(madness::uniqueidT(), key), value);
        }

        /// Send \c value to \c dest

        /// \tparam T The value type
        /// \param world The world that will be used to send the value
        /// \param dest The node where the data will be sent
        /// \param id The unique object id associated with the data
        /// \param key The key that is associated with the data
        /// \param value The data to be sent
        template <typename T>
        static void send(madness::World& world, ProcessID dest,
            const madness::uniqueidT& id, const std::size_t key, const T& value)
        {
          send(world, dest, DistributedID(id, key), value);
        }

        /// Send \c value to \c dest

        /// \tparam T The value type
        /// \param world The world that will be used to send the value
        /// \param dest The node where the data will be sent
        /// \param did The distributed id that is associated with the data
        /// \param value The data to be sent
        template <typename T>
        void send(ProcessID dest, const DistributedID& did, const T& value) {
          send(*world_, dest, did, value);
        }

        /// Send \c value to \c dest

        /// \tparam T The value type
        /// \param world The world that will be used to send the value
        /// \param dest The node where the data will be sent
        /// \param key The key that is associated with the data
        /// \param value The data to be sent
        template <typename T>
        void send(ProcessID dest, const std::size_t key, const T& value) {
          send(*world_, dest, DistributedID(madness::uniqueidT(), key), value);
        }

        /// Send \c value to \c dest

        /// \tparam T The value type
        /// \param dest The node where the data will be sent
        /// \param id The unique object id associated with the data
        /// \param key The key that is associated with the data
        /// \param value The data to be sent
        template <typename T>
        void send(ProcessID dest, const madness::uniqueidT& id,
            const std::size_t key, const T& value)
        {
          send(*world_, dest, DistributedID(id, key), value);
        }

      }; // class DistCache

    } // namespace detail
  }  // namespace dist_op
} // namespace TiledArray

#endif // TILEDARRAY_CACHE_H__INCLUDED
