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

    /// Distributed caching utility

    /// This object implements a caching mechanism, where one process can send
    /// data to another without synchronization. Specific data is matched by
    /// by using a multikey-value system. Keys may or may not be associated
    /// with \c WorldObjects (via unique object identifiers).
    template <typename Key>
    class DistCache {
    private:

      // Forward declarations
      class Cache;
      template <typename> class CacheData;

      /// The container that contains cache data
      typedef madness::ConcurrentHashMap<Key, Cache*> cache_container;
      /// Cache container datum type
      typedef typename cache_container::datumT datum_type;

      static cache_container caches_; ///< Cache container

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

    public:

      /// Set the cache data accosted with \c did

      /// \tparam T The object type that will be used to set the cache (may be
      /// a \c madness::Future type).
      /// \param did The distributed id associated with \c value
      /// \param value The data that will be used to set the cache
      template <typename T>
      static void set_cache_data(const Key& key, const T& value) {
        typedef typename madness::remove_future<T>::type value_type;

        // Retrieve the cached future
        typename cache_container::accessor acc;
        if(caches_.insert(acc, datum_type(key, NULL))) {

          // A new element was inserted, so create a new cache object.
          acc->second = new CacheData<value_type>(value);
          acc.release();

        } else {

          // The element already existed, so retrieve the data
          Cache* cache = acc->second;
          caches_.erase(acc);

          // Set the cache value
          madness::Future<value_type> f = cache->template get<value_type>();
          TA_ASSERT(! f.probe());
          f.set(value);

          // Cleanup cache
          delete cache;
        }
      }

      template <typename T>
      static void get_cache_data(const Key& key, madness::Future<T>& data) {
        // Retrieve the cached future
        typename cache_container::accessor acc;
        if(caches_.insert(acc, datum_type(key, NULL))) {
          // A new element was inserted, so create a new cache object.
          acc->second = new CacheData<T>(data);
          acc.release();
        } else {

          // The element already existed, so retrieve the data
          Cache* cache = acc->second;
          caches_.erase(acc);

          // Get the result
          data.set(cache->template get<T>());
          delete cache;
        }
      }

      template <typename T>
      static madness::Future<T> get_cache_data(const Key& key) {
        // Retrieve the cached future
        typename cache_container::accessor acc;
        if(caches_.insert(acc, datum_type(key, NULL))) {

          // A new element was inserted, so create a new cache object.
          acc->second = new CacheData<T>();
          madness::Future<T> result(acc->second->template get<T>());
          acc.release();

          return result;
        } else {

          // The element already existed, so retrieve the data
          Cache* cache = acc->second;
          caches_.erase(acc);

          // Get the result
          madness::Future<T> result(cache->template get<T>());
          delete cache;

          return result;
        }
      }

    }; // class DistCache

    template <typename Key>
    typename DistCache<Key>::cache_container DistCache<Key>::caches_;

  }  // namespace dist_op
} // namespace TiledArray

#endif // TILEDARRAY_CACHE_H__INCLUDED
