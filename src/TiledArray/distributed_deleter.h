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
 *  distributed_deleter.h
 *  Sep 26, 2013
 *
 */

#ifndef TILEDARRAY_DISTRIBUTED_DELETER_H__INCLUDED
#define TILEDARRAY_DISTRIBUTED_DELETER_H__INCLUDED

#include <TiledArray/lazy_sync.h>

namespace TiledArray {
  namespace detail {

    /// Distributed deleter

    /// This deleter object will cleanup some of the local memory in a
    /// distributed object by calling the member function clear via a lazy sync
    /// point. Deletion of the object is deferred to the next global sync point.
    /// \c T must define the member functions clear, id, and get_world.
    /// \tparam T The type of pointer that will be deleted
    /// \tparam D The deleter type [ default = \c void(*)(T*) ]
    template <typename T, typename D = void(*)(T*)>
    class DistributedDeleter : private madness::DeferredDeleter<T, D>
    {
      /// The base class type
      typedef madness::DeferredDeleter<T, D> deferred_deleter_type;

      /// Lazy cleaner object

      /// This object is used as the operation to lazy_sync and will be called
      /// after all nodes have finished
      class LazyCleaner {
        T* t_; ///< A pointer to the distributed object object

      public:
        /// Default constructor
        LazyCleaner() : t_(NULL) { }

        /// Constructor

        /// \param t The object to be cleaned
        LazyCleaner(T* t) : t_(t) { }

        /// Copy constructor

        /// \param other The object to be copied
        LazyCleaner(const LazyCleaner& other) : t_(other.t_) { }

        /// Assignment operator

        /// \param other The object to be copied
        /// \return A reference to this object
        LazyCleaner& operator=(const LazyCleaner& other) {
          t_ = other.t_;
          return *this;
        }

        /// Do memory cleanup
        void operator()() const {
          TA_ASSERT(t_);
          t_->clear();
        }
      }; // class LazyClear

    public:

      /// Constructor

      /// \param world The world where the distributed object will live
      DistributedDeleter(madness::World& world) :
        deferred_deleter_type(world)
      { }

      DistributedDeleter(const DistributedDeleter<T, D>& other) :
        deferred_deleter_type(other)
      { }

      /// Assignment operator

      /// \param other The object to be copied
      /// \return A reference to this object
      DistributedDeleter<T, D>& operator=(const DistributedDeleter<T, D>& other) {
        deferred_deleter_type::operator=(other);
        return *this;
      }

      /// Deletion operation

      /// This function will cleanup local data and place the pointer in the
      /// MADNESS deferred deletion queue. Local data is cleared via the clear
      /// member function. The pointer will be deleted at the
      /// next global sync point.
      /// \param t The pointer to be deleted
      void operator()(T* t) {
        TA_ASSERT(t);
        lazy_sync(t->get_world(), t->id(), LazyCleaner(t));
        madness::DeferredDeleter<T>::operator()(t);
      }
    }; // class DistributedDeleter

    /// Create a shared pointer to a \c DistributedStorage object.

    /// This will construct a new shared pointer for a distributed object that
    /// has a distributed deleter.
    /// \tparam T The type of the distributed object
    /// \param p The distributed object pointer
    /// \return A shared pointer to p
    template <typename T>
    inline std::shared_ptr<T> make_distributed_shared_ptr(T* p) {
      return std::shared_ptr<T>(p, DistributedDeleter<T>(p->get_world()));
    }

    /// Create a shared pointer to a \c DistributedStorage object.

    /// This will construct a new shared pointer for a distributed object that
    /// has a distributed deleter.
    /// \tparam T The type of the distributed object
    /// \tparam D The type of the deleter object
    /// \param p The distributed object pointer
    /// \return A shared pointer to p
    template <typename T, typename D>
    inline std::shared_ptr<T> make_distributed_shared_ptr(T* p, const D& d) {
      TA_ASSERT(p);
      return std::shared_ptr<T>(p, DistributedDeleter<T, D>(p->get_world(), d));
    }


  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_DISTRIBUTED_DELETER_H__INCLUDED
