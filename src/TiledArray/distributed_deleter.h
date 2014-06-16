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

#include <TiledArray/madness.h>

namespace TiledArray {
  namespace detail {

    /// Distributed deleter

    /// This deleter object will cleanup some of the local memory in a
    /// distributed object by calling the member function clear via a lazy sync
    /// point. Deletion of the object is deferred to the next global sync point.
    /// \c T must define the member functions clear, id, and get_world.
    /// \tparam T The type of pointer that will be deleted
    /// \tparam Deleter The deleter type [ default = \c void(*)(T*) ]
    template <typename T, typename Deleter = void(*)(T*)>
    class DistributedDeleter {

      Deleter deleter_; ///< The pointer deleter function/functor

      /// Lazy cleaner object

      /// This object is used as the operation to lazy_sync and will be called
      /// after all nodes have called lazy_sync for the given object.
      class LazyDelete {
        T* t_; ///< A pointer to the distributed object object
        Deleter deleter_;

      public:
        /// Default constructor
        LazyDelete() : t_(NULL), deleter_() { }

        /// Constructor

        /// \param t The object to be deleted
        /// \param d The deleter function
        LazyDelete(T* t, const Deleter& d) :
          t_(t), deleter_(d)
        { }

        /// Copy constructor

        /// \param other The object to be copied
        LazyDelete(const LazyDelete& other) :
          t_(other.t_), deleter_(other.deleter_)
        { }

        /// Assignment operator

        /// \param other The object to be copied
        /// \return A reference to this object
        LazyDelete& operator=(const LazyDelete& other) {
          t_ = other.t_;
          deleter_ = other.deleter_;
          return *this;
        }

        /// Do memory cleanup
        void operator()() const {
          deleter_(t_);
        }
      }; // class LazyDelete

      /// Construct a default deleter for a function pointer
      template <typename D>
      static typename madness::enable_if<std::is_same<D, void(*)(T*)>, D>::type
      default_deleter() { return & madness::detail::checked_delete<T>; }

      /// Construct a default deleter for a functor
      template <typename D>
      static typename madness::disable_if<std::is_same<D, void(*)(T*)>, D>::type
      default_deleter() { return D(); }

      struct DistributedDeleterTag { };

    public:

      /// Constructor

      /// \param world The world where the distributed object will live
      DistributedDeleter() :
        deleter_(default_deleter<Deleter>())
      { }

      /// Constructor

      /// \param world The world where the distributed object will live
      DistributedDeleter(const Deleter& deleter) :
        deleter_(deleter)
      { }

      DistributedDeleter(const DistributedDeleter<T, Deleter>& other) :
        deleter_(other.deleter_)
      { }

      /// Assignment operator

      /// \param other The object to be copied
      /// \return A reference to this object
      DistributedDeleter<T, Deleter>&
      operator=(const DistributedDeleter<T, Deleter>& other) {
        deleter_ = other.deleter_;
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
        if(madness::initialized())
          t->get_world().gop.lazy_sync(
              madness::TaggedKey<madness::uniqueidT, DistributedDeleterTag>(t->id()),
              LazyDelete(t, deleter_));
        else
          deleter_(t);
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
      return std::shared_ptr<T>(p, DistributedDeleter<T>());
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
      return std::shared_ptr<T>(p, DistributedDeleter<T, D>(d));
    }


  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_DISTRIBUTED_DELETER_H__INCLUDED
