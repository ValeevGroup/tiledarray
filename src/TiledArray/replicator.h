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

#ifndef TILEDARRAY_REPLICATOR_H__INCLUDED
#define TILEDARRAY_REPLICATOR_H__INCLUDED

#include <TiledArray/external/madness.h>

namespace TiledArray {
namespace detail {

/// Replicate a \c Array object

/// This object will create a replicated \c Array from a distributed
/// \c Array.
/// \tparam A The array type
/// Homeworld = M7R-227
template <typename A>
class Replicator : public madness::WorldObject<Replicator<A> >,
                   private madness::Spinlock {
 private:
  typedef Replicator<A> Replicator_;  ///< This object type
  typedef madness::WorldObject<Replicator_>
      wobj_type;  ///< The base object type
  typedef std::stack<madness::CallbackInterface*,
                     std::vector<madness::CallbackInterface*> >
      callback_type;  ///< Callback interface

  A destination_;  ///< The replicated array
  std::vector<typename A::ordinal_type>
      indices_;  ///< List of local tile indices
  std::vector<Future<typename A::value_type> > data_;  ///< List of local tiles
  madness::AtomicInt sent_;  ///< The number of nodes the data has been sent to
  World& world_;
  volatile callback_type callbacks_;  ///< A callback stack
  volatile mutable bool probe_;       ///< Cache for local data probe

  /// \note Assume object is already locked
  void do_callbacks() {
    callback_type& callbacks = const_cast<callback_type&>(callbacks_);
    while (!callbacks.empty()) {
      callbacks.top()->notify();
      callbacks.pop();
    }
  }

  /// Task that will call send when all local tiles are ready to be sent
  class DelaySend : public madness::TaskInterface {
   private:
    Replicator_& parent_;  ///< The parent replicator operation

   public:
    /// Constructor
    DelaySend(Replicator_& parent)
        : madness::TaskInterface(madness::TaskAttributes::hipri()),
          parent_(parent) {
      typename std::vector<Future<typename A::value_type> >::iterator it =
          parent_.data_.begin();
      typename std::vector<Future<typename A::value_type> >::iterator end =
          parent_.data_.end();
      for (; it != end; ++it) {
        if (!it->probe()) {
          madness::DependencyInterface::inc();
          it->register_callback(this);
        }
      }
    }

    /// Virtual destructor
    virtual ~DelaySend() {}

    /// Task send task function
    virtual void run(const madness::TaskThreadEnv&) { parent_.send(); }

  };  // class DelaySend

  /// Probe all local data has been set

  /// \return \c true when all local tiles have been set
  bool probe() const {
    madness::ScopedMutex<madness::Spinlock> locker(this);

    if (!probe_) {
      typename std::vector<Future<typename A::value_type> >::const_iterator it =
          data_.begin();
      typename std::vector<Future<typename A::value_type> >::const_iterator
          end = data_.end();
      for (; it != end; ++it)
        if (!it->probe()) break;

      probe_ = (it == end);
    }

    return probe_;
  }

  /// Send data to the next node when it is ready
  void delay_send() {
    if (probe()) {
      // The data is ready so send it now.
      send();  // Replication is done
    } else {
      // The local data is not ready to be sent, so create a task that will
      // send it when it is ready.
      DelaySend* delay_send_task = new DelaySend(*this);
      world_.taskq.add(delay_send_task);
    }
  }

  /// Send all local data to the next node
  void send() {
    const long sent = ++sent_;
    const ProcessID dest = (world_.rank() + sent) % world_.size();

    if (dest != world_.rank()) {
      wobj_type::task(dest, &Replicator_::send_handler, indices_, data_,
                      madness::TaskAttributes::hipri());
    } else
      do_callbacks();  // Replication is done
  }

  void send_handler(const std::vector<typename A::ordinal_type>& indices,
                    const std::vector<Future<typename A::value_type> >& data) {
    typename std::vector<typename A::ordinal_type>::const_iterator index_it =
        indices.begin();
    typename std::vector<Future<typename A::value_type> >::const_iterator
        data_it = data.begin();
    typename std::vector<Future<typename A::value_type> >::const_iterator
        data_end = data.end();

    for (; data_it != data_end; ++data_it, ++index_it)
      destination_.set(*index_it, data_it->get());

    delay_send();
  }

 public:
  Replicator(const A& source, const A destination)
      : wobj_type(source.world()),
        madness::Spinlock(),
        destination_(destination),
        indices_(),
        data_(),
        sent_(),
        world_(source.world()),
        callbacks_(),
        probe_(false) {
    sent_ = 0;

    // Generate a list of local tiles from other.
    typename A::pmap_interface::const_iterator end = source.pmap()->end();
    typename A::pmap_interface::const_iterator it = source.pmap()->begin();
    indices_.reserve(source.pmap()->local_size());
    data_.reserve(source.pmap()->local_size());
    if (source.is_dense()) {
      // When dense, all tiles are present
      for (; it != end; ++it) {
        indices_.push_back(*it);
        data_.push_back(source.find(*it));
        destination_.set(*it, data_.back());
      }
    } else {
      // When sparse, we need to generate a list
      for (; it != end; ++it)
        if (!source.is_zero(*it)) {
          indices_.push_back(*it);
          data_.push_back(source.find(*it));
          destination_.set(*it, data_.back());
        }
    }

    /// Send the data to the first node
    delay_send();

    // Process any pending messages
    wobj_type::process_pending();
  }

  /// Check that the replication is complete

  /// \return \c true when all data has been transferred.
  bool done() {
    madness::ScopedMutex<madness::Spinlock> locker(this);
    return sent_ == world_.size();
  }

  /// Add a callback

  /// The callback is called when the local data has been sent to all
  /// nodes. If the data has already been sent to all nodes, the callback
  /// is notified immediately.
  /// \param callback The callback object
  void register_callback(madness::CallbackInterface* callback) {
    madness::ScopedMutex<madness::Spinlock> locker(this);
    if (sent_ == world_.size())
      callback->notify();
    else
      const_cast<callback_type&>(callbacks_).push(callback);
  }

};  // class Replicator

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_REPLICATOR_H__INCLUDED
