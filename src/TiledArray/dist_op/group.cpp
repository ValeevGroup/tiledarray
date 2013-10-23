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
 *  group.cpp
 *  Oct 22, 2013
 *
 */

#include <TiledArray/dist_op/group.h>
#include <TiledArray/dist_op/lazy_sync.h>

namespace TiledArray {
  namespace dist_op {


    namespace {

      typedef madness::ConcurrentHashMap<DistributedID, madness::Future<Group> >
          group_registry_container;

      group_registry_container group_registry;

      class GroupSyncKey {
      private:
        DistributedID did_;

      public:
        GroupSyncKey() : did_(madness::uniqueidT(), 0ul) { }

        GroupSyncKey(const DistributedID& did) : did_(did) { }

        GroupSyncKey(const GroupSyncKey& other) : did_(other.did_) { }

        GroupSyncKey& operator=(const GroupSyncKey& other) {
          did_ = other.did_;
          return *this;
        }

        bool operator==(const GroupSyncKey& other) const { return did_ == other.did_; }
        bool operator!=(const GroupSyncKey& other) const { return did_ != other.did_; }

        template <typename Archive>
        void serialize(const Archive& ar) {
          ar & did_;
        }

        friend madness::hashT hash_value(const GroupSyncKey& key) {
          return std::hash_value(key.did_);
        }
      }; // class GroupSyncKey

      class UnregisterGroup {
      private:
        DistributedID did_;

      public:
        UnregisterGroup() : did_(madness::uniqueidT(), 0ul) { }

        UnregisterGroup(const DistributedID& did) : did_(did) { }

        UnregisterGroup(const UnregisterGroup& other) : did_(other.did_) { }

        UnregisterGroup& operator=(const UnregisterGroup& other) {
          did_ = other.did_;
          return *this;
        }

        void operator()() const {
          group_registry_container::accessor acc;
          group_registry.find(acc, did_);
          group_registry.erase(acc);
        }

      }; // class UnregisterGroup

    } // namespace


    /// Register a group for use with distributed algorithms

    /// \param group The group to be registered
    /// \throw TiledArray::Exception When the group is already in the registry
    void register_group(const Group& group) {
      group_registry_container::accessor acc;
      if(! group_registry.insert(acc, group_registry_container::datumT(group.id(),
          madness::Future<Group>(group))))
      {
        TA_ASSERT(! acc->second.probe());
        acc->second.set(group);
      }
    }

    /// Get a registered group

    /// This function is used to acquire the group in an active message handler.
    /// \param did The id associated with the group
    /// \return A future to the group
    madness::Future<Group> get_group(const DistributedID& did) {
      group_registry_container::accessor acc;
      if(group_registry.insert(acc, group_registry_container::datumT(did,
          madness::Future<Group>::default_initializer())))
      {
        acc->second = madness::Future<Group>();
      }

      return acc->second;
    }

    /// Remove the given group from the registry

    /// Groups are removed via a lazy sync operation, which will only remove the
    /// group from the registry once it has been unregistered on all nodes.
    /// param
    void unregister_group(const Group& group) {
      lazy_sync(group, GroupSyncKey(group.id()), UnregisterGroup(group.id()));
    }

  }  // namespace dist_op
} // namespace TiledArray
