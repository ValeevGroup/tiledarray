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

    } // namespace


    void Group::UnregisterGroup::operator()() const {
      group_registry_container::accessor acc;
      group_registry.find(acc, did_);
      Group group = acc->second;
      group_registry.erase(acc);
      group.pimpl_->set_register_status(false);
    }


    /// Register a group

    /// Register a group so that it can be used in active messages and tasks
    /// spawned on remote nodes.
    /// \param group The group to be registered
    /// \throw TiledArray::Exception When the group is empty
    /// \throw TiledArray::Exception When the group is already in the registry
    void Group::register_group() const {
      TA_ASSERT(pimpl_);
      group_registry_container::accessor acc;
      if(! group_registry.insert(acc, group_registry_container::datumT(id(),
          madness::Future<Group>(*this))))
      {
        TA_ASSERT(! acc->second.probe());
        acc->second.set(*this);
      }

      pimpl_->set_register_status(true);
    }

    /// Remove the given group from the registry

    /// Groups are removed via a lazy sync operation, which will only remove the
    /// group from the registry once unregistered has been called on all processes
    /// in the group.
    /// \param group The group to be removed from the registry
    void Group::unregister_group() const {
      TA_ASSERT(pimpl_);
      dist_op::LazySync<GroupSyncKey, UnregisterGroup>::make(*this,
          GroupSyncKey(pimpl_->id()), UnregisterGroup(pimpl_->id()));
    }

    /// Get a registered group

    /// This function is used to acquire the group in an active message handler.
    /// \param did The id associated with the group
    /// \return A future to the group
    madness::Future<Group> Group::get_group(const DistributedID& did) {
      group_registry_container::accessor acc;
      if(group_registry.insert(acc, group_registry_container::datumT(did,
          madness::Future<Group>::default_initializer())))
      {
        acc->second = madness::Future<Group>();
      }

      return acc->second;
    }

  }  // namespace dist_op
} // namespace TiledArray
