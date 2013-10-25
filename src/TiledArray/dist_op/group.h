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
 *  group.h
 *  Oct 13, 2013
 *
 */

#ifndef TILEDARRAY_DIST_EVAL_GROUP_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_GROUP_H__INCLUDED

#include <TiledArray/dist_op/distributed_id.h>
#include <TiledArray/error.h>
#include <TiledArray/utility.h>

namespace TiledArray {
  namespace dist_op {

    class Group {
    private:

      class Impl {
      private:
        madness::World& world_; ///< Parent world for this group
        DistributedID did_; ///< Group id
        std::vector<ProcessID> group_to_world_map_; ///< List of nodes in the group
        ProcessID group_rank_; ///< The group rank of this process

      public:
        /// Constructor

        /// \tparam A An std compliant array (e.g. \c std::array or <tt>std::vector</tt>)
        /// \param world The world that is the basis for this group
        /// \param did The distributed id associated with this group
        /// \param group An array of Processes in world
        template <typename A>
        Impl(madness::World& world, DistributedID did, const A& group) :
          world_(world), did_(did),
          group_to_world_map_(TiledArray::detail::begin(group),
          TiledArray::detail::end(group)), group_rank_(-1)
        {
          // Check that there is at least one process in group
          TA_ASSERT(detail::size(group) > 0ul);

          // Sort and remove duplicates from group
          std::sort(group_to_world_map_.begin(), group_to_world_map_.end());
          group_to_world_map_.erase(std::unique(group_to_world_map_.begin(),
              group_to_world_map_.end()), group_to_world_map_.end());

          // Check that all processes in the group map are contained by world
          TA_ASSERT(group_to_world_map_.front() >= 0);
          TA_ASSERT(group_to_world_map_.back() < world_.size());

          // Get the group rank for this process
          group_rank_ = rank(world_.rank());

          // Check that this process is in the group
          TA_ASSERT(group_rank_ != group_to_world_map_.size());
        }

        /// Parent world accessor

        /// \return A reference to the parent world of this group
        madness::World& get_world() const { return world_; }

        /// Group id accessor

        /// \return A const reference to the group id
        const DistributedID& id() const { return did_; }

        /// Group rank accessor

        /// \return The rank of this process in the group
        ProcessID rank() const { return group_rank_; }

        /// Map world rank to group rank

        /// \param world_rank The world rank to be mapped
        /// \return The group rank of \c world_rank when it is a member of this
        /// group, otherwise \c -1.
        ProcessID rank(const ProcessID world_rank) const {
          ProcessID result = std::distance(group_to_world_map_.begin(),
              std::find(group_to_world_map_.begin(), group_to_world_map_.end(),
              world_rank));
          if(result == group_to_world_map_.size())
            result = -1;
          return result;
        }

        /// Group size accessor

        /// \return The number of processes in the group
        ProcessID size() const { return group_to_world_map_.size(); }

        /// Map group rank to world rank

        /// \return The rank of this process in the world
        ProcessID world_rank(const ProcessID group_rank) const {
          TA_ASSERT(group_rank >= 0);
          TA_ASSERT(group_rank < group_to_world_map_.size());
          return group_to_world_map_[group_rank];
        }

        /// Compute the binary tree parents and children

        /// \param[out] parent The parent node of the binary tree
        /// \param[out] child1 The left child node of the binary tree
        /// \param[out] child2 The right child node of the binary tree
        /// \param[in] group_root The head node of the binary tree
        void make_tree(ProcessID& parent, ProcessID& child1,
            ProcessID& child2, const ProcessID group_root) const
        {
          const ProcessID group_size = group_to_world_map_.size();

          // Check that root is in the range of the group
          TA_ASSERT(group_root >= 0);
          TA_ASSERT(group_root < group_size);

          // Renumber processes so root has me == 0
          const ProcessID me = (group_rank_ + group_size - group_root) % group_size;

          // Compute the group parent
          parent = (me == 0 ? -1 : group_to_world_map_[(((me - 1) >> 1) + group_root) % group_size]);

          // Compute children
          child1 = (me << 1) + 1 + group_root;
          child2 = child1 + 1;

          const ProcessID end = group_size + group_root;
          if(child1 < end)
            child1 = group_to_world_map_[child1 % group_size];
          else
            child1 = -1;
          if(child2 < end)
            child2 = group_to_world_map_[child2 % group_size];
          else
            child2 = -1;
        }
      }; // struct Impl


      std::shared_ptr<Impl> pimpl_;

    public:

      /// Default constructor

      /// Create an empty group
      Group() : pimpl_() { }

      /// Copy constructor

      /// \param other The group to be copied
      /// \note Copy is shallow.
      Group(const Group& other) : pimpl_(other.pimpl_) { }

      /// Create a new group

      /// \tparam A An array type
      /// \param world The parent world for this group
      /// \param did The distributed id associated with this group
      /// \param group An array with a list of process to be included in the
      /// group.
      /// \note All processes in the \c group list must be included in the
      /// parent world.
      template <typename A>
      Group(madness::World& world, const DistributedID& did, const A& group) :
        pimpl_(new Impl(world, did, group))
      { }

      /// Copy assignment operator

      /// \param other The group to be copied
      /// \note Copy is shallow.
      Group& operator=(const Group& other) {
        pimpl_ = other.pimpl_;
        return *this;
      }

      /// Register a group

      /// Register a group so that it can be used in active messages and tasks
      /// spawned on remote nodes.
      /// \param group The group to be registered
      /// \throw TiledArray::Exception When the group is empty
      /// \throw TiledArray::Exception When the group is already in the registry
      void register_group() const;

      /// Remove the given group from the registry

      /// Groups are removed via a lazy sync operation, which will only remove the
      /// group from the registry once unregistered has been called on all processes
      /// in the group.
      /// \param group The group to be removed from the registry
      void unregister_group() const;

      /// Get a registered group

      /// This function is used to acquire the group in an active message handler.
      /// \param did The id associated with the group
      /// \return A future to the group
      static madness::Future<Group> get_group(const DistributedID& did);

      /// Quary empty group

      /// \return \c true when this group is empty
      bool empty() const { return !pimpl_; }

      /// Group id accessor

      /// \return A const reference to the group id
      const DistributedID& id() const {
        TA_ASSERT(pimpl_);
        return pimpl_->id();
      }

      /// Parent world accessor

      /// \return A reference to the parent world of this group
      madness::World& get_world() const {
        TA_ASSERT(pimpl_);
        return pimpl_->get_world();
      }

      /// Group rank accessor

      /// \return The rank of this process in the group
      ProcessID rank() const {
        TA_ASSERT(pimpl_);
        return pimpl_->rank();
      }

      /// Map world rank to group rank

      /// \param world_rank The world rank to be mapped
      /// \return The rank of \c world_rank process in the group
      ProcessID rank(const ProcessID world_rank) const {
        TA_ASSERT(pimpl_);
        return pimpl_->rank(world_rank);
      }

      /// Group size accessor

      /// \return The number of processes in the group
      ProcessID size() const {
        return (pimpl_ ? pimpl_->size() : 0);
      }

      /// Map group rank to world rank

      /// \param group_rank The group rank to be mapped to a world rank
      /// \return The parent world rank of group_rank.
      ProcessID world_rank(const ProcessID group_rank) const {
        TA_ASSERT(pimpl_);
        return pimpl_->world_rank(group_rank);
      }

      /// Compute the binary tree parents and children

      /// \param[out] parent The parent node of the binary tree
      /// \param[out] child1 The left child node of the binary tree
      /// \param[out] child2 The right child node of the binary tree
      /// \param[in] root The head node of the binary tree in the group [default = 0]
      /// \note Output ranks are in the parent world.
      void make_tree(ProcessID& parent, ProcessID& child1,
          ProcessID& child2, const ProcessID group_root = 0) const
      {
        TA_ASSERT(pimpl_);
        pimpl_->make_tree(parent, child1, child2, group_root);
      }

      template <typename Archive>
      void serialize(const Archive&) {
        TA_ASSERT(false); // not allowed
      }
    }; // class Group

  }  // namespace dist_op
} // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_GROUP_H__INCLUDED
