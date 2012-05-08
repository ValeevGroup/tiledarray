#ifndef TILEDARRAY_BCAST_GROUP_H__INCLUDED
#define TILEDARRAY_BCAST_GROUP_H__INCLUDED

#include <TiledArray/error.h>
#include <world/world.h>
#include <vector>
#include <algorithm>

namespace TiledArray {
  namespace detail {

    /// Broadcast and execute an object.

    /// This object will broadcast an executable object to a process group, and
    /// execute that object on each node.
    class BcastGroup {
    public:

      BcastGroup(madness::World& world, const std::vector<ProcessID>& group) :
          world_(world), group_(group), group_rank_(-1), child0_(-1), child1_(-1)
      {
        // Sort and remove duplicates
        std::sort(group_.begin(), group_.end());
        group_.resize(std::distance(group_.begin(), std::unique(group_.begin(), group_.end())));


        // Get children in binary tree
        group_rank_ = std::distance(group_.begin(),
            std::find(group_.begin(), group_.end(), world_.rank()));

        // Make sure this node is in the group.
        TA_ASSERT(group_rank_ < group.size());

        binary_tree(child0_, child1_, group_rank_, group.size(), group_rank_);
      }

      template <typename Op>
      void operator()(const Op& op) const {
        if(child0_ != -1)
          world_.taskq.add(group_[child0_], & BcastGroup::template bcast<Op>,
              world_.id(), child0_, group_rank_, group_, op,
              madness::TaskAttributes::hipri());
        if(child1_ != -1)
          world_.taskq.add(group_[child1_], & BcastGroup::template bcast<Op>,
              world_.id(), child1_, group_rank_, group_, op,
              madness::TaskAttributes::hipri());

        op();
      }

    private:

      static void binary_tree(ProcessID& child0, ProcessID& child1,
          const ProcessID rank, const ProcessID size, const ProcessID root)
      {
        // Renumber processes so root has me=0
        int me = (rank + size - root) % size;

        // Left child
        child0 = (me << 1) + 1 + root;
        if((child0 >= size) && (child0 < (size + root)))
          child0 -= size;
        if(child0 >= size)
          child0 = -1;

        // Right child
        child1 = (me << 1) + 2 + root;
        if((child1 >= size) && (child1 < (size + root)))
          child1 -= size;
        if(child1 >= size)
          child1 = -1;
      }

      template <typename Op>
      static madness::Void bcast(const unsigned long world_id, const ProcessID rank,
          const ProcessID root, const std::vector<ProcessID>& group, const Op& op)
      {
        madness::World* world = madness::World::world_from_id(world_id);
        ProcessID child0 = -1;
        ProcessID child1 = -1;
        binary_tree(child0, child1, rank, group.size(), root);
        if(child0 != -1)
          world->taskq.add(group[child0], & BcastGroup::template bcast<Op>,
              world_id, child0, root, group, op, madness::TaskAttributes::hipri());
        if(child1 != -1)
          world->taskq.add(group[child1], & BcastGroup::template bcast<Op>,
              world_id, child1, root, group, op, madness::TaskAttributes::hipri());

        op();

        return madness::None;
      }


      madness::World& world_;
      std::vector<ProcessID> group_;
      ProcessID group_rank_;
      ProcessID child0_;
      ProcessID child1_;
    }; // class BcastGroup

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_BCAST_GROUP_H__INCLUDED
