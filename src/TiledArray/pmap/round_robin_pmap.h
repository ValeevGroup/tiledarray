//
// Created by Karl Pierce on 7/26/20.
//

#ifndef TILEDARRAY_PMAP_ROUND_ROBIN_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_ROUND_ROBIN_PMAP_H__INCLUDED

#include <TiledArray/pmap/pmap.h>

namespace TiledArray {
  namespace detail {

/// A blocked process map

/// Map N elements among P processes into blocks that are approximately N/P
/// elements in size. A minimum block size may also be specified.
    class RoundRobinPmap : public Pmap {
    protected:
      // Import Pmap protected variables
      using Pmap::procs_;  ///< The number of processes
      using Pmap::rank_;   ///< The rank of this process
      using Pmap::size_;   ///< The number of tiles mapped among all processes

    private:
      const size_type remainder_;          ///< tile remainder (= size_ % procs_)

    public:
      typedef Pmap::size_type size_type;  ///< Key type

      /// Construct Blocked map

      /// \param world The world where the tiles will be mapped
      /// \param size The number of tiles to be mapped
      RoundRobinPmap(World& world, size_type size)
              : Pmap(world, size),
                remainder_(size_ % procs_) {
        auto num_tiles_per_proc = size / procs_;
        if (remainder_ == 0 || rank_ >= remainder_)
          this->local_size_ = num_tiles_per_proc;
        else
          this->local_size_ =  num_tiles_per_proc + 1;
      }

      virtual ~RoundRobinPmap() {}

      /// Maps \c tile to the processor that owns it

      /// \param tile The tile to be queried
      /// \return Processor that logically owns \c tile
      virtual size_type owner(const size_type tile) const {
        TA_ASSERT(tile < size_);
        return (tile % procs_);
      }

      /// Check that the tile is owned by this process

      /// \param tile The tile to be checked
      /// \return \c true if \c tile is owned by this process, otherwise \c false .
      virtual bool is_local(const size_type tile) const {
        return (tile % procs_ == rank_);
      }

    };  // class BlockedPmap

  }  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_PMAP_ROUND_ROBIN_PMAP_H__INCLUDED
