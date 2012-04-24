#ifndef TILEDARRAY_BLOCKED_PMAP_H__INCLUDED
#define TILEDARRAY_BLOCKED_PMAP_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/pmap.h>
#include <world/world.h>
#include <algorithm>

namespace TiledArray {
  namespace detail {

    /// A blocked process map

    /// Map N elements among P processes into blocks that are approximately N/P
    /// elements in size. A minimum block size may also be specified.
    class BlockedPmap : public Pmap<std::size_t> {
    public:
      typedef Pmap<std::size_t>::key_type key_type; ///< Key type
      typedef Pmap<std::size_t>::const_iterator const_iterator;

      /// Construct Blocked map

      /// \param world A reference to the world
      /// \param size The number of elements to be mapped
      /// \param num_blocks The number of blocks [default = world.size()]
      /// \param seed The hashing seed for sudo random selection of nodes
      /// \param min_block_size The smallest block size allowed [default = 1]
      /// \note \c seed must be the same on all nodes.
      BlockedPmap(madness::World& world, std::size_t size, madness::hashT seed = 0ul, std::size_t num_blocks = 0ul, std::size_t min_block_size = 1ul) :
          block_size_(),
          num_blocks_(),
          size_(size),
          procs_(world.size()),
          seed_(seed)
      {
        TA_ASSERT(size_ > 0ul);

        // Check for the default value of num_blocks
        if((num_blocks == 0ul) || (num_blocks > std::size_t(world.size())))
          num_blocks = world.size();

        TA_ASSERT(min_block_size > 0ul);
        block_size_ = std::max<std::size_t>((size_ / num_blocks) + (size_ % num_blocks ? 1 : 0), min_block_size);
        TA_ASSERT(block_size_ > 0ul);

        num_blocks_ = (size_ / block_size_) + (size_ % block_size_ ? 1 : 0);

        // Construct a map of all local processes
        const ProcessID rank = world.rank();
        for(std::size_t block = 0; block < num_blocks_; ++block) {
          if(map_block_to_process(block) == rank) {
            // The block maps to this process so add it to the local list
            local_.reserve(local_.size() + block_size_);
            const std::size_t block_start = block * block_size_;
            const std::size_t block_finish = (block + 1) * block_size_;
            for(std::size_t i = block_start; (i < block_finish) && (i < size_); ++i)
              local_.push_back(i);
          }
        }
      }

      virtual ~BlockedPmap() { }

      /// Maps key to processor

      /// \param key Key for container
      /// \return Processor that logically owns the key
      virtual ProcessID owner(const key_type& key) const {
        TA_ASSERT(key < size_);
        return map_block_to_process(key / block_size_);
      }

      /// Local size accessor

      /// \return The number of local elements
      std::size_t local_size() const { return local_.size(); }

      /// Local elements

      /// \return \c true when there are no local elements, otherwise \c false .
      bool empty() const { return local_.empty(); }

      /// Begin local element iterator

      /// \return An iterator that points to the beginning of the local element set
      const_iterator begin() const { return local_.begin(); }


      /// End local element iterator

      /// \return An iterator that points to the beginning of the local element set
      const_iterator end() const { return local_.end(); }


    private:
      ProcessID map_block_to_process(std::size_t block) const {
        TA_ASSERT(block < num_blocks_);
        madness::hashT seed = seed_;
        madness::hash_combine(seed, block);
        return seed_ % procs_;
      }

      std::size_t block_size_; ///< block size
      std::size_t num_blocks_; ///< number of blocks in map
      const std::size_t size_; ///< The number of elements to be mapped
      const std::size_t procs_; ///< The number of processes in the world
      const madness::hashT seed_; ///< seed for hashing block locations
      std::vector<std::size_t> local_; ///< A list of local elements
    }; // class MapByRow

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_BLOCKED_PMAP_H__INCLUDED
