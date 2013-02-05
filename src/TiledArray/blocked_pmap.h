#ifndef TILEDARRAY_BLOCKED_PMAP_H__INCLUDED
#define TILEDARRAY_BLOCKED_PMAP_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/pmap.h>
#include <TiledArray/madness.h>
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
      BlockedPmap(madness::World& world, std::size_t size, std::size_t num_blocks = 0ul, std::size_t min_block_size = 1ul) :
          block_size_(),
          num_blocks_(),
          size_(size),
          rank_(world.rank()),
          procs_(world.size()),
          seed_(0ul),
          local_()
      {
        TA_ASSERT(size_ > 0ul);

        // Check for the default value of num_blocks
        if((num_blocks == 0ul) || (num_blocks > procs_))
          num_blocks = world.size();

        TA_ASSERT(min_block_size > 0ul);
        block_size_ = std::max<std::size_t>((size_ / num_blocks) + (size_ % num_blocks ? 1 : 0), min_block_size);
        TA_ASSERT(block_size_ > 0ul);

        num_blocks_ = (size_ / block_size_) + (size_ % block_size_ ? 1 : 0);
      }

    private:

      BlockedPmap(const BlockedPmap& other) :
          block_size_(other.block_size_),
          num_blocks_(other.num_blocks_),
          size_(other.size_),
          rank_(other.rank_),
          procs_(other.procs_),
          seed_(0ul),
          local_()
      { }

    public:

      ~BlockedPmap() { }

      virtual void set_seed(madness::hashT seed = 0ul) {
        seed_ = seed;

        // Construct a map of all local processes
        for(std::size_t block = 0; block < num_blocks_; ++block) {
          if(map_block_to_process(block) == rank_) {
            // The block maps to this process so add it to the local list
            local_.reserve(local_.size() + block_size_);
            const std::size_t block_start = block * block_size_;
            const std::size_t block_finish = (block + 1) * block_size_;
            for(std::size_t i = block_start; (i < block_finish) && (i < size_); ++i)
              local_.push_back(i);
          }
        }
      }

      /// Create a copy of this pmap

      /// \return A shared pointer to the new object
      virtual std::shared_ptr<Pmap<key_type> > clone() const {
        return std::shared_ptr<Pmap<key_type> >(new BlockedPmap(*this));
      }

      /// Maps key to processor

      /// \param key Key for container
      /// \return Processor that logically owns the key
      ProcessID owner(const key_type& key) const {
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
        if(num_blocks_ == procs_)
          return block;

        madness::hashT seed = seed_;
        madness::hash_combine(seed, block);
        return seed_ % procs_;
      }

      std::size_t block_size_; ///< block size
      std::size_t num_blocks_; ///< number of blocks in map
      const std::size_t size_; ///< The number of elements to be mapped
      const std::size_t rank_;
      const std::size_t procs_; ///< The number of processes in the world
      madness::hashT seed_; ///< seed for hashing block locations
      std::vector<std::size_t> local_; ///< A list of local elements
    }; // class MapByRow

  }  // namespace detail
}  // namespace TiledArray


#endif // TILEDARRAY_BLOCKED_PMAP_H__INCLUDED
