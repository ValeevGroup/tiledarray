#ifndef TILEDARRAY_VERSIONED_PMAP_H__INCLUDED
#define TILEDARRAY_VERSIONED_PMAP_H__INCLUDED

#include <TiledArray/madness_runtime.h>
#include <world/worlddc.h>

namespace TiledArray {
  namespace detail {

    /// Versioned process map

    /// This process map returns a new mapping for each new version.
    /// \tparam I The index type that will be hashed
    /// \tparam Hasher The hashing function type
    template <typename Key, typename Hasher = madness::Hash<Key> >
    class VersionedPmap : public madness::WorldDCPmapInterface<Key> {
    private:
        const int size_;       ///< The number of processes in the world
        unsigned int version_;  ///< The process map version
        Hasher hashfun_;        ///< The hashing function

    public:
        typedef Key key_type;

        /// Primary constructor

        /// \param s The world size
        /// \param v The initial version number for this pmap (Default = 0 )
        /// \param h The hashing function used to hash (Default = Hasher() )
        VersionedPmap(std::size_t s, unsigned int v = 0, const Hasher& h = Hasher()) :
            size_(s), version_(v), hashfun_(h)
        { }

        virtual ~VersionedPmap() { }

        // Compiler generated copy constructor and assignment operator are fine here

        /// Increment the version counter

        /// \return The new version number for the pmap
        unsigned int version() { return version_; }

        /// Owner of an index

        /// This function calculates the owning process of an index value by
        /// hashing the given index and the pmap version number.
        /// \param k The key to be mapped to a process.
        /// \return The process number associated with the given index. This
        /// process number is less-than-or-equal-to world size.
        virtual ProcessID owner(const key_type& k) const {
          std::size_t seed = hashfun_(k);
          madness::hash_combine(seed, version_);
          return (seed % size_);
        }
    }; // class VersionedPmap

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_VERSIONED_PMAP_H__INCLUDED
