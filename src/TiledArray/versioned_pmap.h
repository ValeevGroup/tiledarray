#ifndef TILEDARRAY_VERSIONED_PMAP_H__INCLUDED
#define TILEDARRAY_VERSIONED_PMAP_H__INCLUDED

#include <TiledArray/madness_runtime.h>
#include <world/worlddc.h>
#include <boost/functional/hash.hpp>

namespace TiledArray {
  namespace detail {

    /// Versioned process map

    /// This process map returns a new mapping for each new version.
    /// \tparam I The index type that will be hashed
    /// \tparam Hasher The hashing function type
    template <typename I, typename Hasher = boost::hash<I> >
    class VersionedPmap : public madness::WorldDCPmapInterface<I> {
    private:
        const int size_;       ///< The number of processes in the world
        unsigned int version_;  ///< The process map version
        Hasher hashfun_;        ///< The hashing function

    public:
        /// Primary constructor

        /// \param w The world that this Pmap will belong to
        /// \param v The initial version number for this pmap (Default = 0 )
        /// \param h The hashing function used to hash (Default = Hasher() )
        VersionedPmap(madness::World& w, unsigned int v = 0, const Hasher& h = Hasher()) :
            size_(w.size()), version_(v), hashfun_(h)
        { }

        virtual ~VersionedPmap() { }

        // Compiler generated copy constructor and assignment operator are fine here

        /// Increment the version counter

        /// \return The new version number for the pmap
        unsigned int version() { return version_; }

        /// Owner of an index

        /// This function calculates the owning process of an index value by
        /// hashing the given index and the pmap version number.
        /// \param i The index value to be mapped to a process.
        /// \return The process number associated with the given index. This
        /// process number is less-than-or-equal-to world size.
        ProcessID owner(const I& i) const {
          std::size_t seed = hashfun_(i);
          boost::hash_combine(seed, version_);
          return (seed % size_);
        }
    }; // class VersionedPmap

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_VERSIONED_PMAP_H__INCLUDED
