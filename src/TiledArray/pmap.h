#ifndef TILEDARRAY_PMAP_H__INCLUDED
#define TILEDARRAY_PMAP_H__INCLUDED

#include <world/worldtypes.h>
#include <world/worldhash.h>
#include <vector>
#include <world/shared_ptr.h>

namespace TiledArray {

  /// Process map base class
  template <typename Key>
  class Pmap {
  public:
    typedef Key key_type;
    typedef typename std::vector<key_type>::const_iterator const_iterator;

    virtual ~Pmap() { }

    /// Initialize the hashing seed and local iterator
    virtual void set_seed(madness::hashT) = 0;

    /// Create a copy of this pmap

    /// \return A shared pointer to the new object
    virtual std::shared_ptr<Pmap<key_type> > clone() const = 0;

    /// Key owner

    /// \return The \c ProcessID of the process that owns \c key .
    virtual ProcessID owner(const key_type& key) const = 0;


    /// Local size accessor

    /// \return The number of local elements
    virtual std::size_t local_size() const = 0;

    /// Local elements

    /// \return \c true when there are no local elements, otherwise \c false .
    virtual bool empty() const = 0;

    /// Begin local element iterator

    /// \return An iterator that points to the beginning of the local element set
    virtual const_iterator begin() const = 0;


    /// End local element iterator

    /// \return An iterator that points to the beginning of the local element set
    virtual const_iterator end() const = 0;

  }; // class Pmap

}  // namespace TiledArray


#endif // TILEDARRAY_PMAP_H__INCLUDED
