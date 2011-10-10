#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/error.h>
#include <world/worlddep.h>
#include <world/shared_ptr.h>
#include <world/worldref.h>

namespace TiledArray {
  namespace detail {

    /// \tparam T Tensor type
    template <typename T>
    class Tile : public madness::DependencyInterface {
      typedef T value_type;

      using madness::DependencyInterface::inc;
      using madness::DependencyInterface::dec;
      using madness::DependencyInterface::probe;

      Tile() :
        madness::DependencyInterface(1),
        tensor_(this)
      { }

      Tile(const value_type& t) :
        madness::DependencyInterface(0),
        tensor_(t)
      { }

      const std::shared_ptr<const value_type>& tensor() const {
        return std::shared_ptr<const value_type>(&tensor_, madness::detail::no_delete<const value_type>);
      }

      template <typename U>
      void set(const U& t) {
        TA_ASSERT(probe());
        t.eval_to(*tensor_);
        dec();
      }

    private:

      value_type tensor_;

    }; // class Tile

  } // namespace detail
} // namespace TiledArray

namespace madness {
  namespace archive {

    template <class Archive, class T>
    struct ArchiveStoreImpl;
    template <class Archive, class T>
    struct ArchiveLoadImpl;

  }
}
#endif // TILEDARRAY_TILE_H__INCLUDED
