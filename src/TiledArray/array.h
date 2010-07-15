#ifndef TILEDARRAY_ARRAY_H__INCLUDED
#define TILEDARRAY_ARRAY_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/key.h>
#include <TiledArray/shape.h>
#include <boost/shared_ptr.hpp>

namespace TiledArray {

  template <typename>
  class Shape;

  template <typename T, typename CS, typename C>
  class Array : madness::WorldObject<Array<T, CS, C> >{
  private:
    typedef detail::Key<typename CS::ordinal_index, typename CS::index> key_type;
    typedef typename C::container_type container_type;

  public:
    typedef Array<T, CS, C> Array_;
    typedef CS coordinate_system;
    typedef typename CS::index index;
    typedef typename CS::ordinal_index ordinal_index;

    typedef typename C::value_type value_type;
    typedef typename C::reference reference;
    typedef typename C::const_reference const_reference;

  private:
    TiledRange<CS> tiled_range_;
    madness::SharedPtr<madness::WorldDCDefaultPmap<key_type> > pmap_;
    boost::shared_ptr<Shape<ordinal_index> > shape_;
    madness::WorldContainer<key_type, container_type> tiles_;
  };

} // namespace TiledArray

#endif // TILEDARRAY_ARRAY_H__INCLUDED
