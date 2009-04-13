#ifndef DISTRIBUTED_ARRAY_H__INCLUDED
#define DISTRIBUTED_ARRAY_H__INCLUDED

#include <boost/shared_ptr.hpp>

#include <array.h>
#include <madness_runtime.h>

namespace TiledArray {

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class DistributedArray : public Array<T, DIM, CS> {

  public:
    typedef Array<T, DIM, CS> Array_;
    typedef T value_type;
    typedef CS coordinate_system;
	typedef typename Array_::tile_index tile_index;
	typedef typename Array_::element_index element_index;
	typedef typename Array_::tile tile;
	typedef boost::shared_ptr<tile> tile_ptr;
    typedef typename Array_::range range;
    typedef typename Array_::range_iterator range_iterator;
    typedef typename Array_::shape shape;
    typedef typename Array_::shape_iterator shape_iterator;
    typedef typename range::ordinal_index ordinal_index;

  private:
    typedef DistributedContainer<ordinal_index,tile_ptr> tile_container;

    tile_container tiles_;

  }; // class DistributedArray

} // namespace TiledArray

#endif // DISTRIBUTED_ARRAY_H__INCLUDED
