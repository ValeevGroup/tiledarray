#ifndef DISTRIBUTED_ARRAY_H__INCLUDED
#define DISTRIBUTED_ARRAY_H__INCLUDED

#include <array.h>

namespace TiledArray {

  /// Tiled Array with data distributed across many nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class DistributedArray : public Array<T, DIM, CS> {
  protected:
	typedef Array<T, DIM, CS> Array_;
    typedef LocalArray<T, DIM, CS> DistributedArray_;
    typedef typename Array_::range range;
    typedef typename Array_::range_iterator range_iterator;
    typedef typename Array_::shape shape;
    typedef typename Array_::shape_iterator shape_iterator;

  public:
    typedef T value_type;
    typedef CS coordinate_system;
	typedef typename Array_::tile_index tile_index;
	typedef typename Array_::element_index element_index;
	typedef typename Array_::tile tile;

  protected:
    typedef typename Array_::array_map array_map;

  }; // class DistributedArray

} // namespace TiledArray

#endif // DISTRIBUTED_ARRAY_H__INCLUDED
