#ifndef REPLICATED_ARRAY_H__INCLUDED
#define REPLICATED_ARRAY_H__INCLUDED
#if 0
#include <array.h>

namespace TiledArray {

  /// Tiled array with all data replicated across all nodes.
  template <typename T, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class ReplicatedArray : public Array<T, DIM, CS> {
  protected:
	typedef Array<T, DIM, CS> Array_;
    typedef LocalArray<T, DIM, CS> ReplicatedArray_;
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

  }; // class ReplicatedArray

} // namespace TiledArray
#endif
#endif // REPLICATED_ARRAY_H__INCLUDED
