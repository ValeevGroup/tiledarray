#ifndef TILEDARRAY_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_DENSE_SHAPE_H__INCLUDED

#include <TiledArray/shape.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/shape.h>
#include <world/sharedptr.h>

namespace TiledArray {

  /// Dense shape used to construct Array objects.

  /// DenseShape is used to represent dense arrays. It is initialized with a
  /// madness world object and a range object. It includes all tiles included by
  /// the range object.
  ///
  /// Template parameters:
  /// \tparam CS Coordiante system.
  class DenseShape : public Shape {
  protected:

  public:
    typedef typename Shape::size_type size_type;

    virtual ~DenseShape() { }

    /// Type info accessor for derived class
    virtual const std::type_info& type() const { return typeid(DenseShape); }

  }; // class DenseShape

}  // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
