#ifndef TILEDARRAY_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_DENSE_SHAPE_H__INCLUDED

#include <TiledArray/shape.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/shape.h>

namespace madness {
  template<typename T>
  class Future;

} // namespace madness
namespace TiledArray {
/*
  /// Dense shape used to construct Array objects.

  /// DenseShape is used to represent dense arrays. It is initialized with a
  /// madness world object and a range object. It includes all tiles included by
  /// the range object.
  ///
  /// Template parameters:
  /// \var \c R is the range object type.
  template<typename I>
  class DenseShape : public Shape<I> {
  protected:
    typedef DenseShape<I> DenseShape_;
    typedef Shape<I> Shape_;

  public:
    typedef typename Shape_::ordinal_type ordinal_type;
    typedef typename Shape_::size_array size_array;

    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    template<typename R>
    DenseShape(const R& r) :
        Shape_(detail::dense_shape, r)
    { }

    /// Constructor defined by an size array.
    template<typename Size>
    DenseShape(const Size& size, detail::DimensionOrderType o) :
        Shape_(detail::dense_shape, size, o)
    { }

    /// Constructor defined by an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    template<typename Index>
    DenseShape(const Index& start, const Index& finish, detail::DimensionOrderType o) :
        Shape_(detail::dense_shape, start, finish, o)
    { }

  private:
    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(ordinal_type) const {
      return madness::Future<bool>(true);
    }

    /// Returns madness::Future<bool> which will be true if the tile is included.
    virtual madness::Future<bool> tile_includes(size_array) const {
      return madness::Future<bool>(true);
    }

    /// Returns true if the local data has been fully initialized.
    virtual bool initialized() const { return true; }

  }; // class DenseShape
*/
}  // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
