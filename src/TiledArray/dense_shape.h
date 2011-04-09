#ifndef TILEDARRAY_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_DENSE_SHAPE_H__INCLUDED

#include <TiledArray/shape.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/shape.h>
#include <world/sharedptr.h>

namespace madness {
  template<typename T>
  class Future;

} // namespace madness
namespace TiledArray {

  /// Dense shape used to construct Array objects.

  /// DenseShape is used to represent dense arrays. It is initialized with a
  /// madness world object and a range object. It includes all tiles included by
  /// the range object.
  ///
  /// Template parameters:
  /// \tparam CS Coordiante system.
  template<typename CS>
  class DenseShape : public Shape<CS> {
  protected:
    typedef DenseShape<CS> DenseShape_;
    typedef Shape<CS> Shape_;

  public:
    typedef CS coordinate_system;                         ///< Shape coordinate system
    typedef typename Shape_::key_type key_type;           ///< The pmap key type
    typedef typename Shape_::index index;
    typedef typename Shape_::ordinal_index ordinal_index;
    typedef typename Shape_::range_type range_type;
    typedef typename Shape_::pmap_type pmap_type;

  private:

    // Default constuctor not allowed
    DenseShape();

    /// Copy constructor

    /// \param other The dense shape to be copied
    DenseShape(const DenseShape_& other) : Shape_(other) { }

  public:

    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    /// \param r A shared pointer to the range object of the shape
    /// \param m A shared pointer to the process map object
    DenseShape(const range_type& r, const pmap_type& m) :
        Shape_(r, m)
    { }

    virtual ~DenseShape() { }

    /// Clone (copy) this object.

    /// Create a copy of this object using the copy constructor and place the
    /// result in a shared pointer to a shape object.
    /// \return A shared pointer
    virtual std::shared_ptr<Shape_> clone() const {
      return std::shared_ptr<Shape_>(static_cast<Shape_*>(new DenseShape_(*this)));
    }

  private:
    // Assignement operator not allowed
    DenseShape_& operator=(const DenseShape_&);

  public:
    virtual const std::type_info& type() const { return typeid(DenseShape_); }

  }; // class DenseShape

}  // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
