#ifndef TILEDARRAY_DENSE_SHAPE_H__INCLUDED
#define TILEDARRAY_DENSE_SHAPE_H__INCLUDED

#include <TiledArray/shape.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/shape.h>
#include <boost/make_shared.hpp>

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
  /// \var \c R is the range object type.
  template<typename CS, typename Key>
  class DenseShape : public Shape<CS, Key> {
  protected:
    typedef DenseShape<CS, Key> DenseShape_;
    typedef Shape<CS, Key> Shape_;

  public:
    typedef typename Shape_::index index;
    typedef typename Shape_::ordinal_index ordinal_index;

    /// Primary constructor

    /// Since all tiles are present in a dense array, the shape is considered
    /// Immediately available.
    DenseShape(const typename Shape_::range_type& r) :
        Shape_(r)
    { }

    DenseShape(const DenseShape_& other) : Shape_(other) { }

    virtual boost::shared_ptr<Shape_> clone() const {
      return boost::dynamic_pointer_cast<Shape_>(
          boost::make_shared<DenseShape_>(*this));
    }

    virtual const std::type_info& type() const { return typeid(DenseShape_); }

  private:

    /// Check that a tiles information is stored locally.

    /// \param i The ordinal index to check.
    virtual bool local(ordinal_index) const { return true; }

    /// Probe for the presence of a tile in the shape

    /// \param i The index to be probed.
    virtual madness::Future<bool> probe(ordinal_index) const {
      return madness::Future<bool>(true);
    }

  }; // class DenseShape

  template <typename CS, typename Key>
  inline bool is_dense(const boost::shared_ptr<Shape<CS, Key> >& s) {
    return s->type() == typeid(DenseShape<CS,Key>);
  }

  template <typename CS, typename Key>
  inline bool is_dense(const boost::shared_ptr<DenseShape<CS, Key> >&) {
    return true;
  }

}  // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
