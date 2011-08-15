#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/range.h>
#include <TiledArray/versioned_pmap.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tile.h>
#include <world/sharedptr.h>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <typeinfo>

namespace TiledArray {

  // Forward declarations
  template <typename>
  class DenseShape;
  template <typename>
  class SparseShape;
  class Shape;

  /// Shape of array tiles

  /// Stores information about the presence or absence of tiles in an array,
  /// and handles interprocess communication when the information is not locally
  /// available.
  /// \tparam CS The \c Shape coordinate system type
  /// \note This is an interface class only and cannot be constructed directly.
  /// Insted use DenseShape or SparseShape.
  class Shape {
  private:

    // not allowed
    Shape(const Shape&);
    Shape& operator=(const Shape&);

  public:
    typedef std::size_t size_type;


    /// Virtual destructor
    virtual ~Shape() { }

    /// Type info accessor for derived class
    virtual const std::type_info& type() const = 0;

    /// Is shape data for key \c k stored locally.

    /// \param i The key to check
    /// \return \c true when shape data for the given tile is stored on this node,
    /// otherwise \c false.
    virtual bool is_local(const size_type& i) const { return true; }

    /// Probe for the presence of an element at key
    virtual bool probe(const size_type& i) const { return true; }
  };

  /// Runtime type checking for dense shape

  /// \tparam T Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c DenseShape class return \c true , otherwise \c false .
  template <typename T>
  inline bool is_dense_shape(const T&) {
    return false;
  }

  /// Runtime type checking for dense shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c DenseShape class return \c true , otherwise \c false .
  template <typename C>
  inline bool is_dense_shape(const Shape<C>& s) {
    return s.type() == typeid(DenseShape<C>);
  }

  /// Runtime type checking for dense shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c DenseShape class return \c true , otherwise \c false .
  template <typename C>
  inline bool is_dense_shape(const DenseShape<C>&) {
    return true;
  }

  /// Runtime type checking for sparse shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c SparseShape class return \c true , otherwise \c false .
  template <typename T>
  inline bool is_sparse_shape(const T&) {
    return false;
  }

  /// Runtime type checking for sparse shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c SparseShape class return \c true , otherwise \c false .
  template <typename C>
  inline bool is_sparse_shape(const Shape<C>& s) {
    return s.type() == typeid(SparseShape<C>);
  }

  /// Runtime type checking for sparse shape

  /// \tparam CS Coordinate system type of shape
  /// \param s The shape to check
  /// \return If shape is a \c SparseShape class return \c true , otherwise \c false .
  template <typename C>
  inline bool is_sparse_shape(const SparseShape<C>&) {
    return true;
  }

  /// Create a copy of the shape

  /// The new shape has the same tiles as the original but may refer to different
  /// process maps and range object.
  /// \tparam CS The shape coordiante system type
  /// \param world The world where the shape exists
  /// \param range The range of the result shape
  /// \param pmap The range process map
  /// \param other The shape to be copied
  template <typename CS>
  Shape<CS>* shape_copy(madness::World& world, const typename Shape<CS>::range_type& range,
      const std::shared_ptr<typename Shape<CS>::pmap_interface>& pmap, const Shape<CS>& other)
  {
    TA_ASSERT(other.range() == range);

    Shape<CS>* result = NULL;

    if(is_dense_shape(other))
      result = new DenseShape<CS>(range, pmap);
    else
      result = new SparseShape<CS>(world, range, pmap,
          static_cast<const SparseShape<CS>&>(other));

    return result;
  }

  /// Create a copy of the shape

  /// The new shape has the same tiles as the original but may refer to different
  /// process maps and range object.
  /// \tparam CS The shape coordiante system type
  /// \param world The world where the shape exists
  /// \param range The range of the result shape
  /// \param pmap The range process map
  /// \param other The shape to be copied
  template <typename CS>
  Shape<CS>* shape_permute(madness::World& world, const typename Shape<CS>::range_type& range,
      const std::shared_ptr<typename Shape<CS>::pmap_interface>& pmap, const Permutation<CS::dim>& p, const Shape<CS>& other)
  {
    TA_ASSERT(other.range() == range);

    Shape<CS>* result = NULL;

    if(is_dense_shape(other))
      result = new DenseShape<CS>(range, pmap);
    else
      result = new SparseShape<CS>(world, range, pmap, p ^ other.make_shape_map());

    return result;
  }

  template <typename CS>
  Shape<CS>* shape_union(madness::World& world, const typename Shape<CS>::range_type& range,
      const std::shared_ptr<typename Shape<CS>::pmap_interface>& pmap, const Shape<CS>& left,
      const Shape<CS>& right)
  {
    TA_ASSERT(range == left.range());
    TA_ASSERT(range == right.range());

    Shape<CS>* result = NULL;

    if(is_dense_shape(left) || is_dense_shape(right))
      result = new DenseShape<CS>(range, pmap);
    else
      result = new SparseShape<CS>(world, range, pmap,
          static_cast<const SparseShape<CS>&>(left),
          static_cast<const SparseShape<CS>&>(right));

    return result;
  }

  template <typename ResCS, typename I, typename LeftCS, typename RightCS>
  Shape<ResCS>* shape_contract(madness::World& world, const typename Shape<ResCS>::range_type& range,
      const std::shared_ptr<typename Shape<ResCS>::pmap_interface>& pmap, const std::shared_ptr<math::Contraction<I> >& cont,
      const Shape<LeftCS>& left, const Shape<RightCS>& right)
  {
    Shape<ResCS>* result = NULL;

    if(is_dense_shape(left) && is_dense_shape(right))
      result = new DenseShape<ResCS>(range, pmap);
    else {
      expressions::ContractionTensor<typename Shape<LeftCS>::array_type,
        typename Shape<RightCS>::array_type> contracted(left.make_shape_map(),
        right.make_shape_map(), cont);

      result = new SparseShape<ResCS>(world, range, pmap, contracted);
    }

    return result;
  }

} // namespace TiledArray

#endif // TILEDARRAY_SHAPE_H__INCLUDED
