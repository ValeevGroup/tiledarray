#ifndef TILEDARRAY_PRED_SHAPE_H__INCLUDED
#define TILEDARRAY_PRED_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/shape.h>
#include <boost/make_shared.hpp>

namespace TiledArray {

  /// PredShape is a used defined shape based on a predicate.

  /// The predicate defines which tiles are present in a shape. The predicate
  /// must be a functor that has the following signature.
  /// \code
  /// class Predicate {
  ///   bool operator()(I i) const;
  ///   template<typename InIter>
  ///   bool operator()(InIter first, InIter last) const;
  /// }; // struct Predicate
  /// \endcode
  /// The first operator() function returns true if the ordinal index is included
  /// by the shape. The second operator() accepts two input iterators that
  /// define a coordinate, and returns true of the coordinate is is included by
  /// the shape.
  /// \tparam I is the ordinal index type.
  template<typename CS, typename Key, typename Pred>
  class PredShape : public Shape<CS, Key> {
  private:
    typedef PredShape<CS, Key, Pred> PredShape_;  ///< This class typedef
    typedef Shape<CS, Key> Shape_;                ///< Base class typedef

  public:
    typedef typename Shape_::index index;               ///< index type
    typedef typename Shape_::ordinal_index ordinal_index; ///< ordinal index type
    typedef typename Shape_::range_type range_type;     ///< Range type of shape
    typedef Pred pred_type;

    /// Construct the shape with a range object

    /// \param r A range object
    /// \param p The shape predicate
    PredShape(const range_type& r, const pred_type& p) :
        Shape_(r), pred_(p)
    { }

    /// Copy constructor

    /// \param other The shape to be copied
    PredShape(const PredShape_& other) : Shape_(other), pred_(other.pred_) { }

    /// Shape virtual destructor
    virtual ~PredShape() { }

    /// Construct a copy of this shape

    /// \return A shared pointer to a copy of this object
    virtual boost::shared_ptr<Shape_> clone() const {
      return boost::dynamic_pointer_cast<Shape_>(
          boost::make_shared<PredShape_>(*this));
    }

    /// Type info accessor for derived class
    virtual std::type_info type() const { return typeid(PredShape_); }

  private:

    /// Check that a tiles information is stored locally.

    /// \param i The ordinal index to check.
    virtual bool local(ordinal_index i) const { return true; }

    /// Probe for the presence of a tile in the shape

    /// \param i The index to be probed.
    virtual madness::Future<bool> probe(ordinal_index i) const {
      return madness::Future<bool>(pred_(i));
    }

    pred_type pred_; ///< The shape predicate
  }; // class PredShape

}  // namespace TiledArray

#endif // TILEDARRAY_PRED_SHAPE_H__INCLUDED
