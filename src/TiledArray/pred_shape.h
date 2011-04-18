#ifndef TILEDARRAY_PRED_SHAPE_H__INCLUDED
#define TILEDARRAY_PRED_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/shape.h>
#include <world/sharedptr.h>

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
  template <typename CS, typename Pred>
  class PredShape : public Shape<CS> {
  private:
    typedef PredShape<CS, Pred> PredShape_;  ///< This class typedef
    typedef Shape<CS> Shape_;                ///< Base class typedef

  public:
    typedef CS coordinate_system;                         ///< Shape coordinate system
    typedef typename Shape_::key_type key_type;           ///< The pmap key type
    typedef typename Shape_::index index;                 ///< index type
    typedef typename Shape_::ordinal_index ordinal_index; ///< ordinal index type
    typedef typename Shape_::range_type range_type;       ///< Range type
    typedef typename Shape_::pmap_type pmap_type;         ///< Process map type
    typedef typename Shape_::array_type array_type;       ///< Dense array type
    typedef Pred pred_type;                               ///< Predicate type

  private:
    // Default constructor not allowed
    PredShape();

    /// Copy constructor

    /// \param other The shape to be copied
    PredShape(const PredShape_& other) : Shape_(other), pred_(other.pred_) { }

  public:

    /// Construct the shape with a range object

    /// \param r A shared pointer to a range object
    /// \param m A shared pointer to a process map
    /// \param p The shape predicate
    PredShape(const range_type& r, const pmap_type& m, const pred_type& p) :
        Shape_(r, m), pred_(p)
    { }


    /// Shape virtual destructor
    virtual ~PredShape() { }

  private:
    // Assignment not allowed
    PredShape_& operator=(const PredShape_&);

  public:

    /// Construct a copy of this shape

    /// \return A shared pointer to a copy of this object
    virtual std::shared_ptr<Shape_> clone() const {
      return std::shared_ptr<Shape_>(static_cast<Shape_*>(new PredShape_(*this)));
    }

    /// Type info accessor for derived class
    virtual const std::type_info& type() const { return typeid(PredShape_); }

    /// Construct a shape map

    /// \return A dense array that contains 1 where tiles exist in the shape and
    /// 0 where tiles do not exist in the shape.
    virtual array_type make_shape_map() const {
      array_type result(this->range(), 0);
      std::size_t vol = this->range().volume();
      for(std::size_t i = 0; i < vol; ++i)
        if(pred_(this->key(i)))
          result[i] = 1;
      return result;
    }

  private:

    /// Probe for tile existence

    /// \param k The index to be probed.
    virtual bool local_probe(const key_type& k) const {
      return pred_(k);
    }

    pred_type pred_; ///< The shape predicate
  }; // class PredShape

}  // namespace TiledArray

#endif // TILEDARRAY_PRED_SHAPE_H__INCLUDED
