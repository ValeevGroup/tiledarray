#ifndef TILEDARRAY_PRED_SHAPE_H__INCLUDED
#define TILEDARRAY_PRED_SHAPE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/madness_runtime.h>
#include <TiledArray/shape.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility.hpp>
#include <typeinfo>

namespace TiledArray {

  namespace math {
    template<typename I, template <typename> class Op >
    class BinaryShapeOp;
  }
/*
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
  template<typename I>
  class PredShape : public Shape<I> {
  private:
    typedef PredShape<I> PredShape_;                    ///< This class typedef
    typedef Shape<I> Shape_;                            ///< Base class typedef

  public:
    typedef typename Shape_::ordinal_type ordinal_type; ///< ordinal index type
    typedef typename Shape_::size_array size_array;     ///< size array type

    /// Construct the shape with a range object

    /// \tparam Range The range object type
    /// \tparam Predicate The is the predicate object type
    /// \param r A range object
    /// \param p The shape predicate
    template<typename Range, typename Predicate>
    PredShape(const Range& r, Predicate p) :
        Shape_(detail::predicated_shape, r), predicate_(new_pred(p))
    { }

    /// Constructor a shape with a range of the given size

    /// \tparam Size The size array type
    /// \tparam Predicate The is the predicate object type
    /// \param s A size array that defines the size of the shape range
    /// \param o The dimension ordering
    /// \param p The shape predicate
    template<typename Size, typename Predicate>
    PredShape(const Size& s, detail::DimensionOrderType o, Predicate p) :
        Shape_(detail::predicated_shape, s, o), predicate_(new_pred(p))
    { }

    /// Constructor defined with an upper and lower bound.

    /// All elements of finish must be greater than or equal to those of start.
    /// \tparam InIter Input iterator type
    /// \tparam Predicate The is the predicate object type
    /// \param start A tile index array that defines the lower boundary of the shape range
    /// \param finish A tile index array that defines the lower boundary of the shape range
    /// \param o The dimension ordering
    /// \param p The shape predicate
    template<typename Index, typename Predicate>
    PredShape(const Index& start, const Index& finish, detail::DimensionOrderType o, Predicate p) :
        Shape_(detail::predicated_shape, start, finish, o), predicate_(new_pred(p))
    { }

    /// Shape virtual destructor
    virtual ~PredShape() { }

    /// Predicate interface class.

    /// This class abstracts the predicate from the shape.
    class PredInterface {
    public:
      virtual boost::shared_ptr<PredInterface> clone() const = 0;
      virtual bool check(const ordinal_type& i) const = 0;
      virtual bool check(const size_array& i) const = 0;
      virtual const std::type_info& type() const = 0;
    }; // class PredInterface

    /// Returns a shared pointer to a copy of the predicate.
    boost::shared_ptr<PredInterface> clone_pred() const {
      return predicate_->clone();
    }

  private:

    /// Returns a shared pointer to the predicate.

    /// This function handles the case where the predicate is already a shared
    /// pointer to a predicate inter
    template<typename Predicate>
    static typename boost::enable_if<
      boost::is_same<Predicate, boost::shared_ptr<PredInterface> >,
    boost::shared_ptr<PredInterface> >::type new_pred(const Predicate& p) {
      return p;
    }

    template<typename Predicate>
    static typename boost::disable_if<
      boost::is_same<Predicate, boost::shared_ptr<PredInterface> >,
    boost::shared_ptr<PredInterface> >::type new_pred(const Predicate& p) {
      return boost::dynamic_pointer_cast<PredInterface>(boost::make_shared<PredHolder<Predicate> >(p));
    }

    /// Indicates the presence or absence of the tile in the shape.

    /// The value of the future will be true if the tile is presence or false if
    /// the tile is absent.
    /// \param i The tile's ordinal index.
    /// \return a \c bool future indicating presence or absence of the tile
    virtual madness::Future<bool> tile_includes(ordinal_type i) const {
      return madness::Future<bool>(predicate_->check(i));
    }

    /// Indicates the presence or absence of the tile in the shape.

    /// The value of the future will be true if the tile is presence or false if
    /// the tile is absent.
    /// \param i A size array with the tiles coordinate index.
    /// \return a \c bool future indicating presence or absence of the tile
    virtual madness::Future<bool> tile_includes(size_array i) const {
      return madness::Future<bool>(predicate_->check(i));
    }

    /// Returns the initialization status of the array.

    /// This function indicates that the array has been fully initialized.
    /// \return bool true = shape is initialized, false = array is not initialized
    virtual bool initialized() const { return true; }

    /// Holds a predicate object of type P

    /// \tparam P is the predicate type.
    template<typename P>
    class PredHolder : public PredInterface {
    private:
      typedef P pred_type; ///< predicate type

    public:
      /// Construct a predicate holder for p

      /// This constructor stores a copy predicate p for use by the shape.
      /// \param p predicate to be held.
      PredHolder(pred_type p) : pred_(p) { }

      virtual ~PredHolder() { }

      virtual boost::shared_ptr<PredInterface> clone() const {
        boost::shared_ptr<PredInterface> result =
            boost::dynamic_pointer_cast<PredInterface>(
            boost::make_shared<PredHolder<P> >(pred_));
        TA_ASSERT(result.get() != NULL, std::runtime_error,
            "Predicate holder dynamic_cast failed.");
        return result;
      }
      virtual bool check(const ordinal_type& i) const { return pred_(i); }
      virtual bool check(const size_array& i) const { return pred_(i.begin(), i.end()); }
      virtual const std::type_info& type() const { return typeid(pred_); }

    private:
      pred_type pred_;
    }; // class PredHolder

    template<typename, template <typename> class >
    friend class math::BinaryShapeOp;

    boost::shared_ptr<PredInterface> predicate_;
  }; // class PredShape
*/
}  // namespace TiledArray

#endif // TILEDARRAY_PRED_SHAPE_H__INCLUDED
