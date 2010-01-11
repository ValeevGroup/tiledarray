#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <TiledArray/iterator.h>
//#include <cstddef>
//#include <algorithm>
//#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
//#include <boost/iterator/filter_iterator.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <typename I, unsigned int DIM, typename CS>
  class Shape;
  template <unsigned int DIM>
  class Permutation;
  template <typename I, unsigned int DIM, typename CS>
  class TiledRange;
  template <typename I, unsigned int DIM, typename CS>
  Shape<I,DIM,CS>* operator ^=(Shape<I,DIM,CS>*, const Permutation<DIM>&);


  /// Abstract Iterator over a subset of RangeIterator's domain. Example of RangeIterator is TiledRange::tile_iterator.
  template <typename I, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Shape {
  public:
    typedef Shape<I,DIM,CS> Shape_;
    typedef TiledRange<I,DIM,CS> tiled_range_type;
    typedef CS coordinate_system;
    typedef typename tiled_range_type::index_type index_type;
    typedef detail::IndexIterator<index_type, Shape> const_iterator;
    friend class detail::IndexIterator< index_type , Shape>;

    static unsigned int dim() { return DIM; }

    virtual boost::shared_ptr<const tiled_range_type> range() const =0;

    virtual const_iterator begin() const =0;
    virtual const_iterator end() const =0;

    // if this index included in the shape?
    virtual bool includes(const index_type&) const =0;

  protected:
    virtual void permute(const Permutation<DIM>& p) =0;
    virtual void increment(index_type&) const =0;

    friend Shape* operator ^= <>(Shape*, const Permutation<DIM>&);

  };

  template <typename I, unsigned int DIM, typename CS>
  Shape<I,DIM,CS>* operator ^=(Shape<I,DIM,CS>* s, const Permutation<DIM>& p) {
    s->permute(p);
    return s;
  }


  /// Concrete ShapeIterator whose iteration domain is determined by Predicate
  template <typename I, unsigned int DIM, typename Predicate, typename CS = CoordinateSystem<DIM> >
  class PredShape : public Shape<I,DIM,CS> {
  public:
    typedef PredShape<I,DIM,Predicate,CS> PredShape_;
    typedef Shape<I,DIM,CS> Shape_;
    typedef Predicate pred_type;
    typedef CS coordinate_system;
    typedef typename Shape<I,DIM,CS>::tiled_range_type tiled_range_type;
    typedef typename Shape<I,DIM,CS>::index_type index_type;
    typedef typename Shape<I,DIM,CS>::const_iterator const_iterator;
    friend class detail::IndexIterator< index_type , Shape_ >;

    /// Iterator main constructor
    PredShape(const boost::shared_ptr<tiled_range_type>& range, pred_type pred = pred_type()) :
        pred_(pred), range_(range) {}

    /// Shape<...> pointer constructor. This constructor automatically does the
    /// dynamic cast of the pointer.
    PredShape(const Shape_* other) : pred_(), range_() {
      const PredShape* temp = dynamic_cast<const PredShape*>(other);
      range_ = boost::make_shared<tiled_range_type>(* temp->range_);
      pred_ = temp->pred_;
    }

    /// Copy constructor
    PredShape(const PredShape& other) : pred_(other.pred_), range_() {
      range_ = boost::make_shared<tiled_range_type>(* other.range_);
    }

    virtual ~PredShape() {}

    boost::shared_ptr<const tiled_range_type> range() const {
      boost::shared_ptr<const tiled_range_type> result = boost::const_pointer_cast<const tiled_range_type>(range_);
      return result;
    }

    /// Predicate accessor function
    const pred_type& predicate() const {
      return pred_;
    }

    /// Returns an input index iterator.

    /// Returns an index iterator that points to the first element included in
    /// the shape. If no tiles are included in the range by the predicate or the
    /// tiled range, the result will point to the end of the range.
    const_iterator begin() const {
      index_type first(range_->tiles().start());
      if(! this->includes(first))
        this->increment(first);

      const_iterator result(first, this);
      return result;
    }

    /// Returns an input iterator that points to the end of the range.
    const_iterator end() const {
      index_type last(range_->tiles().finish());
      const_iterator result(last, this);
      return result;
    }

    /// Returns true if the given index is included in the tiled range and included
    /// by the predicate.
    bool includes(const index_type& index) const {
      return pred_(index) && range_->tiles().includes(index);
    }

  protected:

    /// Permutes the shape (which includes the range).
    void permute(const Permutation<DIM>& p) {
      pred_ ^= p;
      *range_ ^= p;
    }

    virtual void increment(index_type& i) const {
      do {
        detail::IncrementCoordinate<index_type,coordinate_system>(i, range_->tiles().start(), range_->tiles().finish());
      } while( !includes(i) && i != range_->tiles().finish() );
    }

  private:
    // Default construction and assignment is not allowed because a pointer to range is required.
    PredShape();
    PredShape& operator =(const PredShape&);

    pred_type pred_;
    boost::shared_ptr<tiled_range_type> range_;

  };

} // namespace TiledArray


#endif // SHAPE_H__INCLUDED
