#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <iterator.h>
#include <cstddef>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/iterator/filter_iterator.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM>
  class Permutation;
  template <unsigned int DIM, typename CS>
  class Range;


  /// Abstract Iterator over a subset of RangeIterator's domain. Example of RangeIterator is Range::tile_iterator.
  template <unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Shape {
  public:
    typedef Shape<DIM,CS> Shape_;
    typedef Range<DIM,CS> range_type;
    typedef CS coordinate_system;
    typedef typename range_type::index_type index_type;
    typedef detail::IndexIterator<index_type, Shape> const_iterator;
    friend class detail::IndexIterator< index_type , Shape>;

    static unsigned int dim() { return DIM; }

    virtual boost::shared_ptr<const range_type> range() const =0;

    virtual const_iterator begin() const =0;
    virtual const_iterator end() const =0;

    // if this index included in the shape?
    virtual bool includes(const index_type&) const =0;

  protected:

    virtual void increment(index_type&) const =0;

  };


  /// Concrete ShapeIterator whose iteration domain is determined by Predicate
  template <unsigned int DIM, typename Predicate, typename CS = CoordinateSystem<DIM> >
  class PredShape : public Shape<DIM,CS> {
  public:
	typedef PredShape<DIM,Predicate,CS> PredShape_;
    typedef Shape<DIM,CS> Shape_;
    typedef Predicate pred_type;
    typedef CS coordinate_system;
    typedef typename Shape<DIM,CS>::range_type range_type;
    typedef typename Shape<DIM,CS>::index_type index_type;
    typedef typename Shape<DIM,CS>::const_iterator const_iterator;
    friend class detail::IndexIterator< index_type , Shape_ >;

    /// Iterator main constructor
    PredShape(const boost::shared_ptr<range_type>& range, pred_type pred = pred_type()) :
    	pred_(pred), range_(range) {}

    /// Copy constructor
    PredShape(const PredShape& other) :
        range_(other.range_), pred_(other.pred_) {}

    ~PredShape() {}

    boost::shared_ptr<const range_type> range() const {
      boost::shared_ptr<const range_type> result = boost::const_pointer_cast<const range_type>(range_);
      return result;
    }

    /// Predicate accessor function
    const pred_type& predicate() const {
      return pred_;
    }

    /// Begin accessor function
    const_iterator begin() const {
      index_type first(range_->tiles().start());
      if(! this->includes(first))
        this->increment(first);

      const_iterator result(first, this);
      return result;
    }

    /// End accessor function
    const_iterator end() const {
      index_type last(range_->tiles().finish());
      const_iterator result(last, this);
      return result;
    }

    bool includes(const index_type& index) const {
      return pred_(index) && range_->tiles().includes(index);
    }

    /// Permutes range
    PredShape_& operator ^=(const Permutation<DIM>& perm) {
      pred_ ^= perm;
      return *this;
    }

  protected:
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
    boost::shared_ptr<range_type> range_;

  };

} // namespace TiledArray


#endif // SHAPE_H__INCLUDED
