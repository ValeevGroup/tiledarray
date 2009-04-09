#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <coordinates.h>
#include <range.h>
#include <iterator.h>
#include <predicate.h>

namespace TiledArray {

  /// Abstract Iterator over a subset of RangeIterator's domain. Example of RangeIterator is Range::tile_iterator.
  template <unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Shape {
  public:
    typedef Shape<DIM,CS> Shape_;
    typedef Range<DIM,CS> range_type;
    typedef CS coordinate_system;
    typedef detail::IndexIterator<typename range_type::tile_index, Shape> iterator;

    unsigned int dim() const { return DIM; }

    Shape(const boost::shared_ptr<range_type>& range) : range_(range) {}
    Shape(const Shape& other) : range_(other.range_) {}
    virtual ~Shape() {}

    const boost::shared_ptr<range_type>& range() const { return range_; }

    virtual iterator begin() const =0;
    virtual iterator end() const =0;

    // if this index included in the shape?
    virtual bool includes(const typename iterator::value_type& index) const =0;

    // Friend the iterator class so it has access to the increment function.
    friend class detail::IndexIterator<typename range_type::tile_index, Shape>;

  protected:

    virtual void increment(typename iterator::value_type& index) const =0;

  private:
    const boost::shared_ptr<range_type> range_;
  };


  /// Concrete ShapeIterator whose iteration domain is determined by Predicate
  template <unsigned int DIM, typename Predicate = DensePred<DIM>, typename CS = CoordinateSystem<DIM> >
  class PredShape : public Shape<DIM,CS> {
  public:
    typedef Predicate pred_type;
    typedef CS coordinate_system;
    typedef PredShape<DIM,Predicate,CS> PredShape_;
    typedef typename Shape<DIM,CS>::range_type range_type;
    typedef typename Shape<DIM,CS>::iterator iterator;

    /// Iterator main constructor
    PredShape(const boost::shared_ptr<range_type>& range, pred_type pred = pred_type()) :
        Shape<DIM, CS>(range), pred_(pred) {}

    /// Copy constructor
    PredShape(const PredShape& other) :
        Shape<DIM, CS>(other), pred_(other.pred_) {}

    ~PredShape() {}

    /// Predicate accessor function
    const pred_type& predicate() const {
      return pred_;
    }

    /// Begin accessor function
    iterator begin() const {
      iterator result(this->range()->start_tile(), *this);
      if (! this->includes(*result) )
        this->increment(*result);

      return result;
    }

    /// End accessor function
    iterator end() const {
      return iterator(this->range()->finish_tile(), *this);
    }

    bool includes(const typename iterator::value_type& index) const {
      return pred_(index) && this->range()->includes(index);
    }

  protected:
    virtual void increment(typename iterator::value_type& index) const {
      detail::IncrementCoordinate<DIM,typename range_type::tile_index,CS>(index, this->range()->start_tile(), this->range()->finish_tile());
      while( !includes(index) && index != this->range()->finish_tile() )
        detail::IncrementCoordinate<DIM,typename range_type::tile_index,CS>(index, this->range()->start_tile(), this->range()->finish_tile());
    }

  private:
    pred_type pred_;

  };

} // namespace TiledArray


#endif // SHAPE_H__INCLUDED
