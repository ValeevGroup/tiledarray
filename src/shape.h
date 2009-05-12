#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <range.h>
#include <iterator.h>
#include <predicate.h>
#include <permutation.h>

namespace TiledArray {

  /// Abstract Iterator over a subset of RangeIterator's domain. Example of RangeIterator is Range::tile_iterator.
  template <unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Shape {
  public:
    typedef Shape<DIM,CS> Shape_;
    typedef Range<DIM,CS> range_type;
    typedef CS coordinate_system;
    typedef detail::IndexIterator<typename range_type::tile_index, Shape_> iterator;
    INDEX_ITERATOR_FRIENDSHIP(typename range_type::tile_index, Shape_);

    static unsigned int dim() { return DIM; }


    virtual const boost::shared_ptr<range_type>& range() const =0;

    virtual iterator begin() const =0;
    virtual iterator end() const =0;

    // if this index included in the shape?
    virtual bool includes(const typename iterator::value_type& index) const =0;

  protected:

    virtual void increment(typename iterator::value_type& index) const =0;

  };


  /// Concrete ShapeIterator whose iteration domain is determined by Predicate
  template <unsigned int DIM, typename Predicate = DensePred<DIM>, typename CS = CoordinateSystem<DIM> >
  class PredShape : public Shape<DIM,CS> {
  public:
	typedef PredShape<DIM,Predicate,CS> PredShape_;
    typedef Shape<DIM,CS> Shape_;
    typedef Predicate pred_type;
    typedef CS coordinate_system;
    typedef typename Shape<DIM,CS>::range_type range_type;
    typedef typename Shape<DIM,CS>::iterator iterator;
    INDEX_ITERATOR_FRIENDSHIP(typename range_type::tile_index, Shape_);

    /// Iterator main constructor
    PredShape(const boost::shared_ptr<range_type>& range, pred_type pred = pred_type()) :
    	pred_(pred),  range_(range) {}

    /// Copy constructor
    PredShape(const PredShape& other) :
        range_(other.range_), pred_(other.pred_) {}

    ~PredShape() {}

    const boost::shared_ptr<range_type>& range() const {
      return range_;
    }

    /// Predicate accessor function
    const pred_type& predicate() const {
      return pred_;
    }

    /// Begin accessor function
    iterator begin() const {
      iterator result(range()->start_tile(), this);
      if (! this->includes(*result) )
        this->increment(*result);

      return result;
    }

    /// End accessor function
    iterator end() const {
      return iterator(range()->finish_tile(), this);
    }

    bool includes(const typename iterator::value_type& index) const {
      return pred_(index) && range()->includes(index);
    }

    PredShape_ operator ^=(const Permutation<DIM>& perm) {
      pred_ ^= perm;
      range_ = perm ^ range_;
      return *this;
    }

  protected:
    virtual void increment(typename iterator::value_type& index) const {
      detail::IncrementCoordinate<DIM,typename range_type::tile_index,CS>(index, range()->start_tile(), range()->finish_tile());
      while( !includes(index) && index != this->range()->finish_tile() )
        detail::IncrementCoordinate<DIM,typename range_type::tile_index,CS>(index, range()->start_tile(), range()->finish_tile());
    }

  private:
    pred_type pred_;
    boost::shared_ptr<range_type> range_;

  };

} // namespace TiledArray


#endif // SHAPE_H__INCLUDED
