#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <coordinates.h>
#include <range.h>
#include <iterator.h>
#include <predicate.h>

namespace TiledArray {

  /// Abstract Iterator over a subset of RangeIterator's domain. Example of RangeIterator is Range::tile_iterator.
  template <typename Range>
  class Shape {
  public:
    typedef detail::IndexIterator<typename Range::tile_index,Shape> Iterator;
    Shape(const Range& range) : range_(&range) {}
    Shape(const Shape& other) : range_(other.range_) {}
    virtual ~Shape() {}

    const Range* range() const { return range_; }

    virtual Iterator begin() const =0;
    virtual Iterator end() const =0;

    // if this index included in the shape?
    virtual bool includes(const typename Iterator::value_type& index) const =0;
    virtual void increment(typename Iterator::value_type& it) const =0;

  private:
    const Range* range_;

  };


  /// Concrete ShapeIterator whose iteration domain is determined by Predicate
  template <typename Range, typename Predicate>
  class PredShape : public Shape<Range> {
  public:
    typedef Shape<Range> Shape;
    typedef typename Shape::Iterator Iterator;

    /// Iterator main constructor
    PredShape(const Range& range, Predicate pred = Predicate()) :
        Shape(range), pred_(pred) {}

    /// Copy constructor
    PredShape(const PredShape& other) :
        Shape(other), pred_(other.pred_) {}

    ~PredShape() {}

    /// Predicate accessor function
    const Predicate& predicate() const {
      return pred_;
    }

    /// Begin accessor function
    Iterator begin() const {
      Iterator result(this->range()->start_tile(), *this);
      if (!includes(*result) ) {
        this->increment(*result);
      }
      return result;
    }
    /// End accessor function
    Iterator end() const {
      return Iterator(this->range()->finish_tile(), *this);
    }

    bool includes(const typename Iterator::value_type& index) const {
      return pred_(index);
    }

  private:
    Predicate pred_;

    void increment(typename Iterator::value_type& index) const {
      this->range()->increment(index);
      while( !includes(index) && index != this->range()->finish_tile() )
        this->range()->increment(index);
    }

  };

} // namespace TiledArray


#endif // SHAPE_H__INCLUDED
