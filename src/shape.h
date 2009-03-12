#ifndef SHAPE_H__INCLUDED
#define SHAPE_H__INCLUDED

#include <cstddef>
#include <stdexcept>
#include <coordinates.h>
#include <range.h>
//#include <predicate.h>
#include <algorithm>
#include <boost/iterator/iterator_adaptor.hpp>
//#include <boost/smart_ptr.hpp>
//#include <boost/iterator/filter_iterator.hpp>

namespace TiledArray {

  /// Abstract Iterator over a subset of RangeIterator's domain. Example of RangeIterator is Range::tile_iterator.
  template <typename RangeIterator>
  class ShapeIterator : public boost::iterator_adaptor<
    ShapeIterator<RangeIterator>,
    RangeIterator> {
  public:
    typedef typename RangeIterator::value_type value_type;

    // no default constructor for iterators over Range, does it even make sense to have one?
    //ShapeIterator() : ShapeIterator::iterator_adaptor_() {}

    ShapeIterator(const RangeIterator& i) : ShapeIterator::iterator_adaptor_(i) {}

    virtual ~ShapeIterator() {}

    // if this index included in the shape?
    virtual bool includes(const value_type& index) const =0;

  private:
	friend class boost::iterator_core_access;
  };


  /// Concrete ShapeIterator whose iteration domain is determined by Predicate
  template <typename RangeIterator, typename Predicate>
  class PredShapeIterator : public ShapeIterator<RangeIterator> {
  public:
    typedef ShapeIterator<RangeIterator> base_type;
    typedef typename base_type::value_type value_type;

    /// Iterator main constructor
    PredShapeIterator(RangeIterator it, RangeIterator end, Predicate pred = Predicate()) :
        base_type(it), pred_(pred), end_(end)
    {
      if(!pred_(* this->base_reference()))
        this->base_reference() = end_;
    }

    /// Copy constructor
    PredShapeIterator(const PredShapeIterator& it) :
        base_type(it.base()), pred_(it.pred_), end_(it.end_)
    { }

    virtual ~PredShapeIterator() {}

    /// Predicate accessor function
    const Predicate& predicate() const {
      return pred_;
    }

    /// End accessor function
    const RangeIterator& end() const {
      return end_;
    }

    virtual bool includes(const value_type& index) const {
      return pred_(index);
    }

    bool operator==(const RangeIterator& other) const {
      return this->base() == other;
    }

  private:
    Predicate pred_;
    RangeIterator end_;

    friend class boost::iterator_core_access;
    // implements iterator_adaptor::increment
    void increment() {
      RangeIterator& it = this->base_reference();
      ++it;
      while( !pred_(*it) && it != end_ )
        ++it;
    }

  };

  template <typename RangeIterator, typename Predicate>
  bool operator !=(const PredShapeIterator<RangeIterator, Predicate>& PredIt, const RangeIterator& It) {
    return ! (PredIt == It);
  }

  template <typename RangeIterator, typename Predicate>
  bool operator ==(const RangeIterator& It, const PredShapeIterator<RangeIterator, Predicate>& PredIt) {
    return PredIt == It;
  }

  template <typename RangeIterator, typename Predicate>
  bool operator !=(const RangeIterator& It, const PredShapeIterator<RangeIterator, Predicate>& PredIt) {
    return PredIt != It;
  }


} // namespace TiledArray


#endif // SHAPE_H__INCLUDED
