#ifndef ITERATOR_H__INCLUDED
#define ITERATOR_H__INCLUDED

#include <boost/iterator/iterator_facade.hpp>
#include <coordinates.h>

#define INDEX_ITERATOR_FRIENDSHIP(V, C) friend class detail::ArrayIterator< V, C, std::input_iterator_tag>
#define ELEMENT_ITERATOR_FRIENDSHIP(V, C) friend class detail::ArrayIterator< V, C, std::output_iterator_tag>

namespace TiledArray {

  namespace detail {

  /// iterates over an array container
  template <typename Value, typename Container, typename CategoryOrTraversal>
  class ArrayIterator : public boost::iterator_facade<
    ArrayIterator<Value,Container,CategoryOrTraversal>, Value, CategoryOrTraversal >
  {
    typedef ArrayIterator<Value,Container,CategoryOrTraversal> ArrayIterator_;
    typedef boost::iterator_facade<ArrayIterator<Value,Container,CategoryOrTraversal>, Value, CategoryOrTraversal > iterator_facade_;

  public:

    ArrayIterator(const ArrayIterator_& other) :
       container_(other.container_), current_(other.current_) {
    }

    ~ArrayIterator() {}

    ArrayIterator(const Value& cur, const Container* container) :
      container_(container), current_(cur) {
    }

    const Container& container() const {
      return *container_;
    }

  protected:
    friend class boost::iterator_core_access;

    bool equal(const ArrayIterator_ & other) const {
      return current_ == other.current_ && container_ == other.container_;
    }

    // user must provide void Container::increment(index& current) const;
    void increment() {
      container_->increment(current_);
    }

    Value& dereference() const {
      return const_cast<Value&>(current_);
    }

  private:
    ArrayIterator();

    const Container* container_;
    Value current_;
  }; // class ArrayIterator

  /// Array input iterator.
  template <typename Value, typename Container>
  class IndexIterator : public ArrayIterator<Value, Container, std::input_iterator_tag>
  {
  public:
	typedef ArrayIterator<Value, Container, std::input_iterator_tag> ArrayIterator_;
    IndexIterator(const Value& cur, const Container* container) :
      ArrayIterator_(cur, container)
    {}
  private:
    friend class boost::iterator_core_access;

    IndexIterator();
  }; // class IndexIterator

  /// Array output iterator.
  template <typename Value, typename Container>
  class ElementIterator : public ArrayIterator<Value, Container, std::output_iterator_tag>
  {
  public:
	typedef ArrayIterator<Value, Container, std::output_iterator_tag> ArrayIterator_;

	ElementIterator(const Value& cur, const Container* container) :
	  ArrayIterator_(cur, container)
    {}

	template<typename OtherValue>
	ElementIterator(const ElementIterator<OtherValue, Container>& other) :
		ArrayIterator_(other.current_, other.container_)
    {}

  private:
    friend class boost::iterator_core_access;

    template<typename OtherValue>
    bool equal(const ElementIterator<OtherValue, Container>& other) {
      return this->current_ == other.current_ && this->container_ == other.container_;
    }
    ElementIterator();
  }; // class ElementIterator


  template<unsigned int DIM, typename INDEX, typename CS>
  void IncrementCoordinate(INDEX& current, const INDEX& start, const INDEX& finish) {
    assert(current >= start && current < finish);
    // Get order iterators.
    typename DimensionOrder<DIM>::const_iterator order_iter = CS::ordering().begin();
    const typename DimensionOrder<DIM>::const_iterator end_iter = CS::ordering().end();

    // increment least significant, and check to see if the iterator has reached the end
    for(; order_iter != end_iter; ++order_iter) {
      // increment and break if done.
      if( (++(current[*order_iter]) ) < finish[*order_iter])
        return;

      // Reset current index to start value.
      current[*order_iter] = start[*order_iter];
    }

    // Check for end (i.e. current was reset to start)
    if(current == start)
      current = finish;

  }

  } // namespace detail

} // namespace TiledArray

#endif // ITERATOR_H__INCLUDED
