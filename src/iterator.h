#ifndef ITERATOR_H_
#define ITERATOR_H_

#include <boost/iterator/iterator_facade.hpp>
#include <coordinates.h>

namespace TiledArray {

  namespace detail {

  /// iterates over indices to a container
  template <typename Value, typename Container>
  class IndexIterator : public boost::iterator_facade<
    IndexIterator<Value,Container>, Value, std::input_iterator_tag >
  {
      typedef IndexIterator<Value,Container> my_type;
    public:
      IndexIterator(const IndexIterator& other) :
         container_(other.container_), current_(other.current_) {
      }

      ~IndexIterator() {
      }

      IndexIterator(const Value& cur, const Container& container) :
        container_(&container), current_(cur) {
      }

      const Container& container() const {
    	  return *container_;
      }

    private:
      friend class boost::iterator_core_access;

      bool equal(my_type const& other) const {
        return current_ == other.current_ && container_ == other.container_;
      }

      // user must provide container_->increment(current_) function
      // void increment(index& current) const;
      void increment() {
        container_->increment(current_);
      }

      Value& dereference() const {
        return const_cast<Value&>(current_);
      }

      IndexIterator();

      const Container* container_;
      Value current_;
  };

  template<unsigned int DIM, typename Coord, typename CS>
  void IncrementCoordinate(Coord& current, const Coord& start, const Coord& finish) {
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

#endif /*ITERATOR_H_*/
