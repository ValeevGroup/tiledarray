#ifndef ITERATOR_H_
#define ITERATOR_H_

#include <boost/iterator/iterator_facade.hpp>

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

      IndexIterator& operator=(const IndexIterator& other) {
        this->container_ = other.container_;
        this->current_ = other.current_;
        return *this;
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

  } // namespace detail

} // namespace TiledArray

#endif /*ITERATOR_H_*/
