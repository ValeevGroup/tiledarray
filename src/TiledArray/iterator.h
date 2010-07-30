#ifndef TILEDARRAY_ITERATOR_H__INCLUDED
#define TILEDARRAY_ITERATOR_H__INCLUDED

//#include <TiledArray/coordinate_system.h>
#include <boost/iterator/iterator_facade.hpp>

namespace TiledArray {

  namespace detail {

    /// Input iterator used to iterate over multidimensional indices
    template <typename Value, typename Container>
    class IndexIterator : public boost::iterator_facade<
        IndexIterator<Value,Container>, Value, std::input_iterator_tag >
    {
    public:
      typedef IndexIterator<Value,Container> IndexIterator_;
      typedef boost::iterator_facade<IndexIterator_, Value, std::input_iterator_tag > iterator_facade_;

      IndexIterator(const IndexIterator_& other) :
        container_(other.container_), current_(other.current_) {
      }

      ~IndexIterator() {}

      IndexIterator(const Value& cur, const Container* container) :
        container_(container), current_(cur) {
      }

    protected:
      friend class boost::iterator_core_access;

      bool equal(const IndexIterator_ & other) const {
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
      IndexIterator();

      const Container* container_;
      Value current_;
    }; // class IndexIterator

  } // namespace detail

} // namespace TiledArray

#endif // TILEDARRAY_ITERATOR_H__INCLUDED
